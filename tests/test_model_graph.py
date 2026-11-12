"""Independent CPU algebra checks for the native FX program and its derivatives."""
import copy
import json
import sys
import types

import pytest

torch = pytest.importorskip("torch")
from openassembler.modelfront import ModelGraphError, append_training, capture_model, extract_weights, flatten_outputs


def execute_steps(plan, steps, state):
    """Reference interpreter for the serialized program; never a GPU fallback."""
    for step in steps:
        op, names, out, attrs = step["kernel"], step["inputs"], step["output"], step["attrs"]
        xs = [state[name] for name in names]
        dims = plan["buffers"][out]["shape"]
        if op == "linear":
            value = xs[0] @ xs[1].t()
            if len(xs) == 3:
                value = value + xs[2]
        elif op == "linear_dx":
            value = xs[0] @ xs[1]
        elif op == "linear_dw":
            value = xs[0].t() @ xs[1]
        elif op == "bmm":
            value = xs[0] @ xs[1]
        elif op == "bmm_dx":
            value = xs[0] @ xs[1].transpose(-2, -1)
        elif op == "bmm_dw":
            value = xs[1].transpose(-2, -1) @ xs[0]
        elif op == "bias_grad":
            value = xs[0].sum(dim=0)
        elif op == "relu":
            value = torch.relu(xs[0])
        elif op == "relu_backward":
            value = xs[0] * (xs[1] > 0)
        elif op in ("add", "mul"):
            rhs = xs[1] if len(xs) == 2 else attrs["scalar"]
            value = xs[0] + rhs if op == "add" else xs[0] * rhs
        elif op == "copy":
            value = xs[0].reshape(dims).contiguous().clone()
        elif op == "transpose2d":
            value = xs[0].t().contiguous()
        elif op == "permute":
            value = xs[0].permute(attrs["axes"]).contiguous()
        elif op == "softmax":
            value = torch.softmax(xs[0], dim=-1)
        elif op == "softmax_backward":
            value = xs[1] * (xs[0] - (xs[0] * xs[1]).sum(dim=-1, keepdim=True))
        elif op == "gelu":
            value = torch.nn.functional.gelu(xs[0], approximate="tanh")
        elif op == "gelu_backward":
            import math
            x = xs[1]
            t = torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715*x*x*x))
            value = xs[0] * (0.5*(1+t) + 0.5*x*(1-t*t)*math.sqrt(2/math.pi)*(1+3*0.044715*x*x))
        elif op == "layernorm":
            value = torch.nn.functional.layer_norm(xs[0], (xs[0].shape[-1],), eps=attrs["epsilon"])
        elif op == "layernorm_backward":
            dy, x = xs
            center = x-x.mean(-1, keepdim=True)
            inv = torch.rsqrt((center*center).mean(-1, keepdim=True)+attrs["epsilon"])
            norm = center*inv
            value = inv*(dy-dy.mean(-1, keepdim=True)-norm*(dy*norm).mean(-1, keepdim=True))
        elif op in ("channel_affine", "channel_mul"):
            value = xs[0]*xs[1]
            if len(xs) == 3:
                value = value+xs[2]
        elif op == "channel_grad":
            value = (xs[0]*xs[1]).reshape(-1, xs[0].shape[-1]).sum(0)
        elif op == "channel_sum":
            value = xs[0].reshape(-1, xs[0].shape[-1]).sum(0)
        elif op == "concat":
            value = torch.cat(xs, dim=attrs["axis"])
        elif op == "slice_axis":
            value = xs[0].narrow(attrs["axis"], attrs["start"], dims[attrs["axis"]]).contiguous()
        elif op == "mse":
            value = ((xs[0] - xs[1]) ** 2).mean().reshape(1)
        elif op == "mse_grad":
            value = 2 * (xs[0] - xs[1]) / xs[0].numel() * xs[2]
        elif op == "sgd":
            value = xs[0] - attrs["lr"] * xs[1]
        else:
            raise AssertionError(f"unrecognized native test operation {op}")
        assert list(value.shape) == dims
        state[out] = value


def reference_run(plan, weights, inputs, seeds=None):
    state = {name: value.detach().clone() for name, value in weights.items()}
    state.update({spec["name"]: value.detach().clone() for spec, value in zip(plan["inputs"], inputs)})
    with torch.no_grad():
        execute_steps(plan, plan["steps"], state)
        for output, buffer in plan["backward"]["seed_grads"].items():
            state[buffer] = torch.ones_like(state[output]) if seeds is None else seeds[output]
        execute_steps(plan, plan["backward"]["steps"], state)
    return state


def assert_matches_autograd(model, inputs):
    plan = capture_model(model, inputs)
    json.loads(json.dumps(plan))
    actual = reference_run(plan, extract_weights(model, plan), inputs)
    reference_inputs = tuple(x.detach().clone().requires_grad_() for x in inputs)
    output = model(*reference_inputs)
    assert isinstance(output, torch.Tensor)
    torch.testing.assert_close(actual[plan["outputs"][0]], output)
    output.sum().backward()
    for spec, value in zip(plan["inputs"], reference_inputs):
        name = plan["backward"]["grad_inputs"].get(spec["name"])
        if value.grad is None:
            assert name is None
        else:
            torch.testing.assert_close(actual[name], value.grad)
    for spec in plan["parameters"]:
        value = model.get_parameter(spec["target"])
        name = plan["backward"]["grad_parameters"].get(spec["name"])
        if not value.requires_grad:
            assert name is None
        else:
            torch.testing.assert_close(actual[name], value.grad)
    return plan


def test_linear_relu_forward_and_all_gradients():
    torch.manual_seed(4)
    model = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.ReLU(), torch.nn.Linear(7, 3))
    plan = assert_matches_autograd(model, (torch.randn(4, 5),))
    assert [s["kernel"] for s in plan["steps"]] == ["linear", "relu", "linear"]
    assert len(plan["backward"]["grad_parameters"]) == 4
    assert plan["contract"]["fallback"] == "none"


def test_branches_shared_parameters_and_scalar_derivatives():
    class Shared(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(4, 4)
            self.tied = self.projection

        def forward(self, x):
            a = self.projection(x)
            b = self.tied(x)
            return (a * b + a) * 0.25 + 2.0 + x

    plan = assert_matches_autograd(Shared(), (torch.randn(3, 4),))
    assert len(plan["parameters"]) == 2
    assert any(step["kernel"] == "add" for step in plan["backward"]["steps"])


def test_batched_linear_and_specialized_reshape_are_correct():
    class Batched(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 6, bias=False)

        def forward(self, x):
            return self.linear(x).reshape(x.shape[0], -1).transpose(0, 1).contiguous()

    plan = assert_matches_autograd(Batched(), (torch.randn(2, 3, 4),))
    assert sum(s["kernel"] == "copy" for s in plan["steps"]) >= 3
    assert any(s["kernel"] == "transpose2d" for s in plan["steps"])


def test_input_weight_and_multiple_inputs_receive_derivatives():
    class Functional(torch.nn.Module):
        def forward(self, x, weight):
            return torch.nn.functional.linear(x, weight) * 2

    assert_matches_autograd(Functional(), (torch.randn(2, 3), torch.randn(5, 3)))


def test_frozen_parameter_is_not_updated_but_input_gradient_exists():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2))
    model[0].weight.requires_grad_(False)
    plan = assert_matches_autograd(model, (torch.randn(2, 3),))
    assert "parameter:0.weight" not in plan["backward"]["grad_parameters"]
    assert "input_1" in plan["backward"]["grad_inputs"]


def test_output_structure_preserves_repeated_tensor_and_nested_containers():
    class Outputs(torch.nn.Module):
        def forward(self, x):
            y = torch.relu(x)
            return {"first": y, "other": (x, [y])}

    model, x = Outputs(), torch.randn(2, 3)
    plan = capture_model(model, (x,))
    assert len(plan["outputs"]) == 2
    assert plan["output_structure"]["dict"]["first"]["tensor"] == "relu"
    assert plan["output_structure"]["dict"]["other"]["tuple"][1]["list"] == [{"tensor": "relu"}]
    assert set(flatten_outputs(plan, model(x))) == set(plan["outputs"])
    with pytest.raises(ModelGraphError, match="dictionary differs"):
        flatten_outputs(plan, {"first": x})
    with pytest.raises(ModelGraphError, match="no longer aliases"):
        flatten_outputs(plan, {"first": x, "other": (x, [x.clone()])})


def test_native_mse_backward_and_sgd_match_cpu_reference():
    torch.manual_seed(12)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    x, target = torch.randn(5, 3), torch.randn(5, 2)
    base = capture_model(model, (x,))
    plan = append_training(base, learning_rate=0.03)
    state = reference_run(plan, extract_weights(model, plan), (x, target))
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    torch.testing.assert_close(state["_oa_loss"].reshape(()), loss)
    for spec in plan["parameters"]:
        value = model.get_parameter(spec["target"])
        torch.testing.assert_close(state[plan["backward"]["grad_parameters"][spec["name"]]], value.grad)
    execute_steps(plan, plan["training"]["steps"], state)
    for spec in plan["parameters"]:
        value = model.get_parameter(spec["target"])
        torch.testing.assert_close(state[spec["name"]], value - 0.03 * value.grad)
    assert "training" not in base


@pytest.mark.parametrize("bad", [torch.nn.Sigmoid(), torch.nn.Dropout()])
def test_unsupported_operator_fails_with_fx_node(bad):
    with pytest.raises(ModelGraphError, match="unsupported native model node"):
        capture_model(torch.nn.Sequential(bad), (torch.randn(2, 3),))


def test_broadcast_mutation_dtype_and_tensor_slices_fail_closed():
    class Broadcast(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    class Slice(torch.nn.Module):
        def forward(self, x):
            return x[:, 0]

    class Inplace(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.relu(x, True)

    with pytest.raises(ModelGraphError, match="broadcasting"):
        capture_model(Broadcast(), (torch.randn(2, 3), torch.randn(3)))
    with pytest.raises(ModelGraphError, match="in-place ReLU"):
        capture_model(torch.nn.Sequential(torch.nn.ReLU(inplace=True)), (torch.randn(2, 3),))
    with pytest.raises(ModelGraphError, match="in-place ReLU"):
        capture_model(Inplace(), (torch.randn(2, 3),))
    with pytest.raises(ModelGraphError, match="FP32"):
        capture_model(torch.nn.Identity(), (torch.ones(2, 3, dtype=torch.float16),))
    with pytest.raises(ModelGraphError, match="slicing/indexing"):
        capture_model(Slice(), (torch.randn(2, 3),))


def test_autograd_bridge_partial_outputs_and_multiple_pending_forwards(monkeypatch, tmp_path):
    from openassembler.modelfront.autograd import CompiledModel

    class Outputs(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)

        def forward(self, x):
            a = self.linear(x)
            return a, a * 2

    class CPUReferenceRuntime:
        def __init__(self, plan, weights, build_dir):
            self.plan, self.weights = plan, weights

        def forward(self, inputs, require_tape=True):
            self.state = {**self.weights, **inputs}
            execute_steps(self.plan, self.plan["steps"], self.state)
            return {name: self.state[name] for name in self.plan["outputs"]}

        def backward(self, seeds, zero_missing=False):
            assert zero_missing
            for output, buf in self.plan["backward"]["seed_grads"].items():
                self.state[buf] = seeds.get(output, torch.zeros_like(self.state[output]))
            execute_steps(self.plan, self.plan["backward"]["steps"], self.state)
            return {"inputs": {k: self.state[v] for k, v in self.plan["backward"]["grad_inputs"].items()},
                    "parameters": {k: self.state[v] for k, v in self.plan["backward"]["grad_parameters"].items()}}

    monkeypatch.setitem(sys.modules, "openassembler.modelfront.runtime", types.SimpleNamespace(Runtime=CPUReferenceRuntime))
    original = Outputs()
    plan = capture_model(original, (torch.randn(2, 3),))
    compiled = CompiledModel(plan, extract_weights(original, plan), tmp_path)
    a, b = torch.randn(2, 3, requires_grad=True), torch.randn(2, 3, requires_grad=True)
    out_a, out_b = compiled(a), compiled(b)
    torch.autograd.backward((out_a[0], out_b[1]), (torch.ones_like(out_a[0]), torch.ones_like(out_b[1])))
    reference = copy.deepcopy(original)
    expected_a = a.detach().clone().requires_grad_()
    expected_b = b.detach().clone().requires_grad_()
    (reference(expected_a)[0].sum() + reference(expected_b)[1].sum()).backward()
    torch.testing.assert_close(a.grad, expected_a.grad)
    torch.testing.assert_close(b.grad, expected_b.grad)
    for actual, expected in zip(compiled.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad)


def test_captured_graph_passes_native_registry_contracts():
    from openassembler.modelfront.kernels import prepare

    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    plan = capture_model(model, (torch.randn(5, 3),))
    specs, steps, kernels = prepare(plan)
    assert len(steps) == len(plan["steps"]) + len(plan["backward"]["steps"])
    assert kernels and all(step["entry"] in kernels for step in steps)
    prepared_training = append_training(plan)
    prepare(prepared_training)


def test_native_train_step_owns_update_and_reuses_runtime(monkeypatch, tmp_path):
    from openassembler.modelfront.autograd import CompiledModel

    class CPUReferenceRuntime:
        created = 0

        def __init__(self, plan, weights, build_dir):
            self.plan, self.weights = plan, weights
            self.closed = False
            type(self).created += 1

        def train_step(self, inputs, target, lr):
            assert not torch.is_grad_enabled()
            plan = append_training(self.plan, learning_rate=lr)
            state = reference_run(plan, self.weights, [*inputs.values(), target])
            gradients = {name: state[value] for name, value in plan["backward"]["grad_parameters"].items()}
            execute_steps(plan, plan["training"]["steps"], state)
            for name, tensor in self.weights.items():
                tensor.copy_(state[name])
            return {"loss": state["_oa_loss"], "gradients": {"parameters": gradients}}

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "openassembler.modelfront.runtime", types.SimpleNamespace(Runtime=CPUReferenceRuntime))
    original = torch.nn.Sequential(torch.nn.Linear(3, 2))
    x, target = torch.randn(4, 3), torch.randn(4, 2)
    plan = capture_model(original, (x,))
    compiled = CompiledModel(plan, extract_weights(original, plan), tmp_path)
    reference = copy.deepcopy(original)
    for _ in range(2):
        reference.zero_grad(set_to_none=True)
        expected = torch.nn.functional.mse_loss(reference(x), target)
        expected.backward()
        with torch.no_grad():
            for p in reference.parameters():
                p.add_(p.grad, alpha=-0.05)
        actual = compiled.native_train_step(x, target, learning_rate=0.05)
        torch.testing.assert_close(actual["loss"].reshape(()), expected)
        for a, b in zip(compiled.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b)
            torch.testing.assert_close(a.grad, b.grad)
    assert CPUReferenceRuntime.created == 1
    runtime = compiled._native_runtime
    compiled.to("cpu")
    assert runtime.closed and compiled._native_runtime is None


def test_cpu_mathematical_control_agrees_with_original_fp32_autograd():
    from openassembler.modelfront.reference import CPUReference

    model = torch.nn.Sequential(torch.nn.Linear(3, 7), torch.nn.ReLU(), torch.nn.Linear(7, 2))
    x, target = torch.randn(5, 3), torch.randn(5, 2)
    plan = capture_model(model, (x,))
    reference = CPUReference(plan, extract_weights(model, plan))
    x = x.requires_grad_()
    expected_loss = torch.nn.functional.mse_loss(model(x), target)
    expected_loss.backward()
    actual = reference.train_step({plan["inputs"][0]["name"]: x}, target, lr=0.02)
    torch.testing.assert_close(actual["loss"], expected_loss)
    torch.testing.assert_close(actual["gradients"]["inputs"][plan["inputs"][0]["name"]], x.grad)
    for spec in plan["parameters"]:
        value = model.get_parameter(spec["target"])
        torch.testing.assert_close(actual["gradients"]["parameters"][spec["name"]], value.grad)
        torch.testing.assert_close(reference.weights[spec["name"]], value - 0.02 * value.grad)


def test_explicit_half_contract_has_independent_forward_and_backward_rounding():
    from openassembler.modelfront.reference import CPUReference

    model = torch.nn.Sequential(torch.nn.Linear(3, 2))
    x, dy = torch.randn(5, 3), torch.randn(5, 2)
    plan = capture_model(model, (x,), math_mode="f16_rne_f32_accum")
    weights = extract_weights(model, plan)
    reference = CPUReference(plan, weights)
    outputs = reference.forward({plan["inputs"][0]["name"]: x})
    name = plan["outputs"][0]
    weight, bias = model[0].weight.detach(), model[0].bias.detach()
    round_half = lambda z: z.half().float()
    torch.testing.assert_close(outputs[name], round_half(x) @ round_half(weight).t() + bias)
    gradients = reference.backward({name: dy})
    torch.testing.assert_close(gradients["inputs"][plan["inputs"][0]["name"]], round_half(dy) @ round_half(weight))
    torch.testing.assert_close(gradients["parameters"]["parameter:0.weight"], round_half(dy).t() @ round_half(x))
    torch.testing.assert_close(gradients["parameters"]["parameter:0.bias"], dy.sum(0))
    assert plan["contract"]["matrix_operand_rounding"] == "round_to_nearest_even"
    with pytest.raises(ModelGraphError, match="overflows"):
        reference.forward({plan["inputs"][0]["name"]: torch.full_like(x, 1e10)})


def test_state_mutation_overlapping_parameters_and_nonfinite_capture_fail():
    class Mutates(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("offset", torch.zeros(2, 3))

        def forward(self, x):
            self.offset.add_(1)
            return x + self.offset

    class Overlap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            raw = torch.randn(3, 3)
            self.first = torch.nn.Parameter(raw)
            self.second = torch.nn.Parameter(raw)

        def forward(self, x):
            return x + self.first + self.second

    model = Mutates()
    with pytest.raises(ModelGraphError, match="state mutation"):
        capture_model(model, (torch.ones(2, 3),))
    assert torch.equal(model.offset, torch.zeros_like(model.offset))
    with pytest.raises(ModelGraphError, match="overlapping storage"):
        capture_model(Overlap(), (torch.randn(3, 3),))
    with pytest.raises(ModelGraphError, match="finite FP32"):
        capture_model(torch.nn.Identity(), (torch.full((2, 3), float("nan")),))


def test_matrix_products_and_rank_four_attention_shapes():
    class Attention(torch.nn.Module):
        def forward(self, q, k, v):
            score = torch.matmul(q, k.transpose(-2, -1)) * 0.25
            probability = torch.softmax(score, dim=-1)
            return probability @ v

    plan = assert_matches_autograd(Attention(), (torch.randn(2, 3, 5, 4), torch.randn(2, 3, 7, 4),
                                                torch.randn(2, 3, 7, 6)))
    from openassembler.modelfront.kernels import prepare
    prepare(plan)
    assert sum(s["kernel"] == "bmm" for s in plan["steps"]) == 2


def test_rank_two_matmul_and_tanh_gelu_derivatives():
    class Products(torch.nn.Module):
        def forward(self, a, b):
            return torch.nn.functional.gelu(torch.mm(a, b), approximate="tanh")

    plan = assert_matches_autograd(Products(), (torch.randn(3, 5), torch.randn(5, 7)))
    assert any(s["kernel"] == "gelu_backward" for s in plan["backward"]["steps"])
    with pytest.raises(ModelGraphError, match="approximate='tanh'"):
        capture_model(torch.nn.Sequential(torch.nn.GELU()), (torch.randn(2, 3),))


def test_batched_product_registry_shapes_and_launches():
    from openassembler.modelfront.kernels import emit_kernel

    spec = lambda shape: {"shape": list(shape), "dtype": "f32"}
    for op, ins, out in [("bmm", [(2, 3, 5), (2, 5, 7)], (2, 3, 7)),
                         ("bmm_dx", [(2, 3, 7), (2, 5, 7)], (2, 3, 5)),
                         ("bmm_dw", [(2, 3, 7), (2, 3, 5)], (2, 5, 7))]:
        kernel = emit_kernel(op, [spec(s) for s in ins], spec(out))
        assert kernel.grid == (1, out[1], 2)
        assert "div." not in kernel.ptx and "rem." not in kernel.ptx
    permutation = emit_kernel("permute", [spec((2, 3, 5, 7))], spec((2, 5, 3, 7)), {"axes": [0, 2, 1, 3]})
    assert permutation.grid == (1, 3, 10)
    with pytest.raises(ValueError, match="shapes disagree"):
        emit_kernel("bmm", [spec((2, 3, 5)), spec((1, 5, 7))], spec((2, 3, 7)))


@pytest.mark.parametrize("affine,bias", [(True, True), (True, False), (False, False)])
def test_complete_layernorm_forward_and_affine_derivatives(affine, bias):
    model = torch.nn.Sequential(torch.nn.LayerNorm(7, eps=0.0003, elementwise_affine=affine, bias=bias),
                                torch.nn.Linear(7, 3))
    assert_matches_autograd(model, (torch.randn(2, 3, 7),))


def test_functional_layernorm_preserves_epsilon_and_rejects_multi_axis():
    class Norm(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.layer_norm(x, [7], eps=0.003)

    plan = assert_matches_autograd(Norm(), (torch.randn(2, 3, 7),))
    assert next(s for s in plan["steps"] if s["kernel"] == "layernorm")["attrs"] == {"epsilon": 0.003}
    with pytest.raises(ModelGraphError, match="last-dimension"):
        capture_model(torch.nn.Sequential(torch.nn.LayerNorm([3, 7])), (torch.randn(2, 3, 7),))


@pytest.mark.parametrize("axis", [0, 1, -1, -2])
def test_concatenation_and_symbolic_slice_gradients(axis):
    class Concat(torch.nn.Module):
        def forward(self, x, y, z):
            return torch.cat((x, y, z), dim=axis)

    dims = [2, 3, 4, 5]
    shapes = []
    for width in (1, 2, 3):
        shape = dims[:]
        shape[axis] = width
        shapes.append(shape)
    plan = assert_matches_autograd(Concat(), tuple(torch.randn(shape) for shape in shapes))
    from openassembler.modelfront.kernels import prepare
    prepare(plan)
    assert sum(s["kernel"] == "slice_axis" for s in plan["backward"]["steps"]) == 3


def test_storage_views_preserve_branch_outputs_backward_and_clone_ownership():
    from openassembler.modelfront.optimization import optimize_plan

    class Views(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 4)

        def forward(self, x):
            a = self.layer(x).reshape(2, 3, 4)
            independent = a.clone()
            return (a * a + independent).flatten(0, 1)

    model = Views()
    x = torch.randn(6, 4)
    base = capture_model(model, (x,))
    optimized = optimize_plan(base)
    assert len(optimized["aliases"]) >= 2
    clone_step = next(s for s in optimized["steps"] if s["kernel"] == "copy" and not s.get("view"))
    assert clone_step["output"] not in optimized["aliases"]
    weights = extract_weights(model, base)
    expected = reference_run(base, weights, (x,))
    state = {**weights, optimized["inputs"][0]["name"]: x}
    for output, seed in optimized["backward"]["seed_grads"].items():
        state[seed] = torch.ones(optimized["buffers"][output]["shape"])
    with torch.no_grad():
        for step in [*optimized["steps"], *optimized["backward"]["steps"]]:
            if step["output"] in optimized["aliases"]:
                state[step["output"]] = state[step["inputs"][0]].view(optimized["buffers"][step["output"]]["shape"])
            else:
                execute_steps(optimized, [step], state)
    for name in [*base["outputs"], *base["backward"]["grad_inputs"].values(),
                 *base["backward"]["grad_parameters"].values()]:
        assert torch.equal(state[name], expected[name])
    assert state[clone_step["output"]].data_ptr() != state[clone_step["inputs"][0]].data_ptr()


def test_public_compile_model_applies_optional_storage_views(monkeypatch, tmp_path):
    from openassembler.modelfront.autograd import compile_model

    built = []
    monkeypatch.setitem(sys.modules, "openassembler.modelfront.runtime", types.SimpleNamespace(
        build_bundle=lambda plan, weights, path, **kwargs: built.append(plan)))

    class Reshape(torch.nn.Module):
        def forward(self, x):
            return x.reshape(2, 6).clone()

    compile_model(Reshape(), torch.ones(3, 4), tmp_path)
    assert built[-1]["aliases"]
    compile_model(Reshape(), torch.ones(3, 4), tmp_path, storage_views=False)
    assert "aliases" not in built[-1]
    compile_model(Reshape(), torch.ones(3, 4), tmp_path, reuse_storage=True)
    from openassembler.modelfront.optimization import validate_storage
    assert validate_storage(built[-1])['policy'] == 'exact-size-liveness-v1'


def test_native_forward_reuses_storage_refreshes_parameters_and_restores_outputs(monkeypatch, tmp_path):
    from openassembler.modelfront.autograd import CompiledModel

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2)

        def forward(self, x):
            y = self.linear(x)
            return {"prediction": y, "nested": (y, [y * 2])}

    class StorageRuntime:
        created = 0

        def __init__(self, plan, weights, build_dir):
            type(self).created += 1
            self.plan, self.weights = plan, weights
            self.storage = {name: torch.empty(spec["shape"]) for name, spec in plan["buffers"].items()}
            self._tape = None
            self.closed = False

        def forward(self, inputs, require_tape=False):
            assert not torch.is_grad_enabled() and require_tape is False
            state = {**self.weights, **inputs}
            for step in self.plan["steps"]:
                execute_steps(self.plan, [step], state)
                self.storage[step["output"]].copy_(state[step["output"]])
                state[step["output"]] = self.storage[step["output"]]
            return {name: state[name] for name in self.plan["outputs"]}

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "openassembler.modelfront.runtime", types.SimpleNamespace(Runtime=StorageRuntime))
    original = Model()
    x = torch.randn(4, 3, requires_grad=True)
    plan = capture_model(original, (x,))
    compiled = CompiledModel(plan, extract_weights(original, plan), tmp_path)
    result = compiled.native_forward(x)
    ptr = result["prediction"].data_ptr()
    assert result["prediction"] is result["nested"][0]
    assert result["prediction"].grad_fn is None
    torch.testing.assert_close(result["prediction"], original(x)["prediction"])
    # Replace a bound Parameter object, not only its values. The second call
    # must read the current module binding without loading another runtime.
    spec = next(s for s in plan["parameters"] if s["target"] == "linear.bias")
    attr = compiled._state_attributes[compiled._state_names.index(spec["name"])]
    replacement = torch.nn.Parameter(torch.full((2,), 0.2))
    setattr(compiled, attr, replacement)
    with torch.no_grad():
        original.linear.bias.copy_(replacement)
    x2 = torch.randn(4, 3)
    result = compiled.native_forward({plan["inputs"][0]["name"]: x2})
    assert result["prediction"].data_ptr() == ptr and StorageRuntime.created == 1
    torch.testing.assert_close(result["prediction"], original(x2)["prediction"])
    torch.testing.assert_close(result["nested"][1][0], original(x2)["nested"][1][0])
    runtime = compiled._native_runtime
    runtime._tape = {"outstanding": True}
    with pytest.raises(RuntimeError, match="outstanding backward tape"):
        compiled.native_forward(x2)
    runtime._tape = None
    compiled.close()
    assert runtime.closed and compiled._native_runtime is None
    with pytest.raises(ModelGraphError, match="input names"):
        compiled.native_forward({"wrong": x2})
    with pytest.raises(ModelGraphError, match="expects 1"):
        compiled.native_forward((x, x2))


def test_persistent_native_calls_reject_live_autograd_tape(monkeypatch, tmp_path):
    from openassembler.modelfront.autograd import CompiledModel

    class TapeRuntime:
        def __init__(self, plan, weights, build_dir):
            self.plan, self.weights = plan, weights

        def forward(self, inputs, require_tape=False):
            return {self.plan["outputs"][0]: next(iter(inputs.values())).clone()}

        def backward(self, seeds, zero_missing=False):
            return {"inputs": {self.plan["inputs"][0]["name"]: next(iter(seeds.values()))}, "parameters": {}}

    monkeypatch.setitem(sys.modules, "openassembler.modelfront.runtime", types.SimpleNamespace(Runtime=TapeRuntime))
    model = torch.nn.Sequential(torch.nn.ReLU())
    x = torch.randn(2, 3, requires_grad=True)
    plan = capture_model(model, (x,))
    compiled = CompiledModel(plan, {}, tmp_path)
    output = compiled(x)
    with pytest.raises(RuntimeError, match="outstanding autograd tape"):
        compiled.native_forward(x)
    with pytest.raises(RuntimeError, match="outstanding autograd tape"):
        compiled.native_train_step(x, torch.zeros_like(x))
    output.backward(torch.ones_like(output))
    assert not compiled._autograd_runtimes
    torch.testing.assert_close(compiled.native_forward(x), x)
