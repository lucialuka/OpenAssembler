"""Exact-shape FP32 convolution forward and first-order gradients."""
import pytest

torch = pytest.importorskip("torch")
from openassembler.modelfront.graph import ModelGraphError, capture_model, extract_weights
from openassembler.modelfront.kernels import emit_kernel, prepare
from openassembler.modelfront.optimization import optimize_plan
from openassembler.modelfront.reference import CPUReference


class ConvModel(torch.nn.Module):
    def __init__(self, *, bias=True, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, 3, stride=stride, padding=padding,
                                    dilation=dilation, bias=bias)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        return self.activation(self.conv(x))


@pytest.mark.parametrize("shape,stride,padding,dilation", [
    ((1, 2, 5, 6), 1, 1, 1),
    ((2, 2, 7, 8), 2, 1, 1),
    ((1, 2, 8, 9), 1, 2, 2),
])
@pytest.mark.parametrize("bias", [False, True])
def test_conv2d_forward_and_all_parameter_gradients(shape, stride, padding, dilation, bias):
    torch.manual_seed(73571)
    model = ConvModel(bias=bias, stride=stride, padding=padding, dilation=dilation)
    x = torch.randn(shape, requires_grad=True)
    plan = optimize_plan(capture_model(model, (x,)))
    prepare(plan)
    reference = CPUReference(plan, extract_weights(model, plan))
    actual = reference.forward((x,))
    expected = model(x)
    torch.testing.assert_close(actual[plan["outputs"][0]], expected.detach(), rtol=2e-5, atol=2e-6)
    seed = torch.randn_like(expected)
    expected.backward(seed)
    gradients = reference.backward({plan["outputs"][0]: seed})
    torch.testing.assert_close(gradients["inputs"][plan["inputs"][0]["name"]], x.grad,
                               rtol=3e-5, atol=3e-6)
    for spec in plan["parameters"]:
        parameter = dict(model.named_parameters())[spec["target"]]
        torch.testing.assert_close(gradients["parameters"][spec["name"]], parameter.grad,
                                   rtol=3e-5, atol=3e-6)
    convolution_steps = [step for step in plan["steps"] + plan["backward"]["steps"]
                         if step["kernel"].startswith("conv2d")]
    assert {step["kernel"] for step in convolution_steps} == {"conv2d", "conv2d_dx", "conv2d_dw"}
    for step in convolution_steps:
        specs = [plan["buffers"][name] for name in step["inputs"]]
        kernel = emit_kernel(step["kernel"], specs, plan["buffers"][step["output"]], step["attrs"])
        assert kernel.ptx and kernel.grid[0] > 0


def test_conv2d_functional_route_and_shape_contract():
    class Functional(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(3, 2, 1, 1))

        def forward(self, x):
            return torch.nn.functional.conv2d(x, self.weight, stride=(2, 1), padding=(1, 0))

    model = Functional()
    plan = capture_model(model, (torch.ones(1, 2, 4, 5),))
    step, = [step for step in plan["steps"] if step["kernel"] == "conv2d"]
    assert step["attrs"] == {"stride": [2, 1], "padding": [1, 0], "dilation": [1, 1]}
    assert plan["buffers"][step["output"]]["shape"] == [1, 3, 3, 5]


@pytest.mark.parametrize("kwargs", [
    {"groups": 2}, {"padding_mode": "reflect"}, {"padding": "same"},
])
def test_unsupported_conv2d_modes_fail_during_capture(kwargs):
    class Unsupported(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(2, 4, 3, **kwargs)

        def forward(self, x):
            return self.conv(x)

    with pytest.raises(ModelGraphError):
        capture_model(Unsupported(), (torch.ones(1, 2, 5, 5),))


@pytest.mark.parametrize("attrs", [
    {"stride": 1, "padding": [0, 0], "dilation": [1, 1]},
    {"stride": [True, 1], "padding": [0, 0], "dilation": [1, 1]},
    {"stride": [0, 1], "padding": [0, 0], "dilation": [1, 1]},
    {"stride": [1, 1], "padding": [-1, 0], "dilation": [1, 1]},
    {"stride": [1, 1], "padding": [0, 0], "dilation": [0, 1]},
    {"stride": [1, 1], "padding": [0, 0], "dilation": [1, 1], "groups": 1},
])
def test_conv2d_malformed_kernel_contract_rejects(attrs):
    with pytest.raises(ValueError):
        emit_kernel("conv2d", [{"shape": [1, 2, 5, 5]}, {"shape": [3, 2, 3, 3]}],
                    {"shape": [1, 3, 3, 3]}, attrs)


@pytest.mark.parametrize("op,shapes,output", [
    ("conv2d", [[1, 2, 5, 5], [3, 1, 3, 3]], [1, 3, 3, 3]),
    ("conv2d", [[1, 2, 5, 5], [3, 2, 3, 3], [2]], [1, 3, 3, 3]),
    ("conv2d", [[1, 2, 5, 5], [3, 2, 3, 3]], [1, 3, 4, 3]),
    ("conv2d_dx", [[1, 3, 3, 3], [3, 2, 3, 3]], [1, 2, 6, 5]),
    ("conv2d_dw", [[1, 3, 3, 3], [2, 2, 5, 5]], [3, 2, 3, 3]),
])
def test_conv2d_shape_metadata_must_match_formula(op, shapes, output):
    with pytest.raises(ValueError):
        emit_kernel(op, [{"shape": shape} for shape in shapes], {"shape": output},
                    {"stride": [1, 1], "padding": [0, 0], "dilation": [1, 1]})
