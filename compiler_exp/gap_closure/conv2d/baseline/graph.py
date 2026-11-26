"""A bounded CPU Torch FX capture bridge to OA-owned model kernels.

Tracing and shape propagation happen on CPU. The resulting JSON is an explicit
forward/backward program, not an Inductor backend or a generic Python compiler.
Unsupported operations fail with their FX node instead of falling back.
"""
from __future__ import annotations

import copy
import math
import operator
from pathlib import Path


class ModelGraphError(ValueError):
    """The model is outside the admitted native graph contract."""


def _node_error(node, message):
    import re
    frames = re.findall(r'File "([^"]+)", line (\d+), in ([^\n]+)', node.meta.get('stack_trace', ''))
    own = Path(__file__).resolve()
    for filename, line, function in reversed(frames):
        normalized = filename.replace('\\', '/')
        if '/torch/' not in normalized and Path(filename).resolve() != own:
            return f'{filename}:{line} in {function}: {message}'
    return message


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ModelGraphError("native model capture needs the optional PyTorch dependency") from exc
    return torch


def _attribute(module, target):
    for name in target.split("."):
        module = getattr(module, name)
    return module


def extract_weights(model, plan):
    """Bind the CPU model state to the graph's canonical, alias-aware names."""
    torch = _torch()
    weights = {}
    for spec in plan["parameters"]:
        try:
            value = _attribute(model, spec["target"])
        except AttributeError as exc:
            raise ModelGraphError(f"missing captured model state {spec['target']!r}") from exc
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.dtype != torch.float32 or list(value.shape) != spec["shape"]):
            raise ModelGraphError(f"model state {spec['target']!r} changed shape, dtype or CPU ownership")
        weights[spec["name"]] = value.detach().contiguous().clone()
    return weights


def flatten_outputs(plan, actual):
    """Validate a real model result against the captured container structure.

    Unlike a zip over results, this rejects missing/extra outputs and preserves
    canonical names when the same tensor is returned in multiple positions.
    """
    torch = _torch()
    result = {}

    def visit(structure, value, location):
        if "tensor" in structure:
            name = structure["tensor"]
            spec = plan["buffers"][name]
            if (not isinstance(value, torch.Tensor) or list(value.shape) != spec["shape"]
                    or value.dtype != torch.float32):
                raise ModelGraphError(f"{location}: model output differs from captured FP32 tensor shape {spec['shape']}")
            if name in result and result[name] is not value:
                raise ModelGraphError(f"{location}: captured repeated tensor output no longer aliases the same value")
            result[name] = value
        elif "tuple" in structure or "list" in structure:
            key = "tuple" if "tuple" in structure else "list"
            kind = tuple if key == "tuple" else list
            if not isinstance(value, kind) or len(value) != len(structure[key]):
                raise ModelGraphError(f"{location}: model output container differs from capture")
            for index, (child, item) in enumerate(zip(structure[key], value)):
                visit(child, item, f"{location}/{index}")
        elif "dict" in structure:
            if not isinstance(value, dict) or set(value) != set(structure["dict"]):
                raise ModelGraphError(f"{location}: model output dictionary differs from capture")
            for key, child in structure["dict"].items():
                visit(child, value[key], f"{location}/{key}")
        else:
            raise ModelGraphError("invalid native model output structure")

    structure = plan.get("output_structure")
    if structure is None:
        raise ModelGraphError("native model plan does not record its output structure")
    visit(structure, actual, "output")
    if set(result) != set(plan["outputs"]):
        raise ModelGraphError("captured output inventory differs from its container structure")
    return result


def capture_model(model, example_inputs, *, math_mode="fp32"):
    """Capture a CPU ``nn.Module`` and emit a JSON-serializable native program.

    Supported operations include Linear, ReLU, sigmoid, static sum/mean, tanh GELU, last-axis softmax and
    affine LayerNorm, rank-two through rank-four batch-broadcast matmul, reshape,
    rank-two through rank-four permutation and concatenation, broadcast arithmetic,
    dense int64 embeddings and Boolean/causal SDPA. Linear accepts
    any positive leading dimensions; flattening is explicit in the plan.
    Storage, scalar arithmetic and gradients are FP32. The explicit optional
    f16_rne_f32_accum mode rounds each matrix product's operands independently
    to half; batched products retain their independently bounded operand contract.
    Branches and parameter aliases receive explicit gradient accumulation.
    """
    torch = _torch()
    if math_mode not in ("fp32", "f16_rne_f32_accum"):
        raise ModelGraphError("native model capture requires explicit fp32 or f16_rne_f32_accum math")
    if not isinstance(model, torch.nn.Module):
        raise ModelGraphError("model factory must return a torch.nn.Module")
    if isinstance(example_inputs, torch.Tensor):
        example_inputs = (example_inputs,)
    elif isinstance(example_inputs, (tuple, list)):
        example_inputs = tuple(example_inputs)
    else:
        raise ModelGraphError("example_inputs must be a tensor or a sequence of tensors")
    if not example_inputs:
        raise ModelGraphError("capture requires at least one tensor input")
    model_state = [*model.named_parameters(), *model.named_buffers()]
    for label, tensor in [*(('input', x) for x in example_inputs), *model_state]:
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                or tensor.dtype not in ((torch.float32, torch.int64, torch.bool) if label == "input" else (torch.float32,)) or tensor.layout != torch.strided
                or not tensor.is_contiguous() or any(dim <= 0 for dim in tensor.shape)):
            dtypes = "FP32, int64 index or Boolean mask" if label == "input" else "FP32"
            raise ModelGraphError(f"{label}: native model capture requires contiguous positive-shape CPU {dtypes} tensors")
        if tensor.dtype == torch.float32 and not bool(torch.isfinite(tensor).all()):
            raise ModelGraphError(f"{label}: FP32 capture examples and initial state must contain finite FP32 values")
    spans = []
    for name, tensor in model_state:
        begin, end = tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size()
        if any(id(tensor) != identity and begin < old_end and old_begin < end
               for old_begin, old_end, identity in spans):
            raise ModelGraphError(f"{name}: distinct model tensors with overlapping storage are unsupported")
        spans.append((begin, end, id(tensor)))
    captured = copy.deepcopy(model)
    before_state = {name: value.detach().clone() for name, value in
                    [*captured.named_parameters(), *captured.named_buffers()]}
    try:
        # Never execute shape propagation against the caller's mutable state.
        tracer = torch.fx.Tracer()
        tracer.record_stack_traces = True
        traced = torch.fx.GraphModule(captured, tracer.trace(captured))
    except Exception as exc:
        import traceback
        frames = [frame for frame in traceback.extract_tb(exc.__traceback__)
                  if '/torch/' not in frame.filename.replace('\\', '/')
                  and Path(frame.filename).resolve() != Path(__file__).resolve()]
        where = f' at {frames[-1].filename}:{frames[-1].lineno} in {frames[-1].name}' if frames else ''
        raise ModelGraphError(f"FX capture failed{where}: {exc}") from exc
    functional = torch.nn.functional
    function_ops = {
        operator.add: "add", torch.add: "add", operator.mul: "mul", torch.mul: "mul",
        torch.relu: "relu", functional.relu: "relu", functional.linear: "linear",
        torch.sigmoid: "sigmoid", functional.sigmoid: "sigmoid",
        torch.sum: "sum", torch.mean: "mean",
        torch.flatten: "reshape", torch.reshape: "reshape", torch.transpose: "transpose",
        operator.matmul: "matmul", torch.matmul: "matmul", torch.mm: "matmul", torch.bmm: "bmm",
        torch.softmax: "softmax", functional.softmax: "softmax", functional.gelu: "gelu",
        functional.layer_norm: "layernorm",
        functional.embedding: "embedding", functional.scaled_dot_product_attention: "sdpa",
        torch.cat: "concat", torch.concat: "concat",
    }
    module_ops = (torch.nn.Linear, torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Identity, torch.nn.Flatten, torch.nn.Softmax, torch.nn.GELU, torch.nn.LayerNorm, torch.nn.Embedding)
    methods = {"relu", "sigmoid", "sum", "mean", "add", "mul", "view", "reshape", "flatten", "transpose", "t", "permute", "contiguous", "clone", "size", "dim", "matmul", "bmm", "softmax"}
    shape_ops = {operator.getitem, getattr}
    for node in traced.graph.nodes:
        supported = node.op in ("placeholder", "get_attr", "output")
        if node.op == "call_module":
            supported = isinstance(traced.get_submodule(node.target), module_ops)
        elif node.op == "call_function":
            supported = node.target in function_ops or node.target in shape_ops
        elif node.op == "call_method":
            supported = node.target in methods
        if not supported:
            raise ModelGraphError(_node_error(node, f"unsupported native model node {node.format_node()}"))
    try:
        from torch.fx.passes.shape_prop import ShapeProp
        with torch.no_grad():
            ShapeProp(traced).propagate(*(x.detach().clone() for x in example_inputs))
    except Exception as exc:
        # ShapeProp retains the failing node in its error. Its stack_trace now
        # also identifies the user's call site, including call_module nodes.
        raise ModelGraphError(f"CPU model shape propagation failed: {exc}") from exc
    for name, value in [*captured.named_parameters(), *captured.named_buffers()]:
        if name not in before_state or not torch.equal(value.detach(), before_state[name]):
            raise ModelGraphError(f"{name}: model state mutation during capture is unsupported")

    plan = {"schema": 1, "arch": "sm_80", "math": math_mode, "inputs": [], "parameters": [],
            "buffers": {}, "steps": [], "outputs": [],
            "contract": {"backend": "openassembler-native", "device_compute": "OA-emitted SASS only",
                         "shape_policy": "exact", "input_layout": "contiguous", "input_dtype": "f32",
                         "accumulation_dtype": "f32", "output_dtype": "f32", "fallback": "none",
                         "numerical_domain": "finite FP32 inputs and finite FP32 intermediates",
                         "exceptional_values": "NaN/Inf and overflow outside the supported contract"}}
    if math_mode == "f16_rne_f32_accum":
        plan["contract"].update({"matrix_operand_dtype": "f16", "matrix_operand_rounding": "round_to_nearest_even",
                                 "master_parameter_dtype": "f32",
                                 "matrix_backward": "straight-through input cast; dX and dW independently round both operands to FP16",
                                 "numerical_domain": "finite FP32 arithmetic; matrix operands must remain finite after FP16 rounding"})
    if any(x.dtype != torch.float32 for x in example_inputs):
        plan["contract"]["input_dtype"] = "declared per input: f32, i64 indices, bool masks"
    values, constants, aliases = {}, {}, {}
    trainable = set()
    reserved_names = {node.name for node in traced.graph.nodes}
    serial = 0

    def fail(node, reason):
        raise ModelGraphError(_node_error(node, f"{reason}; native model node {node.format_node()}"))

    def shape(node):
        meta = node.meta.get("tensor_meta")
        if meta is None or not hasattr(meta, "shape") or meta.dtype not in (torch.float32, torch.int64, torch.bool):
            fail(node, "expected one tensor value with declared FP32, int64 or Boolean dtype")
        result = list(meta.shape)
        if any(type(x) is not int or x <= 0 for x in result):
            fail(node, "dynamic, empty or negative shapes are unsupported")
        return result

    def buffer(name, dims, dtype="f32"):
        plan["buffers"][name] = {"shape": list(dims), "dtype": dtype}
        return name

    def fresh(stem, dims):
        nonlocal serial
        serial += 1
        while (f"_oa_{stem}_{serial}" in plan["buffers"]
               or f"_oa_{stem}_{serial}" in reserved_names):
            serial += 1
        return buffer(f"_oa_{stem}_{serial}", dims)

    def emit(kernel, inputs, output, attrs=None):
        plan["steps"].append({"kernel": kernel, "inputs": list(inputs), "output": output, "attrs": attrs or {}})
        return output

    def view(inputs, output):
        emit("copy", inputs, output)
        plan["steps"][-1]["view"] = True
        return output

    def parameter(target, node):
        try:
            tensor = _attribute(model, target)
        except AttributeError:
            fail(node, f"captured constant {target!r} is not owned by the input model")
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                or tensor.dtype != torch.float32 or not tensor.is_contiguous()):
            fail(node, f"state {target!r} must be contiguous CPU FP32")
        if id(tensor) in aliases:
            return aliases[id(tensor)]
        name = buffer("parameter:" + target, tensor.shape)
        aliases[id(tensor)] = name
        requires_grad = bool(isinstance(tensor, torch.nn.Parameter) and tensor.requires_grad)
        plan["parameters"].append({"name": name, "target": target, "shape": list(tensor.shape),
                                   "dtype": "f32", "requires_grad": requires_grad})
        if requires_grad:
            trainable.add(name)
        return name

    def resolve(value):
        if isinstance(value, torch.fx.Node):
            return values.get(value, constants.get(value))
        if isinstance(value, (tuple, list)):
            return type(value)(resolve(x) for x in value)
        return value

    def tensor(value, node):
        value = resolve(value)
        if not isinstance(value, str) or value not in plan["buffers"]:
            fail(node, "expected a captured tensor operand")
        return value

    def scalar(value, node):
        value = resolve(value)
        if type(value) not in (float, int) or not math.isfinite(value):
            fail(node, "scalar operands must be finite real constants")
        return float(value)

    def linear(node, x, weight, bias):
        xs, ws = plan["buffers"][x]["shape"], plan["buffers"][weight]["shape"]
        if not xs or len(ws) != 2 or xs[-1] != ws[1]:
            fail(node, "Linear requires [...,K] inputs and [N,K] weight")
        if bias is not None and plan["buffers"][bias]["shape"] != [ws[0]]:
            fail(node, "Linear bias must have shape [N]")
        m = math.prod(xs[:-1])
        if len(xs) != 2:
            x = view([x], fresh("linear_input", [m, ws[1]]))
        out_shape = shape(node)
        name = buffer(node.name, out_shape)
        out = name if len(out_shape) == 2 else fresh("linear_output", [m, ws[0]])
        emit("linear", [x, weight] + ([] if bias is None else [bias]), out,
             {"m": m, "n": ws[0], "k": ws[1]})
        if out != name:
            view([out], name)
        return name

    def structure(value):
        if isinstance(value, torch.fx.Node):
            name = tensor(value, value)
            if name not in plan["outputs"]:
                plan["outputs"].append(name)
            return {"tensor": name}
        if isinstance(value, (tuple, list)):
            return {"tuple" if isinstance(value, tuple) else "list": [structure(x) for x in value]}
        if isinstance(value, dict) and all(isinstance(k, str) for k in value):
            return {"dict": {k: structure(v) for k, v in value.items()}}
        raise ModelGraphError("model outputs must contain tensors, tuples, lists or string-keyed dictionaries")

    input_index = 0
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            if input_index >= len(example_inputs):
                fail(node, "all model arguments must be supplied tensor inputs")
            name = buffer(node.name, shape(node), {torch.float32:"f32", torch.int64:"i64", torch.bool:"bool"}[node.meta["tensor_meta"].dtype])
            plan["inputs"].append({"name": name, **plan["buffers"][name]})
            values[node] = name
            input_index += 1
            continue
        if node.op == "get_attr":
            values[node] = parameter(node.target, node)
            continue
        if node.op == "output":
            plan["output_structure"] = structure(node.args[0])
            continue
        # Shape-only nodes become CPU specialization constants, never kernels.
        if node.op == "call_method" and node.target in ("size", "dim"):
            src = tensor(node.args[0], node)
            dims = plan["buffers"][src]["shape"]
            constants[node] = (len(dims) if node.target == "dim" else
                               (tuple(dims) if len(node.args) == 1 else dims[resolve(node.args[1])]))
            continue
        if node.op == "call_function" and node.target is getattr:
            if len(node.args) != 2 or node.args[1] not in ("shape", "ndim"):
                fail(node, "only tensor shape/ndim attributes are supported")
            dims = plan["buffers"][tensor(node.args[0], node)]["shape"]
            constants[node] = tuple(dims) if node.args[1] == "shape" else len(dims)
            continue
        if node.op == "call_function" and node.target is operator.getitem:
            source = resolve(node.args[0])
            if isinstance(source, str):
                fail(node, "tensor slicing/indexing is unsupported")
            try:
                constants[node] = source[resolve(node.args[1])]
            except (TypeError, IndexError) as exc:
                fail(node, f"invalid specialized shape index: {exc}")
            continue

        module = traced.get_submodule(node.target) if node.op == "call_module" else None
        op = ("embedding" if isinstance(module, torch.nn.Embedding) else
              "linear" if isinstance(module, torch.nn.Linear) else
              "relu" if isinstance(module, torch.nn.ReLU) else
              "sigmoid" if isinstance(module, torch.nn.Sigmoid) else
              "softmax" if isinstance(module, torch.nn.Softmax) else
              "gelu" if isinstance(module, torch.nn.GELU) else
              "layernorm" if isinstance(module, torch.nn.LayerNorm) else
              "reshape" if isinstance(module, (torch.nn.Identity, torch.nn.Flatten)) else
              function_ops.get(node.target) if node.op == "call_function" else node.target)
        if op == "linear":
            x = tensor(node.args[0] if node.args else node.kwargs.get("input"), node)
            if module is not None:
                w = parameter(node.target + ".weight", node)
                b = parameter(node.target + ".bias", node) if module.bias is not None else None
            else:
                w = tensor(node.args[1] if len(node.args) > 1 else node.kwargs.get("weight"), node)
                raw_bias = node.args[2] if len(node.args) > 2 else node.kwargs.get("bias")
                b = tensor(raw_bias, node) if raw_bias is not None else None
            values[node] = linear(node, x, w, b)
            continue
        dims = shape(node)
        name = buffer(node.name, dims)
        if op in ("sum", "mean"):
            x = tensor(node.args[0] if node.args else node.kwargs.get("input"), node)
            xs = plan["buffers"][x]["shape"]
            if (plan["buffers"][x]["dtype"] != "f32" or len(xs) > 4
                    or node.kwargs.get("dtype") not in (None, torch.float32)):
                fail(node, "sum/mean requires FP32 input/output of rank zero through four without dtype conversion")
            if len(node.args) > 3 or set(node.kwargs) - {"input", "dim", "keepdim", "dtype", "out"}:
                fail(node, "unsupported sum/mean argument")
            if node.kwargs.get("out") is not None:
                fail(node, "sum/mean out= mutation is unsupported")
            dim = resolve(node.args[1] if len(node.args) > 1 else node.kwargs.get("dim"))
            keepdim = resolve(node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False))
            if type(keepdim) is not bool:
                fail(node, "sum/mean keepdim must be a static Boolean")
            rank = len(xs)
            raw_axes = (tuple(range(rank)) if dim is None else (dim,) if type(dim) is int else dim)
            if not isinstance(raw_axes, (tuple, list)) or any(type(i) is not int for i in raw_axes):
                fail(node, "sum/mean dim must be static integer axes")
            if not raw_axes:
                raw_axes = tuple(range(rank))
            if rank == 0:
                if tuple(raw_axes) not in ((), (0,), (-1,)):
                    fail(node, "invalid scalar reduction axis")
                axes = ()
            else:
                if any(not -rank <= i < rank for i in raw_axes) or len({i % rank for i in raw_axes}) != len(raw_axes):
                    fail(node, "sum/mean axes must be distinct and in range")
                axes = tuple(sorted(i % rank for i in raw_axes))
            kept = [1 if i in axes else n for i, n in enumerate(xs)]
            expected = kept if keepdim else [n for i, n in enumerate(xs) if i not in axes]
            if dims != expected:
                fail(node, "sum/mean output disagrees with normalized axes")
            reduced = emit("sum_to", [x], fresh("reduced", kept))
            if op == "mean":
                reduced = emit("mul", [reduced], fresh("mean", kept),
                               {"scalar": 1 / math.prod(xs[i] for i in axes)})
            view([reduced], name)
        elif op == "sigmoid":
            x = tensor(node.args[0] if node.args else node.kwargs.get("input"), node)
            if plan["buffers"][x]["dtype"] != "f32" or node.kwargs.get("out") is not None:
                fail(node, "sigmoid requires FP32 input without out= mutation or dtype conversion")
            emit("sigmoid", [x], name)
        elif op == "embedding":
            x = tensor(node.args[0] if node.args else node.kwargs.get("input"), node)
            if module is not None:
                w = parameter(node.target + ".weight", node)
                padding, max_norm, sparse, frequency = module.padding_idx, module.max_norm, module.sparse, module.scale_grad_by_freq
            else:
                get = lambda i, k, default=None: node.args[i] if len(node.args)>i else node.kwargs.get(k, default)
                w = tensor(get(1, "weight"), node)
                padding, max_norm = get(2,"padding_idx"), get(3,"max_norm")
                frequency, sparse = get(5,"scale_grad_by_freq",False), get(6,"sparse",False)
            if max_norm is not None or sparse or frequency or padding is not None:
                fail(node, "embedding requires dense gradients, no padding_idx, max_norm or frequency scaling")
            emit("embedding", [x,w], name)
            domain = {"name": x, "lower": 0, "upper": plan["buffers"][w]["shape"][0]}
            if domain not in plan.setdefault("index_domains", []): plan["index_domains"].append(domain)
        elif op == "sdpa":
            get = lambda i, k, default=None: node.args[i] if len(node.args)>i else node.kwargs.get(k,default)
            q,k,v = [tensor(get(i,key),node) for i,key in enumerate(("query","key","value"))]
            mask,dropout,causal = get(3,"attn_mask"),get(4,"dropout_p",0.0),get(5,"is_causal",False)
            if dropout != 0 or type(causal) is not bool or node.kwargs.get("enable_gqa",False):
                fail(node,"SDPA requires dropout_p=0 and no GQA")
            if mask is not None and causal: fail(node,"SDPA mask and is_causal cannot both be set")
            qs,ks,vs = [plan["buffers"][x]["shape"] for x in (q,k,v)]
            if (not 3<=len(qs)<=4 or len(ks)!=len(qs) or len(vs)!=len(qs)
                    or qs[:-2]!=ks[:-2] or ks[:-2]!=vs[:-2] or qs[-1]!=ks[-1] or ks[-2]!=vs[-2]):
                fail(node,"SDPA requires rank3/4 matching batch axes and compatible head dimensions")
            batch = math.prod(qs[:-2]); length,width,head=qs[-2],ks[-2],qs[-1]
            scale=scalar(head**-0.5 if node.kwargs.get("scale") is None else node.kwargs["scale"],node)
            qv=view([q],fresh("query",[batch,length,head]))
            kv=view([k],fresh("key",[batch,width,head]))
            vv=view([v],fresh("value",[batch,width,vs[-1]]))
            kt=emit("permute",[kv],fresh("kt",[batch,head,width]),{"axes":[0,2,1]})
            scores=emit("bmm",[qv,kt],fresh("scores",[batch,length,width]))
            scaled=emit("mul",[scores],fresh("scaled",[batch,length,width]),{"scalar":scale})
            # Keep original batch rank for mask broadcasting; views never reinterpret dtype.
            sdims=[*qs[:-2],length,width]
            scored=view([scaled],fresh("attention_scores",sdims))
            args=[scored] if mask is None else [scored,tensor(mask,node)]
            probs=emit("masked_softmax",args,fresh("probabilities",sdims),{"causal":causal})
            pv=view([probs],fresh("probabilities3",[batch,length,width]))
            ov=emit("bmm",[pv,vv],fresh("attention_output",[batch,length,vs[-1]]))
            view([ov],name)
        elif op == "concat":
            raw_inputs = node.args[0] if node.args else node.kwargs.get("tensors")
            if not isinstance(raw_inputs, (tuple, list)) or len(raw_inputs) < 2:
                fail(node, "native concatenation requires at least two tensor operands")
            inputs = [tensor(x, node) for x in raw_inputs]
            axis = resolve(node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0))
            if type(axis) is not int or not -len(dims) <= axis < len(dims) or not 2 <= len(dims) <= 4:
                fail(node, "native concatenation requires rank two through four and a constant valid axis")
            if node.kwargs.get("out") is not None:
                fail(node, "concatenation out= mutation is unsupported")
            emit("concat", inputs, name, {"axis": axis % len(dims)})
        elif op == "layernorm":
            x = tensor(node.args[0] if node.args else node.kwargs.get("input"), node)
            if module is not None:
                normalized_shape, epsilon = tuple(module.normalized_shape), module.eps
                weight = parameter(node.target + ".weight", node) if module.weight is not None else None
                bias = parameter(node.target + ".bias", node) if module.bias is not None else None
            else:
                get_arg = lambda index, key, default=None: node.args[index] if len(node.args) > index else node.kwargs.get(key, default)
                normalized_shape = resolve(get_arg(1, "normalized_shape"))
                normalized_shape = (normalized_shape,) if type(normalized_shape) is int else tuple(normalized_shape)
                epsilon = scalar(get_arg(4, "eps", 1e-5), node)
                raw_weight, raw_bias = get_arg(2, "weight"), get_arg(3, "bias")
                weight = tensor(raw_weight, node) if raw_weight is not None else None
                bias = tensor(raw_bias, node) if raw_bias is not None else None
            if normalized_shape != (dims[-1],) or dims[-1] > 1024 or not 0 < epsilon < float("inf"):
                fail(node, "native LayerNorm requires one last-dimension width <=1024 and positive finite epsilon")
            if bias is not None and weight is None:
                fail(node, "native LayerNorm bias requires an affine weight")
            for value in (weight, bias):
                if value is not None and plan["buffers"][value]["shape"] != [dims[-1]]:
                    fail(node, "LayerNorm affine parameters must match the last dimension")
            normalized = fresh("normalized", dims) if weight is not None else name
            emit("layernorm", [x], normalized, {"epsilon": float(epsilon)})
            if weight is not None:
                emit("channel_affine", [normalized, weight] + ([] if bias is None else [bias]), name)
        elif op in ("matmul", "bmm"):
            if node.kwargs.get("out") is not None:
                fail(node, "matmul out= mutation is unsupported")
            lhs = node.args[0] if node.args else node.kwargs.get("input")
            rhs = node.args[1] if len(node.args) > 1 else node.kwargs.get("other", node.kwargs.get("mat2"))
            a, b = tensor(lhs, node), tensor(rhs, node)
            ashape, bshape = plan["buffers"][a]["shape"], plan["buffers"][b]["shape"]
            if len(ashape) < 2 or len(bshape) < 2 or ashape[-1] != bshape[-2]:
                fail(node, "native matmul requires matrix operands and matching matrix inner dimensions")
            if ashape[:-2] != bshape[:-2] and max(len(ashape), len(bshape)) > 4:
                fail(node, "batch-broadcast matmul requires rank two through four")
            if op == "bmm" and (len(ashape) != 3 or len(bshape) != 3 or ashape[0] != bshape[0]):
                fail(node, "native bmm requires rank-three operands with identical batch shapes without broadcasting")
            if len(ashape) == len(bshape) == 2:
                bt = emit("transpose2d", [b], fresh("matmul_weight", bshape[::-1]))
                emit("linear", [a, bt], name, {"m": ashape[0], "n": bshape[1], "k": ashape[1]})
            else:
                from .kernels_semantics import broadcast_shape
                if ashape[:-2] == bshape[:-2]:
                    # Preserve the existing equal-batch flattening route even
                    # when its input rank exceeds the new broadcast-copy bound.
                    batches = ashape[:-2]
                else:
                    try:
                        batches = list(broadcast_shape(ashape[:-2], bshape[:-2]))
                    except ValueError as exc:
                        fail(node, str(exc))
                batch = math.prod(batches)
                av, bv, target = a, b, name
                # Materialize only operands whose batch axes expand. The copy
                # preserves FP32 bits; no arithmetic or dtype conversion occurs.
                if ashape != [*batches, *ashape[-2:]]:
                    av = emit("broadcast_to", [a], fresh("bmm_left_expanded", [*batches, *ashape[-2:]]))
                if bshape != [*batches, *bshape[-2:]]:
                    bv = emit("broadcast_to", [b], fresh("bmm_right_expanded", [*batches, *bshape[-2:]]))
                if len(batches) != 1:
                    av = view([av], fresh("bmm_left", [batch, *ashape[-2:]]))
                    bv = view([bv], fresh("bmm_right", [batch, *bshape[-2:]]))
                    target = fresh("bmm_output", [batch, ashape[-2], bshape[-1]])
                emit("bmm", [av, bv], target)
                if target != name:
                    view([target], name)
        elif op in ("softmax", "gelu"):
            x = tensor(node.args[0] if node.args else node.kwargs.get("input"), node)
            if op == "softmax":
                dim = (module.dim if module is not None else
                       resolve(node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")))
                dtype = node.args[2] if len(node.args) > 2 else node.kwargs.get("dtype")
                if type(dim) is not int or dim not in (-1, len(dims)-1) or dtype is not None:
                    fail(node, "native softmax requires the last dimension and unchanged FP32 dtype")
                if not dims or dims[-1] > 1024:
                    fail(node, "native softmax width must be in [1,1024]")
            else:
                approximate = (module.approximate if module is not None else
                               node.args[1] if len(node.args) > 1 else node.kwargs.get("approximate", "none"))
                if approximate != "tanh":
                    fail(node, "native GELU requires explicit approximate='tanh'")
            emit(op, [x], name)
        elif op in ("add", "mul"):
            if node.kwargs.get("out") is not None:
                fail(node, "out= mutation is unsupported")
            if op == "add" and node.kwargs.get("alpha", 1) != 1:
                fail(node, "add alpha other than one is unsupported")
            lhs = node.args[0] if node.args else node.kwargs.get("input")
            rhs = node.args[1] if len(node.args) > 1 else node.kwargs.get("other")
            a, b = resolve(lhs), resolve(rhs)
            tensors = [x for x in (a, b) if isinstance(x, str) and x in plan["buffers"]]
            if len(tensors) == 2:
                broadcast = any(plan["buffers"][x]["shape"] != dims for x in tensors)
                emit("broadcast_"+op if broadcast else op, tensors, name)
            elif len(tensors) == 1:
                if plan["buffers"][tensors[0]]["shape"] != dims:
                    fail(node, "scalar arithmetic must preserve the tensor shape")
                emit(op, tensors, name, {"scalar": scalar(b if a == tensors[0] else a, node)})
            else:
                fail(node, "arithmetic requires at least one tensor")
        elif op == "relu":
            positional_inplace = (node.op == "call_function" and node.target is functional.relu
                                  and len(node.args) > 1 and node.args[1])
            if (node.kwargs.get("inplace", False) or positional_inplace
                    or isinstance(module, torch.nn.ReLU) and module.inplace):
                fail(node, "in-place ReLU is unsupported")
            emit("relu", [tensor(node.args[0] if node.args else node.kwargs.get("input"), node)], name)
        elif op in ("reshape", "view", "flatten", "contiguous", "clone"):
            x = tensor(node.args[0] if node.args else node.kwargs.get("input"), node)
            if math.prod(plan["buffers"][x]["shape"]) != math.prod(dims):
                fail(node, "reshape must preserve element count")
            if node.kwargs.get("memory_format") not in (None, torch.contiguous_format):
                fail(node, "only contiguous memory format is supported")
            emit("copy", [x], name)
            if op in ("reshape", "view", "flatten") and not isinstance(module, torch.nn.Identity):
                plan["steps"][-1]["view"] = True
        elif op in ("transpose", "t", "permute"):
            x = tensor(node.args[0] if node.args else node.kwargs.get("input"), node)
            before = plan["buffers"][x]["shape"]
            rank = len(before)
            if not 2 <= rank <= 4:
                fail(node, "native transpose requires tensor rank two through four")
            if op == "t":
                if rank != 2:
                    fail(node, "t() requires rank two")
                axes = (1, 0)
            elif op == "permute":
                raw = resolve(node.args[1:])
                axes = tuple(raw[0]) if len(raw) == 1 and isinstance(raw[0], (tuple, list)) else tuple(raw)
                if any(type(x) is not int or not -rank <= x < rank for x in axes):
                    fail(node, "invalid permutation axis")
                axes = tuple(x % rank for x in axes)
                if sorted(axes) != list(range(rank)):
                    fail(node, "invalid tensor permutation")
            else:
                a = resolve(node.args[1] if len(node.args) > 1 else node.kwargs.get("dim0"))
                b = resolve(node.args[2] if len(node.args) > 2 else node.kwargs.get("dim1"))
                if type(a) is not int or type(b) is not int or not -rank <= a < rank or not -rank <= b < rank:
                    fail(node, "transpose dimensions must identify existing axes")
                axes = list(range(rank))
                axes[a % rank], axes[b % rank] = axes[b % rank], axes[a % rank]
                axes = tuple(axes)
            if axes == tuple(range(rank)):
                emit("copy", [x], name)
            elif rank == 2:
                emit("transpose2d", [x], name)
            else:
                emit("permute", [x], name, {"axes": list(axes)})
        else:
            fail(node, "unsupported native model operation")
        values[node] = name
    if input_index != len(example_inputs) or not plan["outputs"]:
        raise ModelGraphError("capture input count differs from examples or produced no tensor output")
    _differentiate(plan, trainable)
    return plan


def _differentiate(plan, trainable):
    """Reverse the lowered native operations, with explicit sum nodes at joins."""
    grads, steps, seeds = {}, [], {}
    serial = 0

    def fresh(stem, shape):
        nonlocal serial
        serial += 1
        name = f"_oa_grad_{stem}_{serial}"
        while name in plan["buffers"]:
            serial += 1
            name = f"_oa_grad_{stem}_{serial}"
        plan["buffers"][name] = {"shape": list(shape), "dtype": "f32"}
        return name

    def emit(op, args, shape, attrs=None):
        name = fresh(op, shape)
        steps.append({"kernel": op, "inputs": list(args), "output": name, "attrs": attrs or {}})
        return name

    def accumulate(name, value):
        if name in grads:
            grads[name] = emit("add", [grads[name], value], plan["buffers"][name]["shape"])
        else:
            grads[name] = value

    for output in plan["outputs"]:
        seed = fresh("seed", plan["buffers"][output]["shape"])
        seeds[output] = seed
        accumulate(output, seed)
    for step in reversed(plan["steps"]):
        op, args, out, attrs = step["kernel"], step["inputs"], step["output"], step["attrs"]
        if out not in grads:
            continue
        dy = grads[out]
        shapes = [plan["buffers"][x]["shape"] for x in args]
        if op == "linear":
            x, w = args[:2]
            accumulate(x, emit("linear_dx", [dy, w], shapes[0], attrs))
            accumulate(w, emit("linear_dw", [dy, x], shapes[1], attrs))
            if len(args) == 3:
                accumulate(args[2], emit("bias_grad", [dy], shapes[2], attrs))
        elif op == "bmm":
            accumulate(args[0], emit("bmm_dx", [dy, args[1]], shapes[0]))
            accumulate(args[1], emit("bmm_dw", [dy, args[0]], shapes[1]))
        elif op == "relu":
            accumulate(args[0], emit("relu_backward", [dy, args[0]], shapes[0]))
        elif op == "copy":
            derivative = emit("copy", [dy], shapes[0])
            if step.get("view"):
                steps[-1]["view"] = True
            accumulate(args[0], derivative)
        elif op == "transpose2d":
            accumulate(args[0], emit("transpose2d", [dy], shapes[0]))
        elif op == "permute":
            axes = attrs["axes"]
            accumulate(args[0], emit("permute", [dy], shapes[0], {"axes": [axes.index(i) for i in range(len(axes))]}))
        elif op == "embedding":
            accumulate(args[1], emit("embedding_grad", [args[0],dy], shapes[1]))
        elif op == "broadcast_to":
            accumulate(args[0], emit("sum_to", [dy], shapes[0]))
        elif op == "sum_to":
            accumulate(args[0], emit("broadcast_to", [dy], shapes[0]))
        elif op in ("broadcast_add", "broadcast_mul"):
            full = plan["buffers"][out]["shape"]
            for index,arg in enumerate(args):
                g = dy if op == "broadcast_add" else emit("broadcast_mul", [dy,args[1-index]], full)
                accumulate(arg, emit("sum_to", [g], shapes[index]) if shapes[index] != full else g)
        elif op in ("softmax", "masked_softmax"):
            accumulate(args[0], emit("softmax_backward", [dy, out], shapes[0]))
        elif op == "gelu":
            accumulate(args[0], emit("gelu_backward", [dy, args[0]], shapes[0]))
        elif op == "sigmoid":
            accumulate(args[0], emit("sigmoid_backward", [dy, out], shapes[0]))
        elif op == "layernorm":
            accumulate(args[0], emit("layernorm_backward", [dy, args[0]], shapes[0], attrs))
        elif op == "channel_affine":
            accumulate(args[0], emit("channel_mul", [dy, args[1]], shapes[0]))
            accumulate(args[1], emit("channel_grad", [dy, args[0]], shapes[1]))
            if len(args) == 3:
                accumulate(args[2], emit("channel_sum", [dy], shapes[2]))
        elif op == "concat":
            start, axis = 0, attrs["axis"]
            for arg, dims in zip(args, shapes):
                accumulate(arg, emit("slice_axis", [dy], dims, {"axis": axis, "start": start}))
                start += dims[axis]
        elif op == "add":
            for name in args:
                accumulate(name, dy)
        elif op == "mul":
            for index, name in enumerate(args):
                inputs = [dy, args[1 - index]] if len(args) == 2 else [dy]
                accumulate(name, emit("mul", inputs, shapes[index], attrs))
        else:
            raise ModelGraphError(f"native derivative is unavailable for {op!r}")
    input_grads = {}
    parameter_grads = {}
    for spec in [*plan["inputs"], *plan["parameters"]]:
        name = spec["name"]
        if name in grads:
            if spec in plan["inputs"]:
                input_grads[name] = grads[name]
            elif name in trainable:
                parameter_grads[name] = grads[name]
    plan["backward"] = {"steps": steps, "seed_grads": seeds,
                        "grad_inputs": input_grads, "grad_parameters": parameter_grads}


def append_training(plan, *, learning_rate=0.001, loss="mse"):
    """Add a complete native mean-square loss, backward and plain SGD program."""
    if loss != "mse" or type(learning_rate) not in (int, float) or not 0 < learning_rate < float("inf"):
        raise ModelGraphError("training supports MSE and a finite positive SGD learning rate")
    if len(plan.get("outputs", [])) != 1 or "training" in plan:
        raise ModelGraphError("native MSE training requires one tensor output and an unmodified graph")
    from .optimization import validate_storage, storage_layout
    reuse_storage = validate_storage(plan) is not None
    result = copy.deepcopy(plan)
    output = result["outputs"][0]
    target, loss_out, loss_seed = "_oa_target", "_oa_loss", "_oa_loss_seed"
    if any(x in result["buffers"] for x in (target, loss_out, loss_seed)):
        raise ModelGraphError("reserved native training buffer name collides with the model")
    result["buffers"][target] = copy.deepcopy(result["buffers"][output])
    result["inputs"].append({"name": target, **result["buffers"][target]})
    result["buffers"][loss_out] = {"shape": [1], "dtype": "f32"}
    result["buffers"][loss_seed] = {"shape": [1], "dtype": "f32"}
    result["steps"].append({"kernel": "mse", "inputs": [output, target], "output": loss_out, "attrs": {}})
    old_seed = result["backward"]["seed_grads"][output]
    result["backward"]["steps"].insert(0, {"kernel": "mse_grad", "inputs": [output, target, loss_seed],
                                           "output": old_seed, "attrs": {}})
    result["backward"]["seed_grads"] = {loss_out: loss_seed}
    result["outputs"].append(loss_out)
    updates = [{"kernel": "sgd", "inputs": [name, grad], "output": name, "attrs": {"lr": float(learning_rate)}}
               for name, grad in result["backward"]["grad_parameters"].items()]
    if not updates:
        raise ModelGraphError("native training requires at least one trainable parameter")
    result["training"] = {"loss": loss, "target": target, "model_output": output, "loss_output": loss_out,
                          "learning_rate": float(learning_rate), "steps": updates}
    if reuse_storage:
        result['storage'] = storage_layout(result)
    return result
