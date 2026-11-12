"""Shape-specialized FP32 baseline kernels for the native tensor graph route.

These deliberately simple kernels establish semantic coverage. They do not use
tensor cores or claim competitive matrix multiplication performance. In particular
all indexing avoids the backend's unvalidated runtime division/remainder path.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
from pathlib import Path


MAX_ELEMENTS = 16 * 1024 * 1024
OPS = frozenset({"linear", "linear_dx", "linear_dw", "bias_grad", "relu",
                 "relu_backward", "add", "mul", "copy", "transpose2d", "fill",
                 "sgd", "mse", "mse_grad", "softmax", "softmax_backward", "gelu", "gelu_backward"})


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def shape_of(spec):
    if not isinstance(spec, dict) or spec.get("dtype", "f32") not in ("f32", "float32"):
        raise ValueError("native model tensors require f32 dtype")
    shape = spec.get("shape")
    if not isinstance(shape, (list, tuple)) or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError("tensor shapes require positive integer dimensions")
    if math.prod(shape) > MAX_ELEMENTS:
        raise ValueError(f"native model tensor exceeds {MAX_ELEMENTS} elements")
    return tuple(shape)


def _float(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("scalar must be a finite FP32 number")
    try:
        raw = struct.pack(">f", value)
    except (OverflowError, struct.error):
        raise ValueError("scalar must fit finite FP32") from None
    if not math.isfinite(struct.unpack(">f", raw)[0]):
        raise ValueError("scalar must fit finite FP32")
    return "0f" + raw.hex()


@dataclass(frozen=True)
class KernelSpec:
    name: str
    op: str
    inputs: tuple[tuple[int, ...], ...]
    output: tuple[int, ...]
    attrs: dict
    grid: tuple[int, int, int]
    threads: int
    ptx: str

    @property
    def launch(self):
        return {"grid": list(self.grid), "threads": self.threads, "smem": 0}

    @property
    def signature(self):
        return {"op": self.op, "inputs": [list(x) for x in self.inputs],
                "output": list(self.output), "attrs": self.attrs}


def emit_kernel(op, inputs, output, attrs=None):
    """Validate a complete op contract and return deterministic supported PTX."""
    from .kernels_concat import CONCAT_OPS, emit_concat_kernel
    if op in CONCAT_OPS:
        return emit_concat_kernel(op, inputs, output, attrs)
    from .kernels_batched import BATCHED_OPS, emit_batched_kernel
    if op in BATCHED_OPS:
        return emit_batched_kernel(op, inputs, output, attrs)
    from .kernels_nn import NN_OPS, emit_nn_kernel
    if op in NN_OPS:
        return emit_nn_kernel(op,inputs,output,attrs)
    if op not in OPS:
        raise ValueError(f"unsupported native model operator: {op}")
    shapes = tuple(shape_of(x) for x in inputs)
    out = shape_of(output)
    attrs = dict(attrs or {})
    # Shape keys are redundant graph annotations; dimensions are validated from
    # tensor metadata, never trusted as a replacement for those metadata.
    permitted = {"m", "n", "k", "input_shape", "output_shape"}
    if op in ("add", "mul"):
        permitted.add("scalar")
    if op == "fill":
        permitted.add("value")
    if op == "sgd":
        permitted.add("lr")
    if set(attrs) - permitted:
        raise ValueError(f"unsupported {op} attributes: {sorted(set(attrs)-permitted)}")
    count = math.prod(out)
    arity = {"linear": (2, 3), "linear_dx": (2,), "linear_dw": (2,),
             "bias_grad": (1,), "relu": (1,), "relu_backward": (2,),
             "add": (1, 2), "mul": (1, 2), "copy": (1,), "transpose2d": (1,),
             "fill": (0,), "sgd": (2,), "mse": (2,), "mse_grad": (2, 3)}[op]
    if len(shapes) not in arity:
        raise ValueError(f"{op} expects {arity} inputs")
    if op in ("add", "mul") and (len(shapes) == 1) != ("scalar" in attrs):
        raise ValueError(f"{op} needs two tensors or one tensor and scalar")
    if op in ("add", "mul", "relu", "relu_backward", "sgd"):
        if any(x != out for x in shapes):
            raise ValueError(f"{op} requires identical tensor shapes (broadcasting unsupported)")
    if op == "copy" and math.prod(shapes[0]) != count:
        raise ValueError("copy/reshape must preserve the element count")
    if op == "transpose2d" and (len(shapes[0]) != 2 or out != shapes[0][::-1]):
        raise ValueError("transpose2d requires reversed rank-two shape")
    if op in ("mse", "mse_grad"):
        if shapes[0] != shapes[1] or (op == "mse" and count != 1):
            raise ValueError("MSE inputs must match and loss must be scalar")
        if op == "mse_grad" and (out != shapes[0] or (len(shapes) == 3 and math.prod(shapes[2]) != 1)):
            raise ValueError("MSE gradient needs matching output and scalar upstream")
    if op in ("linear", "linear_dx", "linear_dw"):
        if any(len(x) != 2 for x in shapes[:2]) or len(out) != 2:
            raise ValueError("linear kernels require rank-two matrices")
        if op == "linear":
            m, k = shapes[0]; n, kw = shapes[1]
            expected = (m, n)
            if kw != k or (len(shapes) == 3 and shapes[2] != (n,)):
                raise ValueError("linear weight/bias shape mismatch")
        elif op == "linear_dx":
            m, n = shapes[0]; nw, k = shapes[1]; expected = (m, k)
            if n != nw:
                raise ValueError("linear_dx weight shape mismatch")
        else:
            m, n = shapes[0]; mx, k = shapes[1]; expected = (n, k)
            if m != mx:
                raise ValueError("linear_dw batch shape mismatch")
        if out != expected:
            raise ValueError(f"{op} output shape mismatch: expected {expected}")
        for key, value in (("m", m), ("n", n), ("k", k)):
            if key in attrs and attrs[key] != value:
                raise ValueError(f"{op} attribute {key} disagrees with tensor shapes")
    if op == "bias_grad" and (len(shapes[0]) != 2 or out != (shapes[0][1],)):
        raise ValueError("bias_grad expects [rows,width] -> [width]")
    for key in ("scalar", "value", "lr"):
        if key in attrs:
            _float(attrs[key])
    signature = {"op": op, "inputs": shapes, "output": out, "attrs": attrs}
    name = "oa_model_" + hashlib.sha256(canonical(signature)).hexdigest()[:20]
    params = [f".param .u64 a{i}" for i in range(len(shapes))] + [".param .u64 output"]
    if op == "sgd":
        params.append(".param .f32 rate")
    lines = [f".visible .entry {name}({','.join(params)}) {{",
             ".reg .b32 %r<16>; .reg .b64 %d<16>; .reg .f32 %f<8>; .reg .pred %p<4>;"]
    for i in range(len(shapes)):
        lines.append(f"ld.param.u64 %d{i},[a{i}];")
    lines.append("ld.param.u64 %d3,[output];")
    lines += ["mov.u32 %r0,%tid.x; mov.u32 %r1,%ctaid.x;",
              "mad.lo.u32 %r2,%r1,128,%r0;"]
    grid, threads = ((count+127)//128, 1, 1), 128

    def load(base, index, dest):
        lines.extend([f"mul.wide.u32 %d8,{index},4; add.u64 %d9,%d{base},%d8;",
                      f"ld.global.f32 %f{dest},[%d9];"])

    def store(index, value="%f0"):
        lines.extend([f"mul.wide.u32 %d8,{index},4; add.u64 %d9,%d3,%d8;",
                      f"st.global.f32 [%d9],{value};"])

    if op in ("linear", "linear_dx", "linear_dw", "transpose2d"):
        if out[0] > 65535:
            raise ValueError("matrix output rows exceed CUDA grid-y limit")
        rows, cols = out
        grid = ((cols+127)//128, rows, 1)
        lines += ["mov.u32 %r3,%ctaid.y;", f"setp.ge.u32 %p0,%r2,{cols}; @%p0 bra done;"]
        if op == "transpose2d":
            lines += [f"mad.lo.u32 %r7,%r2,{rows},%r3;"]
            load(0, "%r7", 0)
        else:
            length = k if op == "linear" else n if op == "linear_dx" else m
            lines += ["mov.f32 %f0,0f00000000; mov.u32 %r6,0;", "loop:"]
            if op == "linear":
                lines += [f"mad.lo.u32 %r7,%r3,{k},%r6;", f"mad.lo.u32 %r8,%r2,{k},%r6;"]
            elif op == "linear_dx":
                lines += [f"mad.lo.u32 %r7,%r3,{n},%r6;", f"mad.lo.u32 %r8,%r6,{k},%r2;"]
            else:
                lines += [f"mad.lo.u32 %r7,%r6,{n},%r3;", f"mad.lo.u32 %r8,%r6,{k},%r2;"]
            load(0, "%r7", 1); load(1, "%r8", 2)
            lines += ["mul.rn.f32 %f3,%f1,%f2; add.rn.f32 %f0,%f0,%f3;",
                      f"add.u32 %r6,%r6,1; setp.lt.u32 %p1,%r6,{length}; @%p1 bra loop;"]
            if op == "linear" and len(shapes) == 3:
                load(2, "%r2", 1)
                lines += ["add.rn.f32 %f0,%f0,%f1;"]
        lines += [f"mad.lo.u32 %r9,%r3,{cols},%r2;"]
        store("%r9")
    elif op in ("bias_grad", "mse"):
        length = shapes[0][0] if op == "bias_grad" else math.prod(shapes[0])
        if op == "mse":
            grid, threads = (1, 1, 1), 1
            lines += ["mov.u32 %r2,0;"]
        lines += [f"setp.ge.u32 %p0,%r2,{count}; @%p0 bra done;",
                  "mov.f32 %f0,0f00000000; mov.u32 %r6,0;", "loop:"]
        if op == "bias_grad":
            lines += [f"mad.lo.u32 %r7,%r6,{shapes[0][1]},%r2;"]
            load(0, "%r7", 1)
        else:
            load(0, "%r6", 1); load(1, "%r6", 2)
            lines += ["sub.rn.f32 %f1,%f1,%f2; mul.rn.f32 %f1,%f1,%f1;"]
        lines += ["add.rn.f32 %f0,%f0,%f1;", f"add.u32 %r6,%r6,1; setp.lt.u32 %p1,%r6,{length}; @%p1 bra loop;"]
        if op == "mse":
            lines += [f"mul.rn.f32 %f0,%f0,{_float(1/length)};"]
        store("%r2")
    else:
        lines += [f"setp.ge.u32 %p0,%r2,{count}; @%p0 bra done;"]
        for i in range(min(len(shapes), 2)):
            load(i, "%r2", i)
        if op == "fill":
            lines += [f"mov.f32 %f0,{_float(attrs.get('value', 1.0))};"]
        elif op == "relu":
            lines += ["max.f32 %f0,%f0,0f00000000;"]
        elif op == "relu_backward":
            lines += ["setp.gt.f32 %p1,%f1,0f00000000; selp.f32 %f0,%f0,0f00000000,%p1;"]
        elif op in ("add", "mul"):
            if len(shapes) == 1:
                lines += [f"mov.f32 %f1,{_float(attrs['scalar'])};"]
            lines += [f"{op}.rn.f32 %f0,%f0,%f1;"]
        elif op == "sgd":
            lines += ["ld.param.f32 %f2,[rate]; mul.rn.f32 %f1,%f1,%f2; sub.rn.f32 %f0,%f0,%f1;"]
        elif op == "mse_grad":
            lines += [f"sub.rn.f32 %f0,%f0,%f1; mul.rn.f32 %f0,%f0,{_float(2/count)};"]
            if len(shapes) == 3:
                lines += ["ld.global.f32 %f1,[%d2]; mul.rn.f32 %f0,%f0,%f1;"]
        store("%r2")
    lines += ["done:", "ret;", "}"]
    kernel = KernelSpec(name, op, shapes, out, attrs, grid, threads, "\n".join(lines)+"\n")
    if op == 'mse':
        from .kernels_loss import ORDERED_MSE_ELEMENTS, ordered_mse
        if math.prod(shapes[0]) in ORDERED_MSE_ELEMENTS:
            return ordered_mse(kernel)
    return kernel


def tensor_specs(plan):
    specs = {}
    for spec in list(plan.get("inputs", [])) + list(plan.get("parameters", [])):
        if spec["name"] in specs:
            raise ValueError("duplicate tensor name")
        specs[spec["name"]] = spec
    for name, spec in plan.get("buffers", {}).items():
        if name in specs:
            if shape_of(specs[name]) != shape_of(spec):
                raise ValueError("buffer metadata disagrees with input/parameter")
            continue
        specs[name] = spec
    for spec in specs.values():
        shape_of(spec)
    return specs


def tensor_recipes(plan, specs=None):
    mode = plan.get("math", "fp32")
    if mode not in ("fp32", "f16_rne_f32_accum"):
        raise ValueError(f"unsupported native model numerical policy: {mode}")
    if mode == "fp32":
        return {}
    from .kernels_tensor import emit_tensor_recipe
    specs = specs or tensor_specs(plan)
    recipes = {}
    for step in plan.get("steps", []) + plan.get("backward", {}).get("steps", []):
        if step["kernel"] in ("linear", "linear_dx", "linear_dw"):
            recipe = emit_tensor_recipe(step["kernel"], [specs[x] for x in step["inputs"]],
                                        specs[step["output"]], {"math_mode":mode})
            recipes[recipe.name] = recipe
    return recipes


def prepare(plan):
    """Return validated executable steps and unique kernel contracts, including training helpers."""
    if plan.get("schema") != 1 or plan.get("arch","sm_80") != "sm_80":
        raise ValueError("unsupported native model plan schema")
    specs = tensor_specs(plan)
    from .optimization import validate_aliases, validate_storage
    aliases = validate_aliases(plan)
    validate_storage(plan)
    recipes = tensor_recipes(plan, specs)
    available = {s["name"] for s in plan.get("inputs", []) + plan.get("parameters", [])}
    steps = []
    kernels = {}
    for phase, raw_steps in (("forward", plan.get("steps", [])),
                             ("backward", plan.get("backward", {}).get("steps", []))):
        if phase == "backward":
            available.update(plan.get("backward", {}).get("seed_grads", {}).values())
        for step in raw_steps:
            names = step.get("inputs", [])
            output = step.get("output")
            if output not in specs or any(x not in available for x in names):
                raise ValueError("native plan has undefined tensor use/output")
            if output in names or output in available:
                raise ValueError("native graph steps require unique, non-aliasing output buffers")
            if output in aliases:
                steps.append({**step, 'phase': phase, 'alias': True})
                available.add(output)
                continue
            kernel = emit_kernel(step["kernel"], [specs[x] for x in names], specs[output], step.get("attrs"))
            if recipes and step["kernel"] in ("linear", "linear_dx", "linear_dw"):
                from .kernels_tensor import emit_tensor_recipe
                recipe = emit_tensor_recipe(step["kernel"], [specs[x] for x in names], specs[output],
                                            {"math_mode":plan["math"]})
                steps.append({**step, "phase": phase, "entry":recipe.name, "tensor":True})
            else:
                kernels[kernel.name] = kernel
                steps.append({**step, "phase": phase, "entry": kernel.name})
            available.add(output)
        if phase == "forward" and any(x not in available for x in plan.get("outputs", [])):
            raise ValueError("native plan output is not defined")
    backward = plan.get("backward", {})
    for output, seed in backward.get("seed_grads", {}).items():
        if output not in plan.get("outputs", []) or seed not in specs or shape_of(specs[output]) != shape_of(specs[seed]):
            raise ValueError("invalid backward seed contract")
        for value in (0.0, 1.0):
            kernel = emit_kernel("fill", [], specs[seed], {"value": value})
            kernels[kernel.name] = kernel
    for category in ("grad_inputs", "grad_parameters"):
        for original, gradient in backward.get(category, {}).items():
            if original not in specs or gradient not in available or shape_of(specs[original]) != shape_of(specs[gradient]):
                raise ValueError("invalid backward gradient contract")
    for output in plan.get("outputs", []):
        for op in ("mse", "mse_grad"):
            kernel = emit_kernel(op, [specs[output]]*2, {"shape": []} if op == "mse" else specs[output])
            kernels[kernel.name] = kernel
    for name in backward.get("grad_parameters", {}):
        kernel = emit_kernel("sgd", [specs[name]]*2, specs[name])
        kernels[kernel.name] = kernel
    return specs, steps, kernels


def build_bundle(plan, weights=None, output_dir=None, *, opt_level="opt", sass_listings=True):
    """Compile all graph kernels with OA; ptxas is only the donor/reference tool.

    Accept ``build_bundle(plan, directory)`` or the frontend's three-argument
    spelling. Weight serialization belongs to the calling project worker.
    """
    if output_dir is None:
        output_dir, weights = weights, None
    root = Path(output_dir).resolve()
    from openassembler import build
    from openassembler.container.elf import Elf, EM_CUDA
    specs, steps, kernels = prepare(plan)
    if not kernels:
        raise ValueError("native model plan contains no executable kernels")
    source = root / "sources" / "native.ptx"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(".version 8.7\n.target sm_80\n.address_size 64\n" + "\n".join(kernels[name].ptx for name in sorted(kernels)))
    if type(sass_listings) is not bool:
        raise ValueError('sass_listings must be a boolean')
    result = build.compile_paths(source, root / "native", build.Options(
        arch="sm_80", opt_level=opt_level, fallback=None, no_sass=not sass_listings))
    if result.rc:
        raise RuntimeError("OA native model kernel compilation failed; no fallback is allowed")
    manifest = json.loads((root / "native" / "manifest.json").read_text())
    entries = manifest.get("sources", [])
    if not entries:
        entries = manifest.get("entries", [])
    if len(entries) != 1 or entries[0].get("failed"):
        raise RuntimeError("native model build is incomplete")
    entry = entries[0]
    if {k["name"] for k in entry["kernels"]} != set(kernels) or any(k.get("compiled_by") != "openassembler" for k in entry["kernels"]):
        raise RuntimeError("native model build contains unknown or non-OA kernels")
    cubin_path = root / "native" / entry["cubin"]
    raw = cubin_path.read_bytes(); elf = Elf.parse(raw)
    if elf.machine != EM_CUDA or elf.flags & 255 != 80:
        raise RuntimeError("native model requires SM80 CUDA cubin")
    sections = {s.name[6:]: s for s in elf.sections if s.name.startswith(".text.") and s.data}
    if set(sections) != set(kernels):
        raise RuntimeError("native cubin contains unknown executable functions")
    records = {}
    for name, kernel in kernels.items():
        records[name] = {"compiled_by": "openassembler", "signature": kernel.signature,
                         "launch": kernel.launch, "text_sha256": hashlib.sha256(sections[name].data).hexdigest(),
                         "registers": next(x["registers"] for x in entry["kernels"] if x["name"] == name)}
    ledger = {"schema": 1, "backend": "openassembler-native", "arch": "sm_80",
              "plan_sha256": hashlib.sha256(canonical(plan)).hexdigest(),
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "cubin": cubin_path.relative_to(root).as_posix(), "cubin_sha256": hashlib.sha256(raw).hexdigest(),
              "kernels": records, "steps": steps,
              "numerical_contract": "FP32 separate multiply/add; serial reductions; finite bounded inputs; no tensor-core substitution"}
    recipes = tensor_recipes(plan,specs)
    if recipes:
        from .kernels_tensor import build_tensor_kernels
        ledger["tensor_kernels"] = build_tensor_kernels(root,recipes.values(),sass_listings=sass_listings)
        ledger["tensor_recipes"] = {name:recipe_record(recipe) for name,recipe in recipes.items()}
        ledger["numerical_contract"] = "FP16 RNE operands and FP32 tensor accumulation/output; straight-through cast derivative; other arithmetic FP32"
    (root / "plan.json").write_bytes(canonical(plan))
    (root / "native-ledger.json").write_bytes(canonical(ledger))
    if weights is not None:
        from .runtime import save_weights
        save_weights(weights, plan, root)
    return ledger


def recipe_record(recipe):
    return {"signature":recipe.signature,"workspaces":recipe.workspaces,
            "stages":[{"entry":s.name,"inputs":list(s.inputs),"output":s.output} for s in recipe.stages]}
