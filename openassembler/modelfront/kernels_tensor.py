"""Explicit FP16-boundary Tensor Core recipes for native model linear operators.

The graph's storage and output remain FP32. Every matrix operand is explicitly
rounded to binary16 RNE before the existing FP32-accumulating native GEMM.
This numerical contract differs from the scalar FP32 graph and is opt-in.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path

from .kernels import canonical, shape_of
from openassembler.kernels.gemm import GemmPlan, generate as gemm_source
from openassembler.gemm_dispatch import select_plan

MATH_MODE = "f16_rne_f32_accum"


@dataclass(frozen=True)
class TensorStage:
    name: str
    op: str
    inputs: tuple[str, ...]
    output: str
    ptx: str
    launch: dict
    signature: dict
    gemm_plan: GemmPlan | None = None


@dataclass(frozen=True)
class TensorRecipe:
    name: str
    op: str
    signature: dict
    workspaces: dict
    stages: tuple[TensorStage, ...]


def _pack(shape, transpose, input_name, output_name):
    rows, cols = shape
    orows, ocols = (cols, rows) if transpose else (rows, cols)
    if ocols % 2:
        raise ValueError("paired FP16 packing requires an even output width")
    if orows > 65535:
        raise ValueError("FP16 packing output rows exceed the CUDA grid-y limit")
    signature = {"op": "pack_f16_rne", "shape": list(shape), "transpose": transpose,
                 "input_dtype": "f32", "output_dtype": "f16", "nan": "canonical_quiet_preserve_sign"}
    name = "oa_pack_" + hashlib.sha256(canonical(signature)).hexdigest()[:20]
    lines = [".version 8.7\n.target sm_80\n.address_size 64",
             f".visible .entry {name}(.param .u64 input,.param .u64 output) {{",
             ".reg .b32 %r<40>; .reg .b64 %d<8>; .reg .f32 %f<4>; .reg .pred %p<4>;",
             "ld.param.u64 %d0,[input]; ld.param.u64 %d1,[output];",
             "mov.u32 %r0,%tid.x; mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ctaid.y;",
             "mad.lo.u32 %r3,%r1,128,%r0;",
             f"setp.ge.u32 %p0,%r3,{ocols//2}; @%p0 bra done;",
             "mul.lo.u32 %r4,%r3,2;"]
    for j in range(2):
        lines += [f"add.u32 %r5,%r4,{j};"]
        if transpose:
            lines += [f"mad.lo.u32 %r6,%r5,{cols},%r2;"]
        else:
            lines += [f"mad.lo.u32 %r6,%r2,{cols},%r5;"]
        lines += ["mul.wide.u32 %d2,%r6,4; add.u64 %d3,%d0,%d2;",
                  "ld.global.b32 %r7,[%d3];",
                  "shr.u32 %r8,%r7,16; and.b32 %r8,%r8,32768;",
                  "and.b32 %r9,%r7,2147483647;",
                  "shr.u32 %r10,%r9,13; and.b32 %r10,%r10,1;",
                  "add.u32 %r11,%r9,3355447295; add.u32 %r11,%r11,%r10; shr.u32 %r11,%r11,13;",
                  # Subnormal conversion: a correctly rounded addition shifts the
                  # significand to half-subnormal units, including ties-to-even.
                  "mov.b32 %f0,%r9; add.rn.f32 %f1,%f0,0f3f000000; mov.b32 %r12,%f1;",
                  "sub.u32 %r12,%r12,1056964608;",
                  "setp.lt.u32 %p1,%r9,947912704; selp.u32 %r13,%r12,%r11,%p1;",
                  "setp.ge.u32 %p1,%r9,1199566848; selp.u32 %r13,31744,%r13,%p1;",
                  "setp.gt.u32 %p1,%r9,2139095040; selp.u32 %r13,32256,%r13,%p1;",
                  f"or.b32 %r{20+j},%r13,%r8;"]
    lines += ["shl.b32 %r21,%r21,16; or.b32 %r22,%r20,%r21;",
              f"mad.lo.u32 %r23,%r2,{ocols//2},%r3;",
              "mul.wide.u32 %d4,%r23,4; add.u64 %d5,%d1,%d4;",
              "st.global.b32 [%d5],%r22;", "done:", "ret;", "}", ""]
    return TensorStage(name, "pack_f16_rne", (input_name,), output_name, "\n".join(lines),
                       {"grid": [(ocols//2+127)//128, orows, 1], "threads": 128, "smem": 0}, signature)


def emit_tensor_recipe(op, inputs, output, attrs=None):
    """Return guarded pack+GEMM stages; unsupported shapes fail explicitly.

    Bindings are local recipe names a0/a1 and output. Each recipe owns two
    reusable packed workspaces; training refreshes them on every invocation.
    Bias is deliberately a separate FP32 graph add rather than a hidden fusion.
    """
    attrs = dict(attrs or {})
    if attrs.pop("math_mode", None) != MATH_MODE:
        raise ValueError(f"tensor recipes require explicit math_mode={MATH_MODE!r}")
    if attrs:
        raise ValueError(f"unsupported tensor recipe attributes: {sorted(attrs)}")
    if op not in ("linear", "linear_dx", "linear_dw") or len(inputs) not in ((2,3) if op=="linear" else (2,)):
        raise ValueError("tensor recipe requires linear/linear_dx/linear_dw with optional forward bias")
    left, right = (shape_of(x) for x in inputs[:2])
    out = shape_of(output)
    if len(left) != 2 or len(right) != 2 or len(out) != 2:
        raise ValueError("tensor linear operands must be rank-two FP32 matrices")
    if op == "linear":
        m, k = left; n, wk = right
        if k != wk or out != (m, n):
            raise ValueError("linear shape mismatch")
        if len(inputs)==3 and shape_of(inputs[2]) != (n,):
            raise ValueError("linear bias shape mismatch")
        a_shape, b_shape, trans_a, trans_b = left, right, False, False
        gm, gn, gk = m, n, k
    elif op == "linear_dx":
        m, n = left; wn, k = right
        if n != wn or out != (m, k):
            raise ValueError("linear_dx shape mismatch")
        a_shape, b_shape, trans_a, trans_b = left, right, False, True
        gm, gn, gk = m, k, n
    else:
        m, n = left; xm, k = right
        if m != xm or out != (n, k):
            raise ValueError("linear_dw shape mismatch")
        a_shape, b_shape, trans_a, trans_b = left, right, True, True
        gm, gn, gk = n, k, m
    tile = "64x64" if gm % 128 or gn % 128 else "128x128"
    plan, decision = select_plan(GemmPlan(gm, gn, gk, "f16", tile=tile))
    signature = {"op": op, "inputs": [list(shape_of(x)) for x in inputs], "output": list(out),
                 "math_mode": MATH_MODE, "cast_derivative": "straight_through_identity",
                 "plan": asdict(plan), "policy": decision["policy"]}
    identity = hashlib.sha256(canonical(signature)).hexdigest()[:20]
    name = "oa_tensor_" + identity
    pa, pb = name + "_packed_a", name + "_packed_b"
    workspaces = {pa: {"shape": [gm, gk], "dtype": "f16", "alignment": 16},
                  pb: {"shape": [gn, gk], "dtype": "f16", "alignment": 16}}
    first = _pack(a_shape, trans_a, "a0", pa)
    second = _pack(b_shape, trans_b, "a1", pb)
    gemm_name = name + "_gemm"
    source = gemm_source(plan).replace(".entry gemm(", f".entry {gemm_name}(")
    gemm = TensorStage(gemm_name, op, (pa, pb), "output", source, plan.launch,
                       {"op": op, "math_mode": MATH_MODE, "plan": asdict(plan)}, plan)
    stages = (first,second,gemm)
    if len(inputs)==3:
        bias_name=name+"_bias"
        bias_signature={"op":"bias_add", "shape":list(out), "dtype":"f32"}
        bias_ptx=f""".version 8.7
.target sm_80
.address_size 64
.visible .entry {bias_name}(.param .u64 input,.param .u64 bias,.param .u64 output) {{
 .reg .b32 %r<8>; .reg .b64 %d<10>; .reg .f32 %f<4>; .reg .pred %p;
 ld.param.u64 %d0,[input]; ld.param.u64 %d1,[bias]; ld.param.u64 %d2,[output];
 mov.u32 %r0,%tid.x; mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ctaid.y;
 mad.lo.u32 %r3,%r1,128,%r0;
 setp.ge.u32 %p,%r3,{out[1]}; @%p bra done;
 mad.lo.u32 %r4,%r2,{out[1]},%r3;
 mul.wide.u32 %d3,%r4,4; add.u64 %d4,%d0,%d3; add.u64 %d5,%d2,%d3;
 mul.wide.u32 %d6,%r3,4; add.u64 %d7,%d1,%d6;
 ld.global.f32 %f0,[%d4]; ld.global.f32 %f1,[%d7];
 add.rn.f32 %f2,%f0,%f1; st.global.f32 [%d5],%f2;
done: ret;
}}
"""
        stages += (TensorStage(bias_name,"bias_add",("output","a2"),"output",bias_ptx,
                               {"grid":[(out[1]+127)//128,out[0],1],"threads":128,"smem":0},bias_signature),)
    return TensorRecipe(name, op, signature, workspaces, stages)


def build_tensor_kernels(root, recipes):
    """Build only OA native text and return per-entry module ownership records."""
    from openassembler import oracle
    from openassembler.backend.tensor import TensorConfig
    from openassembler.build import generate as native_generate, ptxas_build, disasm
    from openassembler.container.cubin import splice_text, text_words, regcount
    from openassembler.container.elf import Elf
    from openassembler.ptx.parse import parse_module
    root = Path(root).resolve()
    records = {}
    for recipe in recipes:
        for stage in recipe.stages:
            if stage.name in records:
                if records[stage.name]["signature"] != stage.signature:
                    raise ValueError("tensor kernel entry identity collision")
                continue
            directory = root / "native_tensor" / stage.name
            directory.mkdir(parents=True, exist_ok=True)
            source = directory / "kernel.ptx"; source.write_text(stage.ptx)
            reference = directory / "reference.cubin"
            reference.write_bytes(ptxas_build(stage.ptx,"sm_80"))
            kernel = parse_module(stage.ptx).kernels[stage.name]
            config = (TensorConfig(pointer_mode=stage.gemm_plan.pointer_mode,
                                   pointer_advance=stage.gemm_plan.pointer_advance)
                      if stage.gemm_plan else None)
            code, labels, words, exits, registers = native_generate(kernel, "sm_80", "opt", tensor_config=config)
            binary = splice_text(reference.read_bytes(), stage.name, words, registers, exits, replace_helpers=True)
            path = directory / "native.cubin"; path.write_bytes(binary)
            sass = disasm(binary)
            (directory / "native.sass").write_text(sass)
            sections = {s.name[6:] for s in Elf.parse(binary).sections if s.name.startswith(".text.") and s.data}
            if sections != {stage.name} or text_words(binary, stage.name) != words:
                raise ValueError("tensor module contains unowned or changed executable text")
            records[stage.name] = {"entry": stage.name, "op": stage.op,
                                    "module": str(path.relative_to(root)),
                                    "cubin_sha256": hashlib.sha256(binary).hexdigest(),
                                    "text_sha256": hashlib.sha256(b"".join(w.to_bytes(16,"little") for w in words)).hexdigest(),
                                    "source_sha256": hashlib.sha256(stage.ptx.encode()).hexdigest(),
                                    "compiled_by": "openassembler", "registers": regcount(binary,stage.name),
                                    "launch": stage.launch, "signature": stage.signature}
    return records
