"""Bounded FP16-RNE batched products, including both backward orientations.

Storage, accumulators and outputs remain FP32. Larger products use the existing
OA Tensor Core mainloop with CTA-z batch strides and dual padded layouts. Small
products round once into FP32 workspaces and use the scalar OA BMM, avoiding the
128x128 padding cost. The threshold and both paths are measured in
exp/attention_half; neither path delegates GPU arithmetic to a vendor compiler.
"""

from dataclasses import replace, asdict
import hashlib, math, re
from openassembler.modelfront.kernels import canonical, shape_of
from openassembler.modelfront.kernels_tensor import TensorStage, TensorRecipe, MATH_MODE
from openassembler.modelfront.kernels_tensor_dual import dual_pack, crop
from openassembler.kernels.gemm import GemmPlan, generate


def batched_stage(stage, batch, strides, transpose_offset=None):
    signature = {
        **stage.signature,
        "batch": batch,
        "batch_strides_bytes": list(strides),
    }
    name = "oa_batched_" + hashlib.sha256(canonical(signature)).hexdigest()[:20]
    ptx = stage.ptx.replace(stage.name, name)
    ptx = ptx.replace(
        ".reg .b64", ".reg .b32 %batch_id; .reg .b64 %batch_offset;\n.reg .b64", 1
    )
    pointer_index = 0
    pattern = r"(ld\.param\.u64\s+(%d\d+),\[([^\]]+)\];)"

    def inject(match):
        nonlocal pointer_index
        stride = strides[pointer_index]
        pointer_index += 1
        return (
            match[1]
            + f" mov.u32 %batch_id,%ctaid.z; mul.wide.u32 %batch_offset,%batch_id,{stride}; add.u64 {match[2]},{match[2]},%batch_offset;"
        )

    ptx = re.sub(pattern, inject, ptx)
    if pointer_index != len(strides):
        raise ValueError("batch pointer stride count mismatch")
    if transpose_offset is not None:
        old, new = transpose_offset
        instruction = f"add.u32 %r15,%r15,{old};"
        if ptx.count(instruction) != 2:
            raise ValueError("dual-pack transpose offset anchor changed")
        ptx = ptx.replace(instruction, f"add.u32 %r15,%r15,{new};")
    return replace(
        stage,
        name=name,
        ptx=ptx,
        launch={**stage.launch, "grid": [*stage.launch["grid"][:2], batch]},
        signature=signature,
    )


def _tensor_recipe(op, inputs, output):
    shapes = [shape_of(s) for s in inputs]
    out = shape_of(output)
    if len(shapes) != 2 or any(len(s) != 3 for s in [*shapes, out]):
        raise ValueError("BMM requires rank-three operands")
    a, b = shapes
    batch = a[0]
    if batch != b[0] or batch != out[0] or batch > 32:
        raise ValueError("BMM equal batch count at most 32 required")
    if op == "bmm":
        _, m, k = a
        n = b[2]
        valid = b[1] == k
        ta, tb = False, True
        operands = shapes
        order = (0, 1)
    elif op == "bmm_dx":
        _, m, k = a
        n = b[1]
        valid = b[2] == k
        ta, tb = False, False
        operands = shapes
        order = (0, 1)
    elif op == "bmm_dw":
        _, k, n = a
        m = b[2]
        valid = b[1] == k
        ta, tb = True, True
        operands = [b, a]
        order = (1, 0)
    else:
        raise ValueError(op)
    if not valid or out != (batch, m, n) or max(m, n, k) > 512:
        raise ValueError("bounded BMM shape mismatch")
    pm, pn, pk = [(v + 127) // 128 * 128 for v in (m, n, k)]
    plan = GemmPlan(pm, pn, pk, "f16")
    signature = {
        "op": op,
        "inputs": [list(s) for s in shapes],
        "output": list(out),
        "math_mode": MATH_MODE,
        "plan": asdict(plan),
        "packing": "dual",
        "batch": batch,
    }
    name = "oa_bmm_" + hashlib.sha256(canonical(signature)).hexdigest()[:20]
    work = {}
    stages = []
    gemm_inputs = []
    for i, (shape, tr, index) in enumerate(zip(operands, (ta, tb), order)):
        base = name + f"_dual{i}"
        view = name + f"_view{i}"
        _, rows, cols = shape
        stage, pr, pc = dual_pack((rows, cols), base)
        stage = batched_stage(
            stage, batch, [rows * cols * 4, pr * pc * 2], (pr * pc, batch * pr * pc)
        )
        stage = replace(
            stage,
            inputs=(f"a{index}",),
            signature={**stage.signature, "shape": list(shape)},
        )
        work[base] = {"shape": [2 * batch * pr * pc], "dtype": "f16", "alignment": 16}
        work[view] = {
            "shape": [batch, pc, pr] if tr else [batch, pr, pc],
            "dtype": "f16",
            "base": base,
            "offset": batch * pr * pc if tr else 0,
        }
        stages.append(stage)
        gemm_inputs.append(view)
    destination = name + "_output"
    work[destination] = {"shape": [batch, pm, pn], "dtype": "f32", "alignment": 16}
    entry = name + "_gemm"
    source = generate(plan).replace(".entry gemm(", f".entry {entry}(")
    gemm = TensorStage(
        entry,
        op,
        tuple(gemm_inputs),
        destination,
        source,
        plan.launch,
        {"op": op, "plan": asdict(plan), "math_mode": MATH_MODE},
        plan,
    )
    stages.append(batched_stage(gemm, batch, [pm * pk * 2, pn * pk * 2, pm * pn * 4]))
    stages.append(
        batched_stage(
            crop((m, n), (pm, pn), False, destination), batch, [pm * pn * 4, m * n * 4]
        )
    )
    return TensorRecipe(name, op, signature, work, tuple(stages))


def rounded_operand(shape, output):
    # For normal half values, round away the low 13 FP32 mantissa bits with
    # ties-to-even. In the half-subnormal range, adding/subtracting 0.5 rounds
    # to multiples of 2**-24. Restore the sign after rounding; the graph's
    # finite-after-half-cast contract excludes overflow and exceptional values.
    signature = {"op": "round_f16_f32", "shape": list(shape), "math_mode": MATH_MODE}
    name = "oa_round_" + hashlib.sha256(canonical(signature)).hexdigest()[:20]
    count = math.prod(shape)
    ptx = f""" .version 8.7
.target sm_80
.address_size 64
.visible .entry {name}(.param .u64 input,.param .u64 output) {{
.reg .b32 %r<16>; .reg .b64 %d<4>; .reg .f32 %f<4>; .reg .pred %p<3>;
ld.param.u64 %d0,[input]; ld.param.u64 %d1,[output];
mov.u32 %r0,%tid.x; mov.u32 %r1,%ctaid.x; mad.lo.u32 %r2,%r1,128,%r0;
setp.ge.u32 %p0,%r2,{count}; @%p0 bra done;
mul.wide.u32 %d2,%r2,4; add.u64 %d3,%d0,%d2; ld.global.b32 %r3,[%d3];
and.b32 %r4,%r3,2147483647; and.b32 %r5,%r3,2147483648;
shr.u32 %r6,%r4,13; and.b32 %r6,%r6,1;
add.u32 %r7,%r4,4095; add.u32 %r7,%r7,%r6; and.b32 %r7,%r7,4294959104;
mov.b32 %f0,%r4; add.rn.f32 %f1,%f0,0f3f000000; sub.rn.f32 %f2,%f1,0f3f000000; mov.b32 %r8,%f2;
setp.lt.u32 %p1,%r4,947912704; selp.u32 %r9,%r8,%r7,%p1;
setp.ge.u32 %p1,%r4,1199566848; selp.u32 %r9,2139095040,%r9,%p1;
or.b32 %r10,%r9,%r5; add.u64 %d3,%d1,%d2; st.global.b32 [%d3],%r10;
done: ret;
}}
"""
    return TensorStage(
        name,
        "round_f16_f32",
        ("a0",),
        output,
        ptx,
        {"grid": [(count + 127) // 128, 1, 1], "threads": 128, "smem": 0},
        signature,
    )


def emit_batched_recipe(op, inputs, output):
    rec = _tensor_recipe(op, inputs, output)
    # Logical reduction length, independent of derivative orientation.
    shapes = [shape_of(s) for s in inputs]
    out = shape_of(output)
    length = shapes[0][-1] if op != "bmm_dw" else shapes[0][-2]
    if max(out[-2], out[-1], length) > 64:
        return rec
    from openassembler.modelfront.kernels_batched import emit_batched_kernel

    work = {}
    stages = []
    bindings = []
    for i, shape in enumerate(shapes):
        dest = rec.name + f"_rounded{i}"
        stage = rounded_operand(shape, dest)
        stages.append(replace(stage, inputs=(f"a{i}",)))
        work[dest] = {"shape": list(shape), "dtype": "f32", "alignment": 16}
        bindings.append(dest)
    kernel = emit_batched_kernel(op, inputs, output)
    stages.append(
        TensorStage(
            kernel.name,
            op,
            tuple(bindings),
            "output",
            ".version 8.7\n.target sm_80\n.address_size 64\n" + kernel.ptx,
            kernel.launch,
            kernel.signature,
        )
    )
    return replace(
        rec,
        name=rec.name + "_scalar",
        signature={**rec.signature, "implementation": "rounded_scalar"},
        workspaces=work,
        stages=tuple(stages),
    )
