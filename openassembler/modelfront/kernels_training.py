"""Exact model contracts admitted by the November 19 A100 full-task campaign.

Bias gradients split the row sum across four warps with coalesced column loads.
The 16K MSE uses 32 partial warp sums and one CTA barrier. Both reductions use
explicit FP32 adds and the existing FP32 tolerance; association differs from the
serial baseline. No cross-CTA atomics or vendor compute are involved.
"""

from dataclasses import replace
import hashlib
from .kernels import canonical

# Only these full matrix contracts passed paired complete model training and
# independent output/guard/replay gates. Leave unmeasured shapes to dispatch.
MODEL_HALF_TILE64 = frozenset(
    {(128, 128, 128), (128, 256, 128), (128, 128, 256), (256, 128, 128)}
)
BIAS_REDUCTION_SHAPES = frozenset({(128, 128), (128, 256)})


def select_tensor_plan(plan, decision):
    if (
        (plan.m, plan.n, plan.k) in MODEL_HALF_TILE64
        and plan.dtype == "f16"
        and plan.tile == "128x128"
        and decision.get("policy") == "baseline"
    ):
        return replace(plan, tile="64x64"), {
            **decision,
            "policy": "model_small_grid_tile64",
            "reason": "exact A100 model contract with complete training admission",
        }
    return plan, decision


def bias_reduce(base, warps=4):
    rows, cols = base.inputs[0]
    threads = 32 * warps
    sig = {
        "baseline": base.signature,
        "warps": warps,
        "implementation": "coalesced_row_partials",
    }
    name = "oa_bias_reduce_" + hashlib.sha256(canonical(sig)).hexdigest()[:20]
    ptx = f""".visible .entry {name}(.param .u64 a0,.param .u64 output) {{
.reg .b32 %r<12>; .reg .b64 %d<6>; .reg .f32 %f<4>; .reg .pred %p<4>;
.shared .align 4 .b8 partials[{threads*4}];
ld.param.u64 %d0,[a0]; ld.param.u64 %d1,[output];
mov.u32 %r0,%tid.x; and.b32 %r1,%r0,31; shr.u32 %r2,%r0,5;
mov.u32 %r3,%ctaid.x; mad.lo.u32 %r3,%r3,32,%r1;
mov.u32 %r4,%r2; mov.f32 %f0,0f00000000;
loop: setp.ge.u32 %p0,%r4,{rows}; @%p0 bra loaded;
mad.lo.u32 %r5,%r4,{cols},%r3; mul.wide.u32 %d2,%r5,4; add.u64 %d3,%d0,%d2;
setp.lt.u32 %p1,%r3,{cols}; mov.f32 %f1,0f00000000; @%p1 ld.global.f32 %f1,[%d3];
add.rn.f32 %f0,%f0,%f1; add.u32 %r4,%r4,{warps}; bra loop;
loaded: mov.u32 %r6,partials; shl.b32 %r7,%r0,2; add.u32 %r7,%r6,%r7;
st.shared.f32 [%r7],%f0; bar.sync 0;
setp.ne.u32 %p2,%r2,0; @%p2 bra done;
mov.f32 %f0,0f00000000; mad.lo.u32 %r7,%r1,4,%r6;
"""
    for w in range(warps):
        ptx += f"add.u32 %r8,%r7,{w*128}; ld.shared.f32 %f1,[%r8]; add.rn.f32 %f0,%f0,%f1;\n"
    ptx += f"""mul.wide.u32 %d2,%r3,4; add.u64 %d3,%d1,%d2;
@%p1 st.global.f32 [%d3],%f0;
done: ret; }}
"""
    return replace(
        base, name=name, ptx=ptx, grid=((cols + 31) // 32, 1, 1), threads=threads
    )


def warp_mse(base, threads=1024):
    import math
    from openassembler.modelfront.kernels import _float

    count = math.prod(base.inputs[0])
    warps = threads // 32
    name = (
        "oa_warp_mse_"
        + hashlib.sha256((base.ptx + str(threads)).encode()).hexdigest()[:20]
    )
    ptx = f""".visible .entry {name}(.param .u64 a0,.param .u64 a1,.param .u64 output) {{
.reg .b32 %r<12>; .reg .b64 %d<8>; .reg .f32 %f<8>; .reg .pred %p<4>;
.shared .align 4 .b8 partials[{warps*4}];
ld.param.u64 %d0,[a0]; ld.param.u64 %d1,[a1]; ld.param.u64 %d2,[output];
mov.u32 %r0,%tid.x; mov.u32 %r1,%r0; mov.f32 %f0,0f00000000;
loop: setp.ge.u32 %p0,%r1,{count}; @%p0 bra loaded;
mul.wide.u32 %d3,%r1,4; add.u64 %d4,%d0,%d3; add.u64 %d5,%d1,%d3;
ld.global.f32 %f1,[%d4]; ld.global.f32 %f2,[%d5];
sub.rn.f32 %f3,%f1,%f2; mul.rn.f32 %f3,%f3,%f3; add.rn.f32 %f0,%f0,%f3;
add.u32 %r1,%r1,{threads}; bra loop;
loaded:
"""
    for offset in [16, 8, 4, 2, 1]:
        ptx += f"shfl.sync.bfly.b32 %f1,%f0,{offset},31,0xffffffff; add.rn.f32 %f0,%f0,%f1;\n"
    ptx += "and.b32 %r2,%r0,31; shr.u32 %r3,%r0,5; mov.u32 %r4,partials; mad.lo.u32 %r5,%r3,4,%r4; setp.eq.u32 %p1,%r2,0; @%p1 st.shared.f32 [%r5],%f0; bar.sync 0;\n"
    ptx += f"setp.ne.u32 %p2,%r3,0; @%p2 bra done; setp.lt.u32 %p1,%r2,{warps}; mad.lo.u32 %r5,%r2,4,%r4; mov.f32 %f0,0f00000000; @%p1 ld.shared.f32 %f0,[%r5];\n"
    for offset in [16, 8, 4, 2, 1]:
        ptx += f"shfl.sync.bfly.b32 %f1,%f0,{offset},31,0xffffffff; add.rn.f32 %f0,%f0,%f1;\n"
    ptx += f"mul.rn.f32 %f0,%f0,{_float(1/count)}; setp.eq.u32 %p1,%r0,0; @%p1 st.global.f32 [%d2],%f0; done: ret; }}\n"
    return replace(base, name=name, ptx=ptx, grid=(1, 1, 1), threads=threads)
