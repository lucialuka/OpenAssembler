"""Native SM80 warp-per-row softmax and population LayerNorm source emission.

Each of four CTA warps cooperates on one contiguous FP32 row. Every lane
participates in the full-warp reductions; column tails use neutral values
and row tails branch uniformly. The scalar backend lowers each shuffle sync
with an actual WARPSYNC and explicit dependency scoreboards.

LayerNorm has no affine parameters. It shifts values by the first element
before a two-pass population mean/variance reduction, preserving small
variance around large row offsets. Inputs must be bounded finite FP32: the
differences, reduction sums and squared differences must remain finite.
Softmax uses max subtraction, approximate base-2 exponentials and reciprocal.
Numerical agreement is tolerance based, not an IEEE bit-exact library promise.
Launch exactly ``plan.launch`` with contiguous FP32 X/Y buffers aligned to
four bytes; partial overlap is not supported. This module only emits PTX.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct


def _fp(x):
    return "0f" + struct.pack(">f", x).hex()


@dataclass(frozen=True, slots=True)
class ReductionPlan:
    rows: int
    width: int
    op: str
    epsilon: float = 1e-5
    arch: str = "sm_80"
    warps_per_cta: int = 4

    def __post_init__(self):
        if type(self.rows) is not int or self.rows <= 0 or self.rows >= 2**32:
            raise ValueError("rows must be a positive integer within CUDA grid-x")
        if type(self.width) is not int or not 1 <= self.width <= 1024:
            raise ValueError("width must be an integer in [1, 1024]")
        if self.rows * self.width > 2**32:
            raise ValueError("element indices must fit unsigned 32 bits")
        if self.op not in ("softmax", "layernorm"):
            raise ValueError("op must be softmax or layernorm")
        if self.arch != "sm_80":
            raise ValueError("parallel row reductions require sm_80")
        if type(self.warps_per_cta) is not int or self.warps_per_cta not in (1, 2, 4, 8):
            raise ValueError("warps_per_cta must be one of 1, 2, 4 or 8")
        if (isinstance(self.epsilon, bool) or not isinstance(self.epsilon, (int, float))):
            raise ValueError("epsilon must be positive and finite")
        try:
            epsilon_f32 = struct.unpack(">f", struct.pack(">f", self.epsilon))[0]
        except (OverflowError, struct.error):
            raise ValueError("epsilon must remain positive and finite in FP32") from None
        if not math.isfinite(epsilon_f32) or epsilon_f32 < 2**-126:
            raise ValueError("epsilon must remain positive, normal and finite in FP32 for rsqrt.ftz")

    @property
    def launch(self):
        warps = self.warps_per_cta
        return {"grid": [(self.rows+warps-1)//warps, 1, 1], "threads": warps*32, "smem": 0}

    @property
    def metadata(self):
        return {"kernel": self.op, "rows": self.rows, "width": self.width,
                "op": self.op, "epsilon": self.epsilon, "arch": self.arch,
                "input_dtype": "f32", "output_dtype": "f32", "accumulation": "f32",
                "layout": "contiguous_row_major", "warps_per_cta": self.warps_per_cta,
                "lanes_per_row": 32, "finite_inputs_only": True,
                "numerical_contract": "bounded finite FP32 inputs; FP32 differences, sums and squares must remain finite",
                "affine": False, "input_alignment": 4, "output_alignment": 4}


def generate(plan):
    if not isinstance(plan, ReductionPlan):
        raise TypeError("generate requires a ReductionPlan")
    chunks = (plan.width+31)//32
    lines = [".version 8.7\n.target sm_80\n.address_size 64",
             f".visible .entry {plan.op}(.param .u64 X,.param .u64 Y) {{",
             ".reg .b32 %r<16>; .reg .b64 %d<8>; .reg .f32 %f<160>; .reg .pred %p<4>;",
             "ld.param.u64 %d0,[X]; ld.param.u64 %d1,[Y];",
             "mov.u32 %r0,%tid.x; mov.u32 %r1,%ctaid.x;",
             "and.b32 %r2,%r0,31; shr.u32 %r3,%r0,5;",
             f"mad.lo.u32 %r4,%r1,{plan.warps_per_cta},%r3;",
             f"setp.ge.u32 %p0,%r4,{plan.rows}; @%p0 bra done;",
             f"mad.lo.u32 %r5,%r4,{plan.width},%r2;",
             "mul.wide.u32 %d2,%r5,4; add.u64 %d3,%d0,%d2; add.u64 %d4,%d1,%d2;"]
    if plan.op == "layernorm":
        lines.extend([f"mul.lo.u32 %r6,%r4,{plan.width};",
                      "mul.wide.u32 %d5,%r6,4; add.u64 %d6,%d0,%d5;",
                      "ld.global.f32 %f119,[%d6];"])
    tail = plan.width % 32
    if tail:
        lines += [f"setp.lt.u32 %p1,%r2,{tail};"]
    for j in range(chunks):
        if tail and j == chunks-1:
            neutral = "0fff800000" if plan.op == "softmax" else "0f00000000"
            lines += [f"mov.f32 %f{j},{neutral};", f"@%p1 ld.global.f32 %f{j},[%d3+{128*j}];"]
        else:
            lines += [f"ld.global.f32 %f{j},[%d3+{128*j}];"]

    def reduce(kind, src, dest):
        lines.append(f"mov.f32 %f{dest},%f{src[0]};")
        for reg in src[1:]:
            lines.append(f"{kind}.f32 %f{dest},%f{dest},%f{reg};")
        for delta in (16,8,4,2,1):
            lines.extend([f"shfl.sync.bfly.b32 %f159,%f{dest},{delta},31,0xffffffff;",
                          f"{kind}.f32 %f{dest},%f{dest},%f159;"])

    if plan.op == "softmax":
        reduce("max", list(range(chunks)), 120)
        for j in range(chunks):
            lines.extend([f"sub.f32 %f{j},%f{j},%f120;",
                          f"mul.f32 %f{j},%f{j},{_fp(math.log2(math.e))};",
                          f"ex2.approx.ftz.f32 %f{j},%f{j};"])
        reduce("add", list(range(chunks)), 121)
        lines.append("rcp.approx.ftz.f32 %f122,%f121;")
        for j in range(chunks):
            lines.append(f"mul.f32 %f{j},%f{j},%f122;")
    else:
        # Shift before summing: small variance around a large row mean must
        # not amplify FP32 sum cancellation through the normalization scale.
        for j in range(chunks):
            lines.append(f"sub.f32 %f{j},%f{j},%f119;")
            if tail and j == chunks-1:
                lines.append(f"@!%p1 mov.f32 %f{j},0f00000000;")
        reduce("add", list(range(chunks)), 120)
        lines.append(f"mul.f32 %f120,%f120,{_fp(1/plan.width)};")
        for j in range(chunks):
            lines.append(f"sub.f32 %f{j},%f{j},%f120;")
            if tail and j == chunks-1:
                # Inactive columns contribute exactly zero to population variance.
                lines.append(f"@!%p1 mov.f32 %f{j},0f00000000;")
            lines.append(f"mul.f32 %f{32+j},%f{j},%f{j};")
        reduce("add", list(range(32,32+chunks)), 121)
        lines.extend([f"mul.f32 %f121,%f121,{_fp(1/plan.width)};",
                      f"add.f32 %f121,%f121,{_fp(plan.epsilon)};",
                      "rsqrt.approx.ftz.f32 %f122,%f121;"])
        for j in range(chunks):
            lines.append(f"mul.f32 %f{j},%f{j},%f122;")
    for j in range(chunks):
        guard = "@%p1 " if tail and j == chunks-1 else ""
        lines.append(f"{guard}st.global.f32 [%d4+{128*j}],%f{j};")
    lines.extend(["done:", "ret;", "}"])
    return "\n".join(lines)+"\n"
