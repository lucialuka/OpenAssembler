"""Full-tile SM80 RCR GEMM with an explicitly synchronized tensor pipeline.

The default schedule uses a 128x128x64 CTA tile, four 64x64 warp tiles,
FP16/BF16 inputs, FP32 accumulation/output, and two shared-memory stages.
It combines paired B-fragment LDSM x4 loads, shared-copy displacements,
constant-stage loop pairs and an aligned vector output epilogue. A/B are
contiguous MxK/NxK row-major arrays; C is contiguous MxN and equals A @ B.T.

This module only generates PTX. Native tensor instruction selection, direct
64-bit pointer updates and scheduling belong to the standard tensor backend.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GemmPlan:
    """Validated full-tile GEMM contract; boundary tiles are not supported.

    ``stagepair`` is the measured default. ``combined`` retains a dynamic
    two-stage loop with paired B loads and vector output for comparison.
    Launch exactly ``launch``'s grid/threads with its dynamic shared memory.
    A/B must be 16-byte aligned and C must be eight-byte aligned.
    """

    m: int
    n: int
    k: int
    dtype: str = "f16"
    schedule: str = "stagepair"
    output_dtype: str = "f32"
    layout: str = "rcr"
    arch: str = "sm_80"
    cta_order: str = "m_fast"
    epilogue: str = "none"
    pointer_mode: str = "legacy"
    pointer_advance: str = "late"
    fragment_schedule: str = "source"

    def __post_init__(self) -> None:
        for name in ("m", "n", "k"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0 or value % 128:
                raise ValueError(f"{name} must be a positive integer multiple of 128")
        if self.dtype not in ("f16", "bf16"):
            raise ValueError("GEMM input dtype must be f16 or bf16")
        if self.schedule not in ("stagepair", "combined"):
            raise ValueError("GEMM schedule must be stagepair or combined")
        if self.fragment_schedule not in ("source", "load_next"):
            raise ValueError("fragment schedule must be source or load_next")
        if self.cta_order not in ("m_fast", "morton8", "grouped_n8"):
            raise ValueError("CTA order must be m_fast, morton8 or grouped_n8")
        if self.epilogue not in ("none", "bias_residual", "bias_residual_relu"):
            raise ValueError("epilogue must be none, bias_residual or bias_residual_relu")
        if self.pointer_mode not in ("legacy", "direct"):
            raise ValueError("pointer mode must be legacy or direct")
        if self.pointer_advance not in ("late", "early", "auto"):
            raise ValueError("pointer advance must be late, early or auto")
        if self.pointer_mode == "direct" and (self.schedule != "stagepair"
                or self.cta_order != "m_fast" or self.epilogue != "none"):
            raise ValueError("direct pointer mode is validated for stagepair, m_fast, unfused GEMM")
        if self.pointer_advance != "late" and (self.schedule != "stagepair"
                or self.cta_order != "m_fast" or self.epilogue != "none"):
            raise ValueError("early pointer advance is validated for stagepair, m_fast, unfused GEMM")
        if self.fragment_schedule == "load_next" and (
                self.schedule != "stagepair" or self.cta_order != "m_fast"
                or self.pointer_mode != "legacy" or self.pointer_advance != "late"
                or self.epilogue != "none"
                or (self.m, self.n, self.k, self.dtype) not in (
                    (1024, 4096, 1024, "f16"), (4096, 1024, 1024, "f16"),
                    (1024, 4096, 1024, "bf16"), (4096, 1024, 1024, "bf16"))):
            raise ValueError("load_next is validated only for stagepair, m_fast, late-pointer, unfused 1024x4096x1024 or 4096x1024x1024 FP16/BF16 GEMM")
        if self.cta_order == "grouped_n8" and (self.schedule != "stagepair"
                or self.pointer_mode != "legacy" or self.pointer_advance != "late"
                or self.epilogue != "none"):
            raise ValueError("grouped_n8 requires the validated stagepair, legacy-pointer, unfused GEMM")
        if self.epilogue != "none" and self.schedule != "stagepair":
            raise ValueError("fused training epilogues require the validated stagepair schedule")
        if self.output_dtype != "f32" or self.layout != "rcr" or self.arch != "sm_80":
            raise ValueError("GEMM requires f32 output, rcr layout and sm_80")
        # The proven generator computes element offsets with mad.lo.u32 before
        # widening to byte addresses. Reject shapes that could wrap that index.
        if max(self.m*self.k, self.n*self.k, self.m*self.n) > 1 << 32:
            raise ValueError("GEMM element indices must fit in unsigned 32 bits")
        if self.n//128 > 65535:
            raise ValueError("GEMM N tile count exceeds the CUDA grid-y limit")
        if self.cta_order == "morton8" and (self.m//128 % 8 or self.n//128 % 8):
            raise ValueError("morton8 requires complete 8x8 CTA groups")
        if self.cta_order == "grouped_n8" and self.n//128 % 8:
            raise ValueError("grouped_n8 requires N tile count divisible by 8")

    @property
    def launch(self) -> dict:
        """CUDA launch configuration; ``smem`` is dynamic shared bytes."""
        if self.cta_order == "grouped_n8":
            grid = [self.m//128 * 8, self.n//128 // 8, 1]
        else:
            grid = [self.m//128, self.n//128, 1]
        return {"grid": grid, "threads": 128, "smem": 65536}

    @property
    def metadata(self) -> dict:
        """Serializable precision/layout and fixed-geometry contract."""
        return {"kernel": "gemm", "arch": self.arch, "m": self.m, "n": self.n,
                "k": self.k, "dtype": self.dtype, "output_dtype": self.output_dtype,
                "accumulation": "f32", "layout": self.layout, "schedule": self.schedule,
                "fragment_schedule": self.fragment_schedule,
                "cta_order": self.cta_order, "epilogue": self.epilogue,
                "pointer_mode": self.pointer_mode, "pointer_advance": self.pointer_advance,
                "tile_mnk": [128, 128, 64], "warp_mn": [64, 64], "stages": 2,
                "shared_swizzle": [7, 7], "input_alignment": 16, "output_alignment": 8}


def _prologue(plan: GemmPlan) -> list[str]:
    entry = ".visible .entry gemm(.param .u64 A,.param .u64 B,.param .u64 C) {"
    regs = ".reg .b32 %r<1000>; .reg .b64 %d<100>; .reg .f32 %c<128>; .reg .f32 %fBias<32>;"
    if plan.epilogue != "none":
        entry = entry.replace(") {", ",.param .u64 Bias,.param .u64 Residual) {")
        regs += " .reg .f32 %fBatch<8>;"
    lines = [".version 8.7\n.target sm_80\n.address_size 64\n.extern .shared .align 16 .b8 smem[];",
             entry, regs + " .reg .pred %p; .reg .pred %pstage;",
             "", "ld.param.u64 %d0,[A]; ld.param.u64 %d1,[B]; ld.param.u64 %d2,[C];",
             "mov.u32 %r0,%tid.x; mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ctaid.y;"]
    if plan.cta_order == "morton8":
        tm = plan.m // 128
        lines += [f"mad.lo.u32 %r700,%r2,{tm},%r1;", "shr.u32 %r701,%r700,6;",
                  "and.b32 %r702,%r700,63;"]
        if tm == 8:
            lines += ["mov.u32 %r703,0;", "mov.u32 %r704,%r701;"]
        else:
            lines += [f"rem.u32 %r703,%r701,{tm // 8};", f"div.u32 %r704,%r701,{tm // 8};"]
        lines += ["mov.u32 %r705,0;", "mov.u32 %r706,0;"]
        for bit in range(3):
            for target, shift in ((705, 2 * bit), (706, 2 * bit + 1)):
                lines += [f"shr.u32 %r707,%r702,{shift};", "and.b32 %r707,%r707,1;",
                          f"mad.lo.u32 %r{target},%r707,{1 << bit},%r{target};"]
        lines += ["mad.lo.u32 %r1,%r703,8,%r705;", "mad.lo.u32 %r2,%r704,8,%r706;"]
    elif plan.cta_order == "grouped_n8":
        # CUTLASS-style grouped-N traversal. The launch grid is
        # (m_tiles*8, n_tiles/8); keep the mapping in the generated PTX so the
        # source contract, cubin and runner all agree on logical coordinates.
        lines += ["shr.u32 %r700,%r1,3;", "and.b32 %r701,%r1,7;",
                  "mul.lo.u32 %r702,%r2,8;", "add.u32 %r2,%r702,%r701;",
                  "mov.u32 %r1,%r700;"]
    lines += ["mul.lo.u32 %r1,%r1,128; mul.lo.u32 %r2,%r2,128;",
             "and.b32 %r3,%r0,31; shr.u32 %r4,%r0,5;",
             "shr.u32 %r5,%r4,1; mul.lo.u32 %r5,%r5,64;",
             "and.b32 %r6,%r4,1; mul.lo.u32 %r6,%r6,64;",
             "mov.u32 %r8,0; mov.u32 %r9,0;"]
    lines += [f"mov.f32 %c{i},0f00000000;" for i in range(128)]
    for which, block, shared_base in ((0, 1, 0), (1, 2, 16384)):
        for chunk in range(8):
            cid = which*8+chunk
            sr, gr = 100+cid, 10+cid
            lines += [f"add.u32 %r20,%r0,{chunk*128};",
                      "shr.u32 %r21,%r20,3;",
                      "and.b32 %r22,%r20,7; mul.lo.u32 %r22,%r22,8;",
                      f"add.u32 %r23,%r21,%r{block}; mad.lo.u32 %r24,%r23,{plan.k},%r22;",
                      f"mul.wide.u32 %d3,%r24,2; add.u64 %d{gr},%d{which},%d3;",
                      "and.b32 %r25,%r21,7; mul.lo.u32 %r25,%r25,8; xor.b32 %r26,%r22,%r25;",
                      f"mad.lo.u32 %r{sr},%r21,64,%r26; mul.lo.u32 %r{sr},%r{sr},2;",
                      f"add.u32 %r{sr},%r{sr},{shared_base};"]
    for ik in range(4):
        a, b = 200+ik*4, 300+ik*8
        lines += ["and.b32 %r30,%r3,15; add.u32 %r30,%r30,%r5;",
                  "add.u32 %r30,%r30,0; shr.u32 %r31,%r3,4; mul.lo.u32 %r31,%r31,8;",
                  f"add.u32 %r31,%r31,{ik*16}; and.b32 %r32,%r30,7;",
                  "mul.lo.u32 %r32,%r32,8; xor.b32 %r31,%r31,%r32;",
                  f"mad.lo.u32 %r{a},%r30,64,%r31; mul.lo.u32 %r{a},%r{a},2;",
                  "and.b32 %r30,%r3,7; add.u32 %r30,%r30,%r6;",
                  "add.u32 %r30,%r30,0; shr.u32 %r31,%r3,3; and.b32 %r31,%r31,1;",
                  "mul.lo.u32 %r31,%r31,8;",
                  f"add.u32 %r31,%r31,{ik*16}; and.b32 %r32,%r30,7;",
                  "mul.lo.u32 %r32,%r32,8; xor.b32 %r31,%r31,%r32;",
                  f"mad.lo.u32 %r{b},%r30,64,%r31; mul.lo.u32 %r{b},%r{b},2;",
                  f"add.u32 %r{b},%r{b},16384;"]
    return lines


def _copies(stage: str, *, folded: bool) -> list[str]:
    lines = []
    for which in range(2):
        for chunk in range(8):
            cid = which*8+chunk
            if folded:
                target = 80+which
                if chunk == 0:
                    lines += [f"add.u32 %r{target},%r{100+which*8},{stage};"]
                offset = f"+{chunk*2048}" if chunk else ""
            else:
                target, offset = 40, ""
                lines += [f"add.u32 %r40,%r{100+cid},{stage};"]
            lines += [f"cp.async.cg.shared.global [%r{target}{offset}],[%d{10+cid}],16;"]
    return lines+["cp.async.commit_group;"]


def _advance() -> list[str]:
    return [f"add.u64 %d{10+cid},%d{10+cid},128;" for cid in range(16)]


def _compute(dtype: str, stage: str) -> list[str]:
    # Upper sixteen lanes supply the adjacent N fragment's row addresses for
    # each paired B x4 load. Each aligned half feeds one HMMA B operand.
    lines = ["shr.u32 %r47,%r3,4;", "mul.lo.u32 %r47,%r47,1024;"]
    for ik in range(4):
        lines += [f"add.u32 %r41,%r{200+ik*4},{stage};",
                  f"add.u32 %r42,%r{300+ik*8},{stage};",
                  "add.u32 %r42,%r42,%r47;"]
        for mi in range(4):
            aa = ",".join(f"%r{400+mi*4+j}" for j in range(4))
            lines += [f"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{aa}}},[%r41+{mi*2048}];"]
        for ni in range(0, 8, 2):
            bb = ",".join(f"%r{500+ni*2+j}" for j in range(4))
            lines += [f"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{bb}}},[%r42+{ni*1024}];"]
        for ni in range(8):
            bb = ",".join(f"%r{500+ni*2+j}" for j in range(2))
            for mi in range(4):
                aa = ",".join(f"%r{400+mi*4+j}" for j in range(4))
                cc = ",".join(f"%c{(mi*8+ni)*4+j}" for j in range(4))
                lines += [f"mma.sync.aligned.m16n8k16.row.col.f32.{dtype}.{dtype}.f32 "
                          f"{{{cc}}},{{{aa}}},{{{bb}}},{{{cc}}};"]
    return lines


def _compute_load_next(dtype: str, stage: str) -> list[str]:
    """Interleave next-K fragment loads between independent N accumulator groups."""
    lines = ["shr.u32 %r47,%r3,4;", "mul.lo.u32 %r47,%r47,1024;"]

    def load(k: int) -> None:
        a_base = 400 + (k & 1) * 64
        b_base = 500 + (k & 1) * 32
        lines.extend([f"add.u32 %r41,%r{200 + k * 4},{stage};",
                      f"add.u32 %r42,%r{300 + k * 8},{stage};",
                      "add.u32 %r42,%r42,%r47;"])
        for mi in range(4):
            regs = ",".join(f"%r{a_base + mi * 4 + j}" for j in range(4))
            lines.append(f"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{regs}}},[%r41+{mi * 2048}];")
        for ni in range(0, 8, 2):
            regs = ",".join(f"%r{b_base + ni * 2 + j}" for j in range(4))
            lines.append(f"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{regs}}},[%r42+{ni * 1024}];")

    def hmma(k: int, ni: int) -> None:
        a_base = 400 + (k & 1) * 64
        b_base = 500 + (k & 1) * 32
        b_regs = ",".join(f"%r{b_base + ni * 2 + j}" for j in range(2))
        for mi in range(4):
            a_regs = ",".join(f"%r{a_base + mi * 4 + j}" for j in range(4))
            c_regs = ",".join(f"%c{(mi * 8 + ni) * 4 + j}" for j in range(4))
            lines.append(f"mma.sync.aligned.m16n8k16.row.col.f32.{dtype}.{dtype}.f32 "
                         f"{{{c_regs}}},{{{a_regs}}},{{{b_regs}}},{{{c_regs}}};")

    load(0)
    for k in range(4):
        for ni in range(4):
            hmma(k, ni)
        if k + 1 < 4:
            load(k + 1)
        for ni in range(4, 8):
            hmma(k, ni)
    return lines


def _mainloop_compute(plan: GemmPlan, stage: str) -> list[str]:
    if plan.fragment_schedule == "load_next":
        return _compute_load_next(plan.dtype, stage)
    return _compute(plan.dtype, stage)


def _body(dtype: str, current: str, following: str, *, folded: bool,
          pointer_advance: str = "late", fragment_schedule: str = "source") -> list[str]:
    prefix = ["cp.async.wait_group 0; bar.sync 0;"] + _copies(following, folded=folded)
    compute = (_compute_load_next if fragment_schedule == "load_next" else _compute)
    if pointer_advance == "early":
        return prefix + _advance() + compute(dtype, current) + ["bar.sync 0;"]
    return prefix + compute(dtype, current) + ["bar.sync 0;"] + _advance()


def _effective_pointer_advance(plan: GemmPlan) -> str:
    # Auto mode is intentionally a narrow dispatch rule. It is admitted only
    # for the wide shape with two independent FP16/BF16 holdout confirmations.
    if plan.pointer_advance == "auto":
        return "early" if (plan.m, plan.n, plan.k) == (1024, 4096, 1024) else "late"
    return plan.pointer_advance


def _epilogue(n: int) -> list[str]:
    lines = ["shr.u32 %r50,%r3,2; add.u32 %r50,%r50,%r5; add.u32 %r50,%r50,%r1;",
             "and.b32 %r51,%r3,3; mul.lo.u32 %r51,%r51,2; add.u32 %r51,%r51,%r6; add.u32 %r51,%r51,%r2;"]
    # The lane owns adjacent columns; even N and 128-column tile origins make
    # each output pair naturally eight-byte aligned. Form each row base once.
    for mi in range(4):
        for j, dr in ((0, 0), (2, 8)):
            lines += [f"add.u32 %r52,%r50,{mi*16+dr}; mad.lo.u32 %r53,%r52,{n},%r51;",
                      "mul.wide.u32 %d3,%r53,4; add.u64 %d4,%d2,%d3;"]
            for ni in range(8):
                c = (mi*8+ni)*4+j
                offset = f"+{ni*32}" if ni else ""
                lines += [f"st.global.v2.f32 [%d4{offset}],{{%c{c},%c{c+1}}};"]
    return lines


def generate(plan: GemmPlan) -> str:
    """Generate deterministic PTX for a validated plan, without compiler effects."""
    if not isinstance(plan, GemmPlan):
        raise TypeError("generate expects a GemmPlan")
    lines = _prologue(plan)
    pointer_advance = _effective_pointer_advance(plan)
    if plan.schedule == "stagepair":
        lines += _copies("0", folded=True)+_advance()
        first = _body(plan.dtype, "0", "32768", folded=True, pointer_advance=pointer_advance,
                      fragment_schedule=plan.fragment_schedule)
        second = _body(plan.dtype, "32768", "0", folded=True, pointer_advance=pointer_advance,
                       fragment_schedule=plan.fragment_schedule)
        if plan.k > 128:
            lines += (["LOOP_STAGE_PAIR:"]+first+second+
                      [f"add.u32 %r8,%r8,128; setp.lt.u32 %p,%r8,{plan.k-128}; @%p bra LOOP_STAGE_PAIR;"])
        # Tile T-2 copies T-1 into stage one. The drain has no speculative copy.
        lines += first+["cp.async.wait_group 0; bar.sync 0;"]+_mainloop_compute(plan, "32768")+["bar.sync 0;"]
    else:
        lines += _copies("%r9", folded=False)+_advance()+["LOOP:",
                  "cp.async.wait_group 0; bar.sync 0;", "xor.b32 %r10,%r9,32768;"]
        lines += _copies("%r10", folded=False)+_compute(plan.dtype, "%r9")+["bar.sync 0;"]+_advance()
        lines += ["add.u32 %r8,%r8,64; xor.b32 %r9,%r9,32768;",
                  f"setp.lt.u32 %p,%r8,{plan.k-64}; @%p bra LOOP;",
                  "cp.async.wait_group 0; bar.sync 0;"]+_compute(plan.dtype, "%r9")+["bar.sync 0;"]
    source = "\n".join(lines + _epilogue(plan.n) + ["ret;", "}", ""])
    return _fused_epilogue(plan, source) if plan.epilogue != "none" else source


def _fused_epilogue(plan: GemmPlan, source: str) -> str:
    """Emit the A100-validated batched bias/residual epilogue."""
    marker = "shr.u32 %r50,%r3,2; add.u32 %r50,%r50,%r5; add.u32 %r50,%r50,%r1;"
    if source.count(marker) != 1:
        raise ValueError("unexpected GEMM epilogue boundary")
    prefix, _ = source.split(marker, 1)
    lines = [marker,
             "and.b32 %r51,%r3,3; mul.lo.u32 %r51,%r51,2; add.u32 %r51,%r51,%r6; add.u32 %r51,%r51,%r2;",
             "ld.param.u64 %d30,[Bias]; ld.param.u64 %d31,[Residual];",
             "mul.wide.u32 %d32,%r51,4; add.u64 %d33,%d30,%d32;"]
    for ni in range(8):
        off = f"+{ni * 32}" if ni else ""
        lines.append(f"ld.global.v2.f32 {{%fBias{ni * 2},%fBias{ni * 2 + 1}}},[%d33{off}];")
    for mi in range(4):
        for j, dr in ((0, 0), (2, 8)):
            lines += [f"add.u32 %r52,%r50,{mi * 16 + dr}; mad.lo.u32 %r53,%r52,{plan.n},%r51;",
                      "mul.wide.u32 %d3,%r53,4; add.u64 %d4,%d2,%d3; add.u64 %d35,%d31,%d3;"]
            for start in (0, 4):
                for slot, ni in enumerate(range(start, start + 4)):
                    off = f"+{ni * 32}" if ni else ""
                    lines.append(f"ld.global.v2.f32 {{%fBatch{2 * slot},%fBatch{2 * slot + 1}}},[%d35{off}];")
                for slot, ni in enumerate(range(start, start + 4)):
                    c = (mi * 8 + ni) * 4 + j
                    for dc in (0, 1):
                        lines += [f"add.f32 %c{c + dc},%c{c + dc},%fBias{ni * 2 + dc};",
                                  f"add.f32 %c{c + dc},%c{c + dc},%fBatch{2 * slot + dc};"]
                        if plan.epilogue == "bias_residual_relu":
                            lines.append(f"max.f32 %c{c + dc},%c{c + dc},0f00000000;")
                    off = f"+{ni * 32}" if ni else ""
                    lines.append(f"st.global.v2.f32 [%d4{off}],{{%c{c},%c{c + 1}}};")
    return prefix + "\n".join(lines + ["ret;", "}", ""])
