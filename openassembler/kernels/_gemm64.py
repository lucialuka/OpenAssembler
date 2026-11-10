"""Native 64x64x64 SM80 GEMM source.

The PTX below is byte-identical to the bounded A100-validated small-tile
experiment. Admission, numerical/layout constraints and launch metadata belong
to :class:`openassembler.kernels.gemm.GemmPlan`; automatic selection belongs to
the measured GEMM dispatch policy. Smaller tiles can regress large workloads.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openassembler.kernels.gemm import GemmPlan

M_FRAGMENTS = 2
N_FRAGMENTS = 4
COPY_CHUNKS = 4
OPERAND_BYTES = 8192
STAGE_BYTES = 16384


def _prologue(plan: GemmPlan) -> list[str]:
    lines = [
        ".version 8.7\n.target sm_80\n.address_size 64\n.extern .shared .align 16 .b8 smem[];",
        ".visible .entry gemm(.param .u64 A,.param .u64 B,.param .u64 C) {",
        ".reg .b32 %r<600>; .reg .b64 %d<100>; .reg .f32 %c<32>; .reg .pred %p;",
        "ld.param.u64 %d0,[A]; ld.param.u64 %d1,[B]; ld.param.u64 %d2,[C];",
        "mov.u32 %r0,%tid.x; mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ctaid.y;",
        "mul.lo.u32 %r1,%r1,64; mul.lo.u32 %r2,%r2,64;",
        "and.b32 %r3,%r0,31; shr.u32 %r4,%r0,5;",
        "shr.u32 %r5,%r4,1; mul.lo.u32 %r5,%r5,32;",
        "and.b32 %r6,%r4,1; mul.lo.u32 %r6,%r6,32;",
        "mov.u32 %r8,0;",
    ]
    lines += [f"mov.f32 %c{i},0f00000000;" for i in range(32)]
    for which, cta_row_register in ((0, 1), (1, 2)):
        chunks = range(1) if plan.address_init == "reuse" else range(COPY_CHUNKS)
        for chunk in chunks:
            cid = which * COPY_CHUNKS + chunk
            lines += [
                f"add.u32 %r20,%r0,{chunk * 128};",
                "shr.u32 %r21,%r20,3;",
                "and.b32 %r22,%r20,7; mul.lo.u32 %r22,%r22,8;",
                f"add.u32 %r23,%r21,%r{cta_row_register}; mad.lo.u32 %r24,%r23,{plan.k},%r22;",
                f"mul.wide.u32 %d3,%r24,2; add.u64 %d{10 + cid},%d{which},%d3;",
                "and.b32 %r25,%r21,7; mul.lo.u32 %r25,%r25,8; xor.b32 %r26,%r22,%r25;",
                f"mad.lo.u32 %r{100 + cid},%r21,64,%r26; mul.lo.u32 %r{100 + cid},%r{100 + cid},2;",
                f"add.u32 %r{100 + cid},%r{100 + cid},{which * OPERAND_BYTES};",
            ]
        if plan.address_init == "reuse":
            for chunk in range(1, COPY_CHUNKS):
                cid = which * COPY_CHUNKS + chunk
                lines.append(f"add.u64 %d{10 + cid},%d{10 + which * COPY_CHUNKS},{chunk * 32 * plan.k};")
    for ik in range(4):
        a, b = 200 + ik * 4, 300 + ik * 8
        lines += [
            "and.b32 %r30,%r3,15; add.u32 %r30,%r30,%r5;",
            "shr.u32 %r31,%r3,4; mul.lo.u32 %r31,%r31,8;",
            f"add.u32 %r31,%r31,{ik * 16}; and.b32 %r32,%r30,7;",
            "mul.lo.u32 %r32,%r32,8; xor.b32 %r31,%r31,%r32;",
            f"mad.lo.u32 %r{a},%r30,64,%r31; mul.lo.u32 %r{a},%r{a},2;",
            "and.b32 %r30,%r3,7; add.u32 %r30,%r30,%r6;",
            "shr.u32 %r31,%r3,3; and.b32 %r31,%r31,1; mul.lo.u32 %r31,%r31,8;",
            f"add.u32 %r31,%r31,{ik * 16}; and.b32 %r32,%r30,7;",
            "mul.lo.u32 %r32,%r32,8; xor.b32 %r31,%r31,%r32;",
            f"mad.lo.u32 %r{b},%r30,64,%r31; mul.lo.u32 %r{b},%r{b},2;",
            f"add.u32 %r{b},%r{b},{OPERAND_BYTES};",
        ]
    return lines


def _copies(stage: int) -> list[str]:
    lines = []
    for which in range(2):
        target = 80 + which
        lines.append(f"add.u32 %r{target},%r{100 + which * COPY_CHUNKS},{stage};")
        for chunk in range(COPY_CHUNKS):
            offset = f"+{chunk * 2048}" if chunk else ""
            gr = 10 + which * COPY_CHUNKS + chunk
            lines.append(f"cp.async.cg.shared.global [%r{target}{offset}],[%d{gr}],16;")
    return lines + ["cp.async.commit_group;"]


def _advance() -> list[str]:
    return [f"add.u64 %d{10 + cid},%d{10 + cid},128;" for cid in range(8)]


def _compute(dtype: str, stage: int) -> list[str]:
    lines = ["shr.u32 %r47,%r3,4;", "mul.lo.u32 %r47,%r47,1024;"]
    for ik in range(4):
        lines += [f"add.u32 %r41,%r{200 + ik * 4},{stage};",
                  f"add.u32 %r42,%r{300 + ik * 8},{stage};",
                  "add.u32 %r42,%r42,%r47;"]
        for mi in range(M_FRAGMENTS):
            regs = ",".join(f"%r{400 + mi * 4 + j}" for j in range(4))
            lines.append(f"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{regs}}},[%r41+{mi * 2048}];")
        for ni in range(0, N_FRAGMENTS, 2):
            regs = ",".join(f"%r{500 + ni * 2 + j}" for j in range(4))
            lines.append(f"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{regs}}},[%r42+{ni * 1024}];")
        for ni in range(N_FRAGMENTS):
            b = ",".join(f"%r{500 + ni * 2 + j}" for j in range(2))
            for mi in range(M_FRAGMENTS):
                a = ",".join(f"%r{400 + mi * 4 + j}" for j in range(4))
                c = ",".join(f"%c{(mi * N_FRAGMENTS + ni) * 4 + j}" for j in range(4))
                lines.append(f"mma.sync.aligned.m16n8k16.row.col.f32.{dtype}.{dtype}.f32 "
                             f"{{{c}}},{{{a}}},{{{b}}},{{{c}}};")
    return lines


def _body(dtype: str, current: int, following: int) -> list[str]:
    return (["cp.async.wait_group 0; bar.sync 0;"] + _copies(following)
            + _compute(dtype, current) + ["bar.sync 0;"] + _advance())


def _body_early(dtype: str, current: int, following: int) -> list[str]:
    return ( ["cp.async.wait_group 0; bar.sync 0;"] + _copies(following)
            + _advance() + _compute(dtype, current) + ["bar.sync 0;"] )


def _epilogue(n: int) -> list[str]:
    lines = [
        "shr.u32 %r50,%r3,2; add.u32 %r50,%r50,%r5; add.u32 %r50,%r50,%r1;",
        "and.b32 %r51,%r3,3; mul.lo.u32 %r51,%r51,2; add.u32 %r51,%r51,%r6; add.u32 %r51,%r51,%r2;",
    ]
    for mi in range(M_FRAGMENTS):
        for j, row_offset in ((0, 0), (2, 8)):
            lines += [f"add.u32 %r52,%r50,{mi * 16 + row_offset}; mad.lo.u32 %r53,%r52,{n},%r51;",
                      "mul.wide.u32 %d3,%r53,4; add.u64 %d4,%d2,%d3;"]
            for ni in range(N_FRAGMENTS):
                c = (mi * N_FRAGMENTS + ni) * 4 + j
                offset = f"+{ni * 32}" if ni else ""
                lines.append(f"st.global.v2.f32 [%d4{offset}],{{%c{c},%c{c + 1}}};")
    return lines


def generate(plan: GemmPlan) -> str:
    pointer_advance = plan.pointer_advance
    if pointer_advance == "auto":
        pointer_advance = ("early" if (plan.m, plan.n, plan.k) == (1024, 4096, 1024)
                           else "late")
    body = _body_early if pointer_advance == "early" else _body
    lines = _prologue(plan) + _copies(0) + _advance()
    first, second = body(plan.dtype, 0, STAGE_BYTES), body(plan.dtype, STAGE_BYTES, 0)
    if plan.k > 128:
        lines += (["LOOP_STAGE_PAIR:"] + first + second
                  + [f"add.u32 %r8,%r8,128; setp.lt.u32 %p,%r8,{plan.k - 128}; @%p bra LOOP_STAGE_PAIR;"])
    lines += (first + ["cp.async.wait_group 0; bar.sync 0;"]
              + _compute(plan.dtype, STAGE_BYTES) + ["bar.sync 0;"])
    return "\n".join(lines + _epilogue(plan.n) + ["ret;", "}", ""])

