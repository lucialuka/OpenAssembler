"""Experiment-only 64x64x64 SM80 GEMM with four 32x32 warp tiles.

This is an explicit source generator, not an inferred production GemmPlan.
It preserves the known lane/fragment layout and two-stage synchronization but
halves both CTA output dimensions. Smaller grids gain CTAs and each thread
owns 32 rather than 128 accumulators; redundant operand traffic also increases.
Neither compilation nor the structural certificate proves a hardware speedup.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib


TILE = 64
WARP = 32
M_FRAGMENTS = 2
N_FRAGMENTS = 4
COPY_CHUNKS = 4
OPERAND_BYTES = 8192
STAGE_BYTES = 16384
SHARED_BYTES = 32768


@dataclass(frozen=True, slots=True)
class SmallTilePlan:
    m: int
    n: int
    k: int
    dtype: str = "f16"

    def __post_init__(self) -> None:
        for name, multiple in (("m", TILE), ("n", TILE), ("k", 128)):
            value = getattr(self, name)
            if type(value) is not int or value <= 0 or value % multiple:
                raise ValueError(f"{name} must be a positive integer multiple of {multiple}")
        if self.dtype not in {"f16", "bf16"}:
            raise ValueError("only f16/bf16 inputs with f32 accumulation/output are supported")
        if max(self.m * self.k, self.n * self.k, self.m * self.n) > 1 << 32:
            raise ValueError("element offsets must fit unsigned 32 bits")
        if self.n // TILE > 65535:
            raise ValueError("N tile count exceeds CUDA grid-y limit")

    @property
    def launch(self) -> dict:
        return {"grid": [self.m // TILE, self.n // TILE, 1],
                "threads": 128, "smem": SHARED_BYTES}


def _prologue(plan: SmallTilePlan) -> list[str]:
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
        for chunk in range(COPY_CHUNKS):
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


def generate(plan: SmallTilePlan) -> str:
    if not isinstance(plan, SmallTilePlan):
        raise TypeError("expected SmallTilePlan")
    lines = _prologue(plan) + _copies(0) + _advance()
    first, second = _body(plan.dtype, 0, STAGE_BYTES), _body(plan.dtype, STAGE_BYTES, 0)
    if plan.k > 128:
        lines += (["LOOP_STAGE_PAIR:"] + first + second
                  + [f"add.u32 %r8,%r8,128; setp.lt.u32 %p,%r8,{plan.k - 128}; @%p bra LOOP_STAGE_PAIR;"])
    lines += (first + ["cp.async.wait_group 0; bar.sync 0;"]
              + _compute(plan.dtype, STAGE_BYTES) + ["bar.sync 0;"])
    return "\n".join(lines + _epilogue(plan.n) + ["ret;", "}", ""])


def output_coordinates(thread: int) -> list[tuple[int, int]]:
    """Independent scalar description of the known MMA output lane map."""
    if type(thread) is not int or not 0 <= thread < 128:
        raise ValueError("thread must be in [0,128)")
    lane, warp = thread % 32, thread // 32
    return [(32 * (warp // 2) + 16 * mi + lane // 4 + 8 * (j // 2),
             32 * (warp % 2) + 8 * ni + 2 * (lane % 4) + j % 2)
            for mi in range(2) for ni in range(4) for j in range(4)]


def copy_coordinates(thread: int, chunk: int, operand: int, stage: int = 0) -> dict:
    if (type(thread) is not int or not 0 <= thread < 128 or type(chunk) is not int
            or not 0 <= chunk < 4 or operand not in (0, 1) or stage not in (0, 1)):
        raise ValueError("copy coordinate is outside the bounded layout")
    vector = thread + 128 * chunk
    row, col = divmod(vector, 8)
    col *= 8
    physical = 2 * (row * 64 + (col ^ ((row % 8) * 8)))
    return {"row": row, "column": col,
            "shared_byte": stage * STAGE_BYTES + operand * OPERAND_BYTES + physical,
            "bytes": 16}


def ldsm_addresses(operand: int, warp: int, ik: int, fragment: int, stage: int = 0) -> list[int]:
    if (operand not in (0, 1) or warp not in range(4) or ik not in range(4)
            or stage not in (0, 1) or fragment not in (range(2) if operand == 0 else (0, 2))):
        raise ValueError("LDSM coordinate is outside the bounded layout")
    rows = []
    for lane in range(32):
        if operand == 0:
            row = 32 * (warp // 2) + 16 * fragment + lane % 16
            col = 16 * ik + 8 * (lane // 16)
        else:
            row = 32 * (warp % 2) + 8 * fragment + lane % 8 + 8 * (lane // 16)
            col = 16 * ik + 8 * ((lane // 8) % 2)
        rows.append(stage * STAGE_BYTES + operand * OPERAND_BYTES
                    + 2 * (row * 64 + (col ^ ((row % 8) * 8))))
    return rows


def layout_certificate() -> dict:
    outputs = Counter(x for thread in range(128) for x in output_coordinates(thread))
    assert set(outputs) == {(m, n) for m in range(64) for n in range(64)}
    assert set(outputs.values()) == {1}
    for operand in range(2):
        for stage in range(2):
            base = stage * STAGE_BYTES + operand * OPERAND_BYTES
            copies = Counter(byte for thread in range(128) for chunk in range(4)
                             for byte in range(copy_coordinates(thread, chunk, operand, stage)["shared_byte"],
                                               copy_coordinates(thread, chunk, operand, stage)["shared_byte"] + 16))
            assert set(copies) == set(range(base, base + OPERAND_BYTES))
            assert set(copies.values()) == {1}
            for warp in range(4):
                for ik in range(4):
                    for fragment in (range(2) if operand == 0 else (0, 2)):
                        for address in ldsm_addresses(operand, warp, ik, fragment, stage):
                            assert address % 16 == 0
                            assert base <= address and address + 16 <= base + OPERAND_BYTES
    return {"tile_mnk": [64, 64, 64], "warp_mn": [32, 32], "warps": 4,
            "accumulators_per_thread": 32, "output_elements": 4096,
            "output_ownership_exact_once": True, "copied_bytes_per_operand_stage": 8192,
            "copy_coverage_exact_once": True, "ldsm_aligned_in_bounds": True,
            "stages": 2, "shared_bytes": SHARED_BYTES, "hardware_validated": False,
            "scope": "layout/source certificate; not instruction-timing or hardware proof"}


def source_for_small_tile(plan: SmallTilePlan) -> tuple[str, dict]:
    source = generate(plan)
    return source, {**layout_certificate(), "plan": asdict(plan), "launch": plan.launch,
                    "manual_launch": plan.launch, "manual_launch_experiment": True,
                    "experimental": True, "full_work_preserved": True,
                    "k_order_preserved": True,
                    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "input_alignment": 16, "output_alignment": 8,
                    "operation": "C[M,N] = A[M,K] @ B[N,K].T; contiguous inputs/output",
                    "input_dtype": plan.dtype, "accumulation_dtype": "f32", "output_dtype": "f32",
                    "hmma_control": 8, "epoch_drain": 15, "production_promoted": False}


def source_for_tile64(plan) -> tuple[str, dict]:
    """Adapt a baseline production plan to an explicit experiment contract.

    No production plan or inferred source marker is modified. Nonbaseline
    pointer/traversal/fragment/epilogue choices require their own experiments.
    """
    from openassembler.kernels.gemm import GemmPlan
    if not isinstance(plan, GemmPlan):
        raise TypeError("expected the baseline GemmPlan for the role runner")
    if (plan.schedule != "stagepair" or plan.cta_order != "m_fast"
            or plan.fragment_schedule != "source" or plan.pointer_mode != "legacy"
            or plan.pointer_advance != "late" or plan.epilogue != "none"):
        raise ValueError("tile64 experiment requires baseline stagepair/m-fast/source/legacy/late/unfused")
    return source_for_small_tile(SmallTilePlan(plan.m, plan.n, plan.k, plan.dtype))


def static_summary(plan: SmallTilePlan) -> dict:
    from openassembler.backend.tensor import generate as encode
    from openassembler.ptx.parse import parse_module
    source, certificate = source_for_small_tile(plan)
    code, labels, words, exits, registers = encode(parse_module(source).kernels["gemm"], "sm_80", "safe")
    counts = Counter(text.split()[0].split(".")[0] for text, _ in code)
    return {"certificate": certificate, "static": {"instructions": len(code),
            "registers": registers, "encoded_words": len(words), "opcodes": dict(counts),
            "labels": len(labels), "exits": len(exits)},
            "evidence_scope": "native source-inclusive compile only; execution/timing untested"}


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--dtype", choices=("f16", "bf16"), default="f16")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = static_summary(SmallTilePlan(args.m, args.n, args.k, args.dtype))
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
        print(args.out)
    else:
        print(text)
