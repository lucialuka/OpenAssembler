"""Source-level address reuse with the original tensor and synchronization order.

For a 16-byte copy, thread t and chunk c select vector t+128*c.
Because t is in [0,128), chunk c adds exactly 16*c rows and leaves the
within-row column and low three swizzle row bits unchanged. Form one global
base per operand and derive the other chunk bases with checked 64-bit adds.
This is a bounded alternative source emitter, not a generic algebraic pass.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib

from exp.shape_coverage import small_tile_gemm as small


@dataclass(frozen=True)
class Candidate:
    label: str
    address_init: str = "original"
    pointer_mode: str = "legacy"
    early_advance: bool = False


CANDIDATES = (
    Candidate("legacy"),
    Candidate("direct", pointer_mode="direct"),
    Candidate("early", early_advance=True),
    Candidate("direct_early", pointer_mode="direct", early_advance=True),
    Candidate("init_reuse", address_init="reuse"),
    Candidate("init_reuse_direct", address_init="reuse", pointer_mode="direct"),
    Candidate("init_reuse_direct_early", address_init="reuse", pointer_mode="direct", early_advance=True),
)


def _old_init(plan: small.SmallTilePlan, which: int) -> str:
    lines = []
    for chunk in range(4):
        cid = which * 4 + chunk
        lines += [
            f"add.u32 %r20,%r0,{chunk * 128};",
            "shr.u32 %r21,%r20,3;",
            "and.b32 %r22,%r20,7; mul.lo.u32 %r22,%r22,8;",
            f"add.u32 %r23,%r21,%r{1 + which}; mad.lo.u32 %r24,%r23,{plan.k},%r22;",
            f"mul.wide.u32 %d3,%r24,2; add.u64 %d{10 + cid},%d{which},%d3;",
            "and.b32 %r25,%r21,7; mul.lo.u32 %r25,%r25,8; xor.b32 %r26,%r22,%r25;",
            f"mad.lo.u32 %r{100 + cid},%r21,64,%r26; mul.lo.u32 %r{100 + cid},%r{100 + cid},2;",
            f"add.u32 %r{100 + cid},%r{100 + cid},{which * 8192};",
        ]
    return "\n".join(lines)


def source(plan: small.SmallTilePlan, candidate: Candidate) -> tuple[str, dict]:
    if candidate not in CANDIDATES:
        raise ValueError("candidate must belong to the bounded test inventory")
    result = small.generate(plan)
    baseline = result
    if candidate.address_init == "reuse":
        # The original U32 element-index contract bounds every actual pointer.
        # The offset adds are U64 and retain carry, even when the allocation's
        # low address wraps. No assumptions are made about CUDA virtual bases.
        for which in range(2):
            old = _old_init(plan, which)
            if result.count(old) != 1:
                raise ValueError("copy-address initializer no longer matches the audited source")
            first = "\n".join(old.splitlines()[:8])
            replacement = first + "\n" + "\n".join(
                f"add.u64 %d{10 + which * 4 + chunk},%d{10 + which * 4},{chunk * 32 * plan.k};"
                for chunk in range(1, 4))
            result = result.replace(old, replacement)
    if candidate.early_advance:
        for current, following in ((0, small.STAGE_BYTES), (small.STAGE_BYTES, 0)):
            old = "\n".join(small._body(plan.dtype, current, following))
            new = "\n".join(["cp.async.wait_group 0; bar.sync 0;"]
                + small._copies(following) + small._advance()
                + small._compute(plan.dtype, current) + ["bar.sync 0;"])
            expected = (2 if plan.k > 128 else 1) if current == 0 else (1 if plan.k > 128 else 0)
            if result.count(old) != expected:
                raise ValueError("stage-pair source no longer matches the audited envelope")
            result = result.replace(old, new)
    def protected(text):
        # Address arithmetic and its position may change; math, operand-copy,
        # visibility and CTA barrier order must remain byte-for-byte identical.
        return [line for line in text.splitlines()
                if any(op in line for op in ("mma.sync.", "ldmatrix.sync.", "cp.async.", "bar.sync"))]
    if protected(result) != protected(baseline):
        raise AssertionError("a tensor/copy/visibility/barrier operation changed")
    return result, {
        "candidate": candidate.__dict__, "tile_mnk": [64,64,64],
        "manual_launch": plan.launch, "source_sha256": hashlib.sha256(result.encode()).hexdigest(),
        "baseline_source_sha256": hashlib.sha256(baseline.encode()).hexdigest(),
        "tensor_copy_visibility_barrier_stream_equal": True,
        "pointer_add_width": 64, "dependent_accumulator_k_order_equal": True,
        "hardware_validated": False,
    }


def address_certificate() -> dict:
    """Exhaustive local layout identities, including low-address carry samples."""
    count = 0
    for k in (128, 512, 1024, 4096, 32768):
        for tid in range(128):
            row0, col = divmod(tid, 8)
            col *= 8
            for chunk in range(4):
                row, col0 = divmod(tid + 128 * chunk, 8)
                assert col0 * 8 == col and row == row0 + 16 * chunk
                assert (row % 8) == (row0 % 8)
                for cta_row in (0, 64, 1024):
                    old = 2 * ((row + cta_row) * k + col)
                    new = 2 * ((row0 + cta_row) * k + col) + chunk * 32 * k
                    assert old == new
                    for base in (0, 0xfffffff0, 0x1fffffff0, 0x7fff12345670):
                        assert ((base + old) & ((1 << 64) - 1)) == ((base + new) & ((1 << 64) - 1))
                    count += 1
                shared = 2 * (row * 64 + (col ^ ((row % 8) * 8)))
                shared0 = 2 * (row0 * 64 + (col ^ ((row0 % 8) * 8)))
                assert shared == shared0 + 2048 * chunk
    return {"copy_address_equivalences": count, "all_threads_and_chunks": True,
            "checked_pointer_carry_samples": True, "shared_swizzle_preserved": True}


def static_summary(plan: small.SmallTilePlan, candidate: Candidate) -> dict:
    from openassembler.backend.tensor import generate, TensorConfig
    from openassembler.ptx.parse import parse_module
    text, cert = source(plan, candidate)
    config = TensorConfig(pointer_mode=candidate.pointer_mode,
                          pointer_advance="early" if candidate.early_advance else "late")
    code, labels, words, exits, regs = generate(parse_module(text).kernels["gemm"], "sm_80", "opt", config=config)
    return {"label": candidate.label, "source": cert, "instructions": len(code),
            "registers": regs, "opcodes": dict(Counter(x.split()[0] for x,c in code)),
            "hmma_control": sorted({c.stall for x,c in code if x.startswith("HMMA")}),
            "native_words_sha256": hashlib.sha256(b"".join(x.to_bytes(16,"little") for x in words)).hexdigest()}
