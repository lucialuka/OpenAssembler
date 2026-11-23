"""A generic correctness check: compile a kernel with our backend, run it on
the SASS emulator, run the same PTX on the reference interpreter, and diff
every pointer-typed parameter's memory afterward.

Unlike openassembler/emu/probe.py (the original probe harness, fixed to one
specific 4-parameter signature), this works for any kernel the pyfront/
cppfront frontends -- or anyone's hand-written PTX -- produce: it reads the
kernel's own `.param` list and launch geometry instead of assuming one.
Trade-off: it doesn't know what each array actually means (is `b` an input
or scratch space?), so every pointer parameter is treated as length `n`
(matching the launch's total thread count) and filled with random data; a
kernel that writes outside that bound needs a hand-written check instead
(same as the original probes' fixed ABI didn't generalize either).

Geometry: by default one CTA of `n_elems` threads (the original behaviour).
With `grid=g` the same n_elems elements are spread over g CTAs of
ceil(n_elems / g) threads, so a kernel indexing with
%ctaid.x * %ntid.x + %tid.x touches every element and really runs several
blocks; `block=` overrides the block size. Each CTA runs separately on both
engines (one Machine / Interp per %ctaid, own shared memory, one shared
global memory) in the same order, which is exact for kernels whose CTAs don't
communicate through global memory within one launch -- the CUDA model gives
no ordering between CTAs anyway.
"""

from __future__ import annotations

import random
import struct
from dataclasses import dataclass

from openassembler.emu.machine import Launch as SassLaunch
from openassembler.emu.machine import Machine, Memory
from openassembler.ptx.interp import Launch as InterpLaunch
from openassembler.ptx.interp import run_kernel
from openassembler.ptx.parse import Kernel, Module

_SCALAR_FMT = {"u32": "<I", "s32": "<i", "f32": "<f", "u64": "<Q", "s64": "<q", "f64": "<d"}
PARAM_BASE = {"sm_80": 0x160, "sm_86": 0x160, "sm_89": 0x160, "sm_90": 0x210, "sm_100": 0x380}


@dataclass
class Mismatch:
    param: str
    index: int
    ours: int
    expected: int


def _rand_scalar(rng: random.Random, ptx_type: str, n_elems: int):
    """A plausible scalar value for a parameter that isn't a pointer. Integer
    scalars default to a SMALL bounded range (0..n_elems), not a wide random
    range: in virtually any real kernel an integer scalar is a dimension,
    loop bound, or index multiplier (batch size, a matrix's inner
    dimension, ...), and a large random value here either blows the
    emulator's step limit (a loop bounded by it) or reads/writes outside
    the n_elems-sized buffers this checker allocates. Floats are data
    values, not structural, so they get a wider plausible range."""
    if ptx_type in ("f32", "f64"):
        return rng.uniform(-8, 8)
    return rng.randrange(1, max(1, n_elems) + 1)


def _align(chunks: list[bytes], size: int) -> None:
    """Pad the packed kernel-parameter bytes so the next value starts at a
    multiple of its own size -- the layout CUDA (and our backend, see
    Selector.param_off) uses. Without it, `(float a, float* x, ...)` puts the
    pointer at offset 4 instead of 8 and the kernel reads garbage."""
    have = sum(len(c) for c in chunks)
    if have % size:
        chunks.append(b"\0" * (size - have % size))


Dim = int | tuple[int, int, int]


def _dim3(d: Dim) -> tuple[int, int, int]:
    t = (d, 1, 1) if isinstance(d, int) else tuple(d)
    if len(t) != 3 or any(not isinstance(v, int) or v < 1 for v in t):
        raise ValueError(f"launch dimension must be a positive int or 3 positive ints, got {d!r}")
    return t


def launch_geometry(n_elems: int, grid: Dim = 1, block: Dim | None = None
                    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """(grid, block) for run_and_diff: block defaults to ceil(n_elems / grid.x)
    threads along x (n_elems itself for one CTA, the original behaviour)."""
    g = _dim3(grid)
    b = _dim3(block) if block is not None else (-(-n_elems // g[0]), 1, 1)
    if b[0] * b[1] * b[2] > 1024:
        raise ValueError(f"block {b} has more than 1024 threads (CUDA's limit); pass a larger grid")
    return g, b


def run_and_diff(mod: Module, kname: str, sass_code, sass_labels, n_elems: int = 64,
                  arch: str = "sm_80", seed: int = 0, grid: Dim = 1,
                  block: Dim | None = None) -> list[Mismatch]:
    """Build matching inputs, run both engines, return every differing
    (param, index) pair -- empty means bit-exact. `sass_code`/`sass_labels`
    are exactly what openassembler.backend.ra.compile_kernel returns.
    `grid`/`block`: see the module docstring (defaults: 1 CTA of n_elems)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from bench_build import listing  # noqa: E402  (code+labels -> emulator SASS text)

    k: Kernel = mod.kernels[kname]
    gdim, bdim = launch_geometry(n_elems, grid, block)
    rng = random.Random(seed)

    sass_mem, ptx_mem = Memory(), Memory()
    interp_params: dict[str, int] = {}
    sass_param_bytes: list[bytes] = []
    arrays: dict[str, int] = {}

    # Anything literally named n/count/size/len becomes n_elems, so a kernel
    # guarded by `if i < n` (the common pattern) touches every element.
    n_like = {"n", "count", "size", "len", "num", "length"}
    base_addr = 0x7F00_0000_0000
    for ptx_type, name in k.params:
        if ptx_type == "u64":
            addr = base_addr
            base_addr += max(4096, n_elems * 8 + 256)  # generous gap, 8B/elem covers f64 too
            arrays[name] = addr
            # We don't know the element type from the signature alone (PTX
            # pointers are untyped); default to f32 and let ld/st's own width
            # govern what's actually read back -- this matches every example
            # kernel this project ships (fp32 is the compiler's supported
            # float width, see docs/STATUS.md Gaps).
            buf = bytearray()
            for _ in range(n_elems):
                buf += struct.pack("<f", rng.uniform(-8, 8))
            sass_mem.write_bytes(addr, bytes(buf))
            ptx_mem.write_bytes(addr, bytes(buf))
            interp_params[name] = addr
            _align(sass_param_bytes, 8)
            sass_param_bytes.append(struct.pack("<Q", addr))
        else:
            if name.lower() in n_like:
                v = n_elems
            else:
                v = _rand_scalar(rng, ptx_type, n_elems)
            interp_params[name] = (struct.unpack("<I", struct.pack("<f", v))[0]
                                    if ptx_type == "f32" else int(v) & 0xFFFFFFFF)
            _align(sass_param_bytes, 8 if ptx_type.endswith("64") else 4)
            sass_param_bytes.append(struct.pack(_SCALAR_FMT.get(ptx_type, "<i"),
                                                 v if ptx_type in ("f32", "f64") else int(v)))

    sass_params = b"".join(sass_param_bytes)
    sass_text = listing(sass_code, sass_labels)
    from openassembler.emu.program import load_listing

    prog = load_listing(sass_text)
    ctaids = [(x, y, z) for z in range(gdim[2]) for y in range(gdim[1]) for x in range(gdim[0])]
    for c in ctaids:
        sass_launch = SassLaunch(grid=gdim, block=bdim, ctaid=c, params=sass_params,
                                  param_base=PARAM_BASE.get(arch, 0x160))
        Machine(prog, sass_launch, sass_mem).run()
    for c in ctaids:
        interp_launch = InterpLaunch(block=bdim, grid=gdim, ctaid=c, params=interp_params)
        run_kernel(mod, kname, interp_launch, ptx_mem)

    mismatches = []
    for name, addr in arrays.items():
        for i in range(n_elems):
            g = sass_mem.read(addr + 4 * i, 4)
            e = ptx_mem.read(addr + 4 * i, 4)
            if g != e:
                mismatches.append(Mismatch(name, i, g, e))
    return mismatches
