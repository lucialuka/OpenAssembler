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
import json
from dataclasses import dataclass

from openassembler.emu.machine import Launch as SassLaunch
from openassembler.emu.machine import Machine, Memory
from openassembler.ptx.interp import Launch as InterpLaunch
from openassembler.ptx.interp import run_kernel
from openassembler.ptx.parse import Kernel, Module, parse_module

_SCALAR_FMT = {"b8":"<B", "u8":"<B", "s8":"<b", "b16":"<H", "u16":"<H", "s16":"<h", "u32": "<I", "s32": "<i", "f32": "<f", "u64": "<Q", "s64": "<q", "f64": "<d"}
PARAM_BASE = {"sm_80": 0x160, "sm_86": 0x160, "sm_89": 0x160, "sm_90": 0x210, "sm_100": 0x380}
_TYPE_METADATA = '// openassembler-pyfront-types-v1 '
_TYPED_WIDTH = {'f32': 4, 's32': 4, 'u32': 4, 's64': 8, 'u64': 8}


@dataclass
class Mismatch:
    param: str
    index: int
    ours: int
    expected: int


def _parameter_contract(kernel: Kernel, contract: dict) -> list[dict]:
    """Validate explicit array/scalar roles against the actual PTX parameter ABI."""
    if (not isinstance(contract, dict) or set(contract) != {'schema', 'kernel', 'parameters'}
            or type(contract['schema']) is not int or contract['schema'] != 1
            or contract['kernel'] != kernel.name or not isinstance(contract['parameters'], list)):
        raise ValueError('invalid pyfront parameter contract schema or kernel identity')
    parameters = contract['parameters']
    if (len(parameters) != len(kernel.params)
            or len({name for _, name in kernel.params}) != len(kernel.params)):
        raise ValueError('parameter contract does not match PTX parameter count')
    for parameter, (ptx_type, name) in zip(parameters, kernel.params):
        if (not isinstance(parameter, dict) or set(parameter) != {'name', 'kind', 'dtype'}
                or parameter['name'] != name or parameter['kind'] not in ('array', 'scalar')
                or parameter['dtype'] not in _TYPED_WIDTH):
            raise ValueError(f'invalid parameter contract for PTX parameter {name}')
        required = 'u64' if parameter['kind'] == 'array' else parameter['dtype']
        if ptx_type != required:
            raise ValueError(f'parameter contract for {name} requires PTX .{required}, got .{ptx_type}')
    return parameters


def parameter_contract_from_ptx(source: str, kernel: Kernel) -> dict | None:
    """Read optional pyfront metadata; malformed/duplicate tags never fall back."""
    if any(line.strip().startswith('// openassembler-pyfront-buffer-v1') for line in source.splitlines()):
        raise ValueError('shaped buffer verification requires explicit shape/stride/storage bounds; generic allocation is not verified')
    records = {}
    for line_number, line in enumerate(source.splitlines(), 1):
        line = line.strip()
        if not line.startswith(_TYPE_METADATA.rstrip()):
            continue
        if not line.startswith(_TYPE_METADATA):
            raise ValueError(f'invalid pyfront parameter metadata at line {line_number}')
        try:
            record = json.loads(line[len(_TYPE_METADATA):])
        except (ValueError, TypeError) as exc:
            raise ValueError(f'invalid pyfront parameter metadata at line {line_number}: {exc}') from None
        if not isinstance(record, dict) or not isinstance(record.get('kernel'), str):
            raise ValueError(f'invalid pyfront parameter metadata at line {line_number}')
        if record['kernel'] in records:
            raise ValueError(f'duplicate pyfront parameter metadata for {record["kernel"]}')
        records[record['kernel']] = record
    if records:
        module = parse_module(source)
        for name, record in records.items():
            if name not in module.kernels:
                raise ValueError(f'parameter contract names unknown PTX kernel {name}')
            _parameter_contract(module.kernels[name], record)
    contract = records.get(kernel.name)
    if contract is not None:
        _parameter_contract(kernel, contract)
    return contract


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
    if ptx_type in ('b8','u8','s8','b16','u16','s16'):
        bits = int(ptx_type[1:])
        return rng.randrange(-(1 << (bits-1)), 1 << (bits-1)) if ptx_type.startswith('s') else rng.randrange(1 << bits)
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
                  block: Dim | None = None, parameter_contract: dict | None = None) -> list[Mismatch]:
    """Build matching inputs, run both engines, return every differing
    (param, index) pair -- empty means bit-exact. `sass_code`/`sass_labels`
    are exactly what openassembler.backend.ra.compile_kernel returns.
    `grid`/`block`: see the module docstring (defaults: 1 CTA of n_elems)."""
    if parameter_contract is not None:
        return _run_typed_and_diff(mod, kname, sass_code, sass_labels, n_elems, arch, seed,
                                   grid, block, parameter_contract)
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
        if ptx_type == "u64" or k.parameter_attributes.get(name,{}).get('pointer'):
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
            if ptx_type in ('b8','u8','s8','b16','u16','s16') and name.lower() in n_like:
                bits = int(ptx_type[1:])
                maximum = (1 << (bits-int(ptx_type.startswith('s'))))-1
                if not 0 <= v <= maximum:raise ValueError('verification dimension does not fit narrow parameter '+name)
            interp_params[name] = (struct.unpack("<I", struct.pack("<f", v))[0]
                                    if ptx_type == "f32" else int(v) & 0xFFFFFFFF)
            _align(sass_param_bytes, struct.calcsize(_SCALAR_FMT.get(ptx_type, "<i")))
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


def _typed_bits(rng, dtype, index):
    """Deterministic boundary bits followed by seeded data, preserving high words."""
    if dtype == 'f32':
        return struct.unpack('<I', struct.pack('<f', rng.uniform(-8, 8)))[0]
    width = _TYPED_WIDTH[dtype] * 8
    edges = (0, 1, (1 << (width - 1)) - 1, 1 << (width - 1), (1 << width) - 1,
             0xffffffff, 0x100000000 & ((1 << width) - 1), 0x13579bdf2468ace0 & ((1 << width) - 1))
    return edges[index % len(edges)] if index < len(edges) else rng.getrandbits(width)


def _run_typed_and_diff(mod, kname, sass_code, sass_labels, n_elems, arch, seed,
                        grid, block, contract):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
    from bench_build import listing
    from openassembler.emu.program import load_listing

    kernel = mod.kernels[kname]
    parameters = _parameter_contract(kernel, contract)
    if type(n_elems) is not int or n_elems < 1:
        raise ValueError('typed verification requires a positive element count')
    gdim, bdim = launch_geometry(n_elems, grid, block)
    rng = random.Random(seed)
    sass_mem, ptx_mem = Memory(), Memory()
    interp_params, packed, arrays = {}, [], {}
    address = 0x7f00_0000_0000
    guard = bytes([0xa5]) * 64
    dimension_names = {'n', 'count', 'size', 'len', 'num', 'length', 'rows', 'cols', 'width', 'height', 'batch', 'stride'}
    for parameter in parameters:
        name, dtype = parameter['name'], parameter['dtype']
        width = _TYPED_WIDTH[dtype]
        if parameter['kind'] == 'array':
            address = (address + 15) // 16 * 16
            values = [_typed_bits(rng, dtype, i + seed) for i in range(n_elems)]
            payload = b''.join(value.to_bytes(width, 'little') for value in values)
            arrays[name] = (address, width, len(payload))
            for memory in (sass_mem, ptx_mem):
                memory.write_bytes(address - len(guard), guard + payload + guard)
            bits, width = address, 8
            address += max(4096, len(payload) + 256)
        else:
            bits = n_elems if dtype != 'f32' and name.lower() in dimension_names else _typed_bits(rng, dtype, seed)
        interp_params[name] = bits
        _align(packed, width)
        packed.append(bits.to_bytes(width, 'little'))
    sass_params = b''.join(packed)
    program = load_listing(listing(sass_code, sass_labels))
    poisoned_mem = Memory(dict(sass_mem.data))
    ctaids = [(x, y, z) for z in range(gdim[2]) for y in range(gdim[1]) for x in range(gdim[0])]
    for poison, memory in ((None, sass_mem), (0xdeadbeef, poisoned_mem)):
        for ctaid in ctaids:
            Machine(program, SassLaunch(grid=gdim, block=bdim, ctaid=ctaid, params=sass_params,
                    param_base=PARAM_BASE.get(arch, 0x160), poison=poison), memory).run()
    for ctaid in ctaids:
        run_kernel(mod, kname, InterpLaunch(grid=gdim, block=bdim, ctaid=ctaid, params=interp_params), ptx_mem)
    mismatches = []
    for name, (address, width, size) in arrays.items():
        for tag, memory in (('', sass_mem), (':poisoned', poisoned_mem)):
            for index in range(n_elems):
                actual, expected = memory.read(address + width * index, width), ptx_mem.read(address + width * index, width)
                if actual != expected:
                    mismatches.append(Mismatch(name + tag, index, actual, expected))
        # Check each engine's guards against their initial bytes, including when
        # both engines make the same out-of-bounds write. This is a bounded probe,
        # not general proof that arbitrary caller buffer extents are safe.
        for tag, memory in ((':sass-guard', sass_mem), (':poisoned-guard', poisoned_mem), (':ptx-guard', ptx_mem)):
            for offset in (*range(-len(guard), 0), *range(size, size + len(guard))):
                value = memory.read(address + offset, 1)
                if value != 0xa5:
                    mismatches.append(Mismatch(name + tag, offset, value, 0xa5))
    return mismatches
