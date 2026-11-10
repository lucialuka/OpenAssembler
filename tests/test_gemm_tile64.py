"""Native64 source parity, exact address coverage and poisoned output checks.

Golden source hashes were captured from the passing November 9 A100 experiment;
these tests deliberately do not import experiment code or historical artifacts.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError
from hashlib import sha256
import random
import struct

import pytest

from openassembler.kernels import GemmPlan, generate
from openassembler.ptx.parse import Addr, ImmI, Var, parse_module


_SOURCE_HASHES = {
    (64, 64, 128, "f16"): "1ff1c4b16f4e30aeadf5870c0663a08225e00441d4b6de36e55c5f3dd6b9db16",
    (64, 64, 128, "bf16"): "03c47f9c8d006f4dcc24973213a3fa78523c72c11332ae23c2c480bdc1d5b492",
    (128, 512, 512, "f16"): "37188e6c06d1b1215774ff563e23aec90e4632e7685de54133d90d8d7ed548b0",
    (128, 512, 512, "bf16"): "9092b69ed57f045639907754a2d79d1cc3e7b3c8692570dec3de5816382a644d",
    (1024, 1024, 4096, "f16"): "4b76f6832b5b58e78a083c4e47bd5aa18ce0826cb307e77e1c61fbd17d23a8f7",
    (1024, 1024, 4096, "bf16"): "02c61f25aee62a03fca6823663b1dd4db818b884f059d6dd05f183d0a974bc00",
}


@pytest.mark.parametrize("m,n,k,dtype", _SOURCE_HASHES)
def test_tile64_source_matches_hardware_validated_experimental_bytes(m, n, k, dtype):
    source = generate(GemmPlan(m, n, k, dtype, tile="64x64"))
    assert sha256(source.encode()).hexdigest() == _SOURCE_HASHES[m, n, k, dtype]
    kernel = parse_module(source).kernels["gemm"]
    assert kernel.params == [("u64", "A"), ("u64", "B"), ("u64", "C")]
    pairs = 2 if k > 128 else 1
    assert source.count("mma.sync.") == 64 * pairs
    assert source.count("ldmatrix.sync.") == 32 * pairs
    assert source.count("cp.async.cg.") == 16 * pairs
    assert source.count("cp.async.commit_group") == 2 * pairs
    assert source.count("cp.async.wait_group 0") == 2 * pairs
    assert source.count("bar.sync 0") == 4 * pairs
    assert source.count("st.global.v2.f32") == 16


@pytest.mark.parametrize("changes", [
    {"tile": "32x32"}, {"m": 0}, {"m": 65}, {"n": 96}, {"k": 64},
    {"m": True}, {"n": 64.0}, {"k": "128"}, {"dtype": "tf32"},
    {"m": 131072, "n": 131072}, {"n": 65536 * 64},
    {"schedule": "combined"}, {"cta_order": "morton8"},
    {"cta_order": "grouped_n8"},
    {"pointer_advance": "early"}, {"pointer_advance": "auto"},
    {"fragment_schedule": "load_next"}, {"epilogue": "bias_residual"},
    {"epilogue": "bias_residual_relu"}, {"output_dtype": "f16"},
    {"layout": "rrr"}, {"arch": "sm_90"},
])
def test_tile64_contract_rejects_unvalidated_shapes_or_policy_composition(changes):
    args = dict(m=64, n=64, k=128, tile="64x64")
    args.update(changes)
    with pytest.raises(ValueError):
        GemmPlan(**args)


def test_tile64_metadata_and_launch_are_explicit_independent_and_immutable():
    plan = GemmPlan(192, 320, 384, "bf16", tile="64x64")
    assert plan.launch == {"grid": [3, 5, 1], "threads": 128, "smem": 32768}
    assert plan.metadata["tile_mnk"] == [64, 64, 64]
    assert plan.metadata["warp_mn"] == [32, 32]
    assert plan.metadata["stages"] == 2
    assert plan.metadata["shared_swizzle"] == [7, 7]
    assert plan.metadata["input_alignment"] == 16
    assert plan.metadata["output_alignment"] == 8
    assert plan.metadata["tile"] == "64x64"
    launch = plan.launch
    launch["grid"][0] = 100
    assert plan.launch["grid"] == [3, 5, 1]
    with pytest.raises(FrozenInstanceError):
        plan.tile = "128x128"


def test_tile64_direct_pointer_contract_is_shape_validated():
    plan = GemmPlan(128, 512, 512, tile="64x64", pointer_mode="direct")
    assert plan.pointer_mode == "direct"
    assert plan.launch == {"grid": [2, 8, 1], "threads": 128, "smem": 32768}


def _trace(plan, tid):
    """Interpret actual scalar instructions and certify tensor/copy boundaries.

    Arithmetic is checked separately by the SASS emulator. Here every async
    copy and LDSM address is evaluated, including loop backedges and last CTA.
    """
    kernel = parse_module(generate(plan)).kernels["gemm"]
    a, b, c = 0, 1 << 48, 2 << 48
    regs = {"%tid.x": tid, "%ctaid.x": plan.m // 64 - 1,
            "%ctaid.y": plan.n // 64 - 1}
    params = {"A": a, "B": b, "C": c}
    pending, copies, reads, stores, updates, consumed = [], [], [], [], [], []
    stages = {}
    visible = False
    current_tile = None
    barriers = 0

    def value(arg):
        if isinstance(arg, ImmI):
            return arg.value
        if isinstance(arg, Var):
            return regs[arg.name]
        if isinstance(arg, Addr):
            return regs[arg.base] + arg.off
        raise AssertionError(arg)

    pc = steps = 0
    while pc < len(kernel.body):
        steps += 1
        assert steps < 500000
        ins = kernel.body[pc]
        pc += 1
        op, args = ins.op, ins.args
        if ins.guard is not None and not (regs[ins.guard[0]] ^ ins.guard[1]):
            continue
        if op == "ld" and "param" in ins.mods:
            regs[args[0].name] = params[args[1].base]
        elif op == "mov" and "f32" not in ins.mods:
            regs[args[0].name] = value(args[1])
        elif op in ("add", "mul", "mad", "and", "xor", "shr"):
            values = [value(arg) for arg in args[1:]]
            if op == "add": result = values[0] + values[1]
            elif op == "mul": result = values[0] * values[1]
            elif op == "mad": result = values[0] * values[1] + values[2]
            elif op == "and": result = values[0] & values[1]
            elif op == "xor": result = values[0] ^ values[1]
            else: result = values[0] >> values[1]
            width = 64 if "u64" in ins.mods or "wide" in ins.mods else 32
            regs[args[0].name] = result & ((1 << width) - 1)
        elif op == "setp":
            assert ins.mods == ("lt", "u32")
            regs[args[0].name] = value(args[1]) < value(args[2])
        elif op == "bra":
            pc = kernel.labels[args[0].name]
        elif op == "cp" and "global" in ins.mods:
            shared, glob = value(args[0]), value(args[1])
            operand = int(glob >= b)
            base, rows = (b, plan.n) if operand else (a, plan.m)
            index = (glob - base) // 2
            row, col = divmod(index, plan.k)
            assert 0 <= row < rows and col % 8 == 0 and col + 8 <= plan.k
            stage = shared // 16384
            assert stage in (0, 1) and shared % 16 == 0
            assert (shared % 16384) // 8192 == operand
            copies.append((operand, stage, col // 64, row, col, shared))
            pending.append((stage, col // 64))
        elif op == "cp" and "commit_group" in ins.mods:
            assert len(pending) == 8 and len(set(pending)) == 1
            stage, tile = pending[0]
            stages[stage] = (tile, False)
            pending.clear()
        elif op == "cp" and "wait_group" in ins.mods:
            assert value(args[0]) == 0
            stages = {stage: (tile, True) for stage, (tile, _) in stages.items()}
            visible = False
            current_tile = None
        elif op == "bar":
            barriers += 1
            visible = True
        elif op == "ldmatrix":
            assert visible
            address = value(args[1])
            stage = address // 16384
            operand = (address % 16384) // 8192
            base = stage * 16384 + operand * 8192
            assert address % 16 == 0 and base <= address <= base + 8192 - 16
            tile, ready = stages[stage]
            assert ready
            if current_tile is None:
                current_tile = tile
                consumed.append(tile)
            assert tile == current_tile
            reads.append(address)
        elif op == "mma":
            assert current_tile is not None
            accumulator = tuple(reg.name for reg in args[0].elems)
            assert accumulator == tuple(reg.name for reg in args[3].elems)
            updates.append((accumulator, current_tile))
        elif op == "st":
            assert ins.mods == ("global", "v2", "f32")
            index = (value(args[0]) - c) // 4
            assert 0 <= index < plan.m * plan.n - 1 and index % 2 == 0
            stores += [index, index + 1]
        elif op == "ret":
            break
        else:
            assert op == "mov" and "f32" in ins.mods, (op, ins.mods)
    assert not pending
    assert consumed == list(range(plan.k // 64))
    assert barriers == 2 * (plan.k // 64)
    assert len(reads) == 16 * (plan.k // 64)
    accumulators = {group for group, _ in updates}
    assert len(accumulators) == 8
    for group in accumulators:
        assert [tile for current, tile in updates if current == group] == [
            tile for tile in range(plan.k // 64) for _ in range(4)]
    assert len(stores) == len(set(stores)) == 32
    return copies, reads, stores


@pytest.mark.parametrize("k", [128, 256, 384, 4096])
def test_actual_pipeline_preserves_every_k_dependency_and_visibility(k):
    plan = GemmPlan(192, 320, k, tile="64x64")
    for tid in (0, 15, 16, 31, 32, 64, 127):
        _trace(plan, tid)


def test_emitted_copy_and_output_addresses_cover_each_byte_once():
    plan = GemmPlan(192, 320, 128, tile="64x64")
    copies, stores = [], []
    for tid in range(128):
        ct, _, st = _trace(plan, tid)
        copies.extend(ct)
        stores.extend(st)
    expected_outputs = {row * plan.n + col for row in range(plan.m - 64, plan.m)
                        for col in range(plan.n - 64, plan.n)}
    assert Counter(stores) == Counter({index: 1 for index in expected_outputs})
    for operand in (0, 1):
        for tile in (0, 1):
            selected = [item for item in copies if item[0] == operand and item[2] == tile]
            assert len(selected) == 512
            rows = plan.m if operand == 0 else plan.n
            logical = Counter((row, col + offset) for _, _, _, row, col, _ in selected
                              for offset in range(8))
            assert logical == Counter({(row, col): 1 for row in range(rows - 64, rows)
                                       for col in range(tile * 64, tile * 64 + 64)})
            stage = tile % 2
            base = stage * 16384 + operand * 8192
            physical = Counter(address + offset for *_, address in selected for offset in range(16))
            assert physical == Counter({byte: 1 for byte in range(base, base + 8192)})


@pytest.mark.parametrize("dtype,k,opt,pointer_mode", [
    ("f16", 128, "safe", "legacy"), ("bf16", 128, "opt", "legacy"),
    ("f16", 384, "opt", "legacy"), ("bf16", 384, "safe", "legacy"),
    ("f16", 384, "safe", "direct"),
])
def test_native_tile64_poisoned_full_output_matches_integer_oracle(dtype, k, opt, pointer_mode):
    from openassembler.backend.tensor import TensorConfig, generate as compile_tensor
    from openassembler.emu.machine import Launch, Machine, Memory
    from openassembler.emu.program import load_listing
    from tests.backend_harness import listing

    plan = GemmPlan(64, 64, k, dtype, tile="64x64", pointer_mode=pointer_mode)
    code, labels, words, exits, registers = compile_tensor(
        parse_module(generate(plan)).kernels["gemm"], "sm_80", opt,
        config=TensorConfig(pointer_mode=pointer_mode))
    rng = random.Random(198)
    av = [rng.randrange(-2, 3) for _ in range(plan.m * k)]
    bv = [rng.randrange(-2, 3) for _ in range(plan.n * k)]
    expected = [sum(av[row * k + i] * bv[col * k + i] for i in range(k))
                for row in range(plan.m) for col in range(plan.n)]

    def pack(values):
        if dtype == "f16":
            return b"".join(struct.pack("<e", value) for value in values)
        return b"".join(struct.pack("<H", struct.unpack("<I", struct.pack("<f", value))[0] >> 16)
                        for value in values)

    a, b, c = 0x100000, 0x200000, 0x300000
    memory = Memory()
    memory.write_bytes(a, pack(av))
    memory.write_bytes(b, pack(bv))
    canary = b"\x76\x54\x32\x10" * 32
    memory.write_bytes(c - len(canary), canary)
    memory.write_bytes(c, b"\x00\x00\xc0\x7f" * (plan.m * plan.n))
    memory.write_bytes(c + 4 * plan.m * plan.n, canary)
    Machine(load_listing(listing(code, labels)),
            Launch(grid=(1, 1, 1), block=(128, 1, 1), params=struct.pack("<QQQ", a, b, c),
                   smem_static=plan.launch["smem"], poison=0xDEADBEEF),
            memory).run(max_steps=6000000)
    got = [struct.unpack("<f", struct.pack("<I", memory.read(c + 4 * i, 4)))[0]
           for i in range(plan.m * plan.n)]
    assert got == expected
    read_bytes = lambda address, size: memory.read(address, size).to_bytes(size, "little")
    assert read_bytes(c - len(canary), len(canary)) == canary
    assert read_bytes(c + 4 * plan.m * plan.n, len(canary)) == canary
    assert read_bytes(a, len(pack(av))) == pack(av)
    assert read_bytes(b, len(pack(bv))) == pack(bv)
    assert registers == (72 if k == 128 else 88)
    assert len(words) >= len(code) and len(words) % 8 == 0 and exits
    assert sum(text.startswith("STG.E.64") for text, _ in code) == 16
    assert not any(text.startswith(("LDL", "STL", "CALL", "JCAL")) for text, _ in code)
    if pointer_mode == "direct":
        legacy, *_ = compile_tensor(parse_module(generate(
            GemmPlan(64, 64, k, dtype, tile="64x64"))).kernels["gemm"], "sm_80", opt)
        assert len(code) < len(legacy)
        assert sum(text.startswith("MOV") for text, _ in code) + 64 == sum(
            text.startswith("MOV") for text, _ in legacy)
