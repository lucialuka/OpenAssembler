"""Row-reduction contracts, poisoned native semantics and dependency regressions."""
from dataclasses import FrozenInstanceError
import math
import struct

import pytest

from openassembler.backend import ra
from openassembler.backend.compile import Out, schedule
from openassembler.build import generate as native_generate
from openassembler.emu.machine import Launch, Machine, Memory
from openassembler.emu.program import load_listing
from openassembler.isa.encode import Encoder
from openassembler.kernels.reductions import ReductionPlan, generate
from openassembler.ptx.parse import Kernel, parse_module
from tests.backend_harness import compile_and_diff, listing


@pytest.mark.parametrize("change", [
    {"rows": 0}, {"rows": -1}, {"rows": True}, {"rows": 1.0}, {"rows": 2**32},
    {"width": 0}, {"width": 1025}, {"width": True}, {"width": 1.5},
    {"op": "rmsnorm"}, {"arch": "sm_90"},
    {"epsilon": 0.}, {"epsilon": -1.}, {"epsilon": float("nan")},
    {"epsilon": float("inf")}, {"epsilon": True}, {"epsilon": 1e-320},
    {"epsilon": 1e-40}, {"epsilon": 1e308}, {"epsilon": 10**1000},
    {"rows": 2**31, "width": 3},
])
def test_plan_rejects_unimplemented_or_unrepresentable_contract(change):
    args = dict(rows=128, width=128, op="softmax")
    args.update(change)
    with pytest.raises(ValueError):
        ReductionPlan(**args)


def test_plan_immutable_and_launch_metadata_do_not_alias():
    plan = ReductionPlan(129, 513, "layernorm")
    assert plan.launch == {"grid": [33, 1, 1], "threads": 128, "smem": 0}
    assert plan.metadata["affine"] is False
    assert plan.metadata["input_dtype"] == plan.metadata["output_dtype"] == "f32"
    assert "bounded finite" in plan.metadata["numerical_contract"]
    launch = plan.launch
    launch["grid"][0] = 1
    assert plan.launch["grid"][0] == 33
    with pytest.raises(FrozenInstanceError):
        plan.rows = 1
    with pytest.raises(TypeError):
        generate({"rows": 1, "width": 1, "op": "softmax"})
    # The u32 compare accepts its largest representable bound, never 2**32.
    largest = ReductionPlan(2**32-1, 1, "softmax")
    assert "setp.ge.u32 %p0,%r4,4294967295;" in generate(largest)


@pytest.mark.parametrize("op", ["softmax", "layernorm"])
def test_native_shuffle_sync_keeps_warpsync_and_read_scoreboards(op):
    plan = ReductionPlan(5, 33, op)
    kernel = parse_module(generate(plan)).kernels[op]
    assert kernel.params == [("u64", "X"), ("u64", "Y")]
    code, _, _, _, _ = native_generate(kernel, "sm_80", "opt")
    communication = [(text, ctl) for text, ctl in code
                     if text.startswith(("WARPSYNC", "SHFL"))]
    assert len(communication) == 20
    assert all(text.startswith("WARPSYNC") for text, _ in communication[::2])
    assert all(text.startswith("SHFL.BFLY") and ctl.wbar != 7 and ctl.rbar != 7
               for text, ctl in communication[1::2])
    assert not any("LDS" in text or "STS" in text for text, _ in code)


@pytest.mark.parametrize("op", ["softmax", "layernorm"])
@pytest.mark.parametrize("width", [1, 31, 33, 128, 513, 1024])
def test_poisoned_native_outputs_and_both_tails_match_fp64(op, width):
    rows = 5 if width < 128 else 3
    plan = ReductionPlan(rows, width, op)
    code, labels, _, _, _ = native_generate(parse_module(generate(plan)).kernels[op], "sm_80", "opt")
    program = load_listing(listing(code, labels))
    # Distinct input/output windows, with poison on both sides of the output.
    xbase, storage = 0x100000000, 0x200000000
    outbase = storage + 256
    guard = 0xC640E400
    for offset in (0.0, 1000.0):
        scale = 2. if offset == 0. else .01
        values = [struct.unpack("<f", struct.pack("<f", offset + scale*math.sin(i*.31)))[0]
                  for i in range(rows*width)]
        memory = Memory()
        memory.write_bytes(xbase, struct.pack(f"<{len(values)}f", *values))
        memory.write_bytes(storage, struct.pack(f"<{len(values)+128}I", *([guard]*64+[0x7fc00000]*len(values)+[guard]*64)))
        for block in range(plan.launch["grid"][0]):
            Machine(program, Launch(block=(128,1,1), grid=tuple(plan.launch["grid"]),
                    ctaid=(block,0,0), params=struct.pack("<QQ", xbase, outbase),
                    poison=0xdeadbeef), memory).run()
        actual = [struct.unpack("<f", struct.pack("<I", memory.read(outbase+4*i,4)))[0]
                  for i in range(len(values))]
        assert all(memory.read(storage+4*i,4) == guard for i in range(64))
        assert all(memory.read(outbase+4*(len(values)+i),4) == guard for i in range(64))
        for row in range(rows):
            xs = values[row*width:(row+1)*width]
            if op == "softmax":
                exps = [math.exp(value-max(xs)) for value in xs]
                expected = [value/sum(exps) for value in exps]
                rtol, atol = 5e-5, 1e-6
            else:
                mean = sum(xs)/width
                inv = 1/math.sqrt(sum((value-mean)**2 for value in xs)/width+1e-5)
                expected = [(value-mean)*inv for value in xs]
                rtol, atol = 1e-4, 2e-5
            for value, reference in zip(actual[row*width:(row+1)*width], expected):
                assert math.isfinite(value)
                assert abs(value-reference) <= atol+rtol*abs(reference)


def test_copy_from_existing_ptx_home_does_not_steal_its_producer():
    ptx = """.version 8.7
.target sm_80
.address_size 64
.visible .entry home_copy(.param .u64 out) {
.reg .b64 %d; .reg .f32 %f<2>; .reg .u32 %r; .reg .pred %p;
ld.param.u64 %d,[out]; mov.u32 %r,%tid.x; setp.ge.u32 %p,%r,1; @%p bra done;
mov.f32 %f0,0f40000000; add.f32 %f0,%f0,0f3f800000;
mov.f32 %f1,%f0; add.f32 %f1,%f1,0f3f800000;
st.global.f32 [%d],%f0; st.global.f32 [%d+4],%f1;
done: ret; }
"""
    _, _, bad = compile_and_diff(ptx,"home_copy",{"out":bytes(8)},{},block=(32,1,1))
    assert not bad, bad


def test_packed_env_alias_preserves_a_component_and_fresh_bind_still_retargets():
    selector = ra.Selector(Kernel("test", [], [], {}))
    old, other = selector.new(), selector.new()
    selector.homes = {"target":None}
    selector.env = {"packed":(old,other)}
    producer = selector.emit("MOV {d0}, 0x7",[old],[])
    selector.bind("target", old)
    assert producer.defs == [old]
    assert selector.ops[-1].uses == [old]
    fresh = selector.new()
    producer = selector.emit("MOV {d0}, 0x8",[fresh],[])
    selector.bind("target",fresh)
    assert producer.defs == [selector.homes["target"]]


def _load(dest, base):
    return Out(f"LDG.E R{dest}, [R{base}.64]", True, [base,base+1], [dest])


def test_multiple_async_readers_are_all_waited_before_source_overwrite():
    code = [_load(20,10),_load(24,10),Out("MOV R10, R9",False,[9],[10])]
    scheduled = schedule(code)
    assert scheduled[2][1].wait & ((1<<1)|(1<<3)) == (1<<1)|(1<<3)


def test_waiting_a_reader_retires_all_of_its_source_aliases():
    code = [_load(20,10),Out("MOV R10, R9",False,[9],[10]),Out("MOV R11, R9",False,[9],[11])]
    scheduled = schedule(code)
    assert scheduled[1][1].wait == 1<<1
    assert scheduled[2][1].wait == 0


def test_recycled_write_and_read_slots_retire_old_generations_before_publish():
    code = [_load(20,10),_load(24,12),_load(26,14),_load(28,16),
            Out("MOV R30, R28",False,[28],[30])]
    scheduled = schedule(code)
    fourth = scheduled[3][1]
    assert fourth.wbar == 0 and fourth.rbar == 1
    assert fourth.wait & 3 == 3
    # Reusing a waited old W0 must not erase this instruction's new W0 mapping.
    assert scheduled[4][1].wait & 1


@pytest.mark.parametrize("text,reads,dest", [
    ("MUFU.EX2 R4, R8",[8],8),
    ("SHFL.BFLY PT, R4, R8, R9, 0x1f",[8,9],9),
])
def test_operand_collection_sources_remain_live_until_the_read_barrier(text,reads,dest):
    scheduled = schedule([Out(text,True,reads,[4]),Out(f"MOV R{dest}, R12",False,[12],[dest])])
    assert scheduled[0][1].wbar == 0 and scheduled[0][1].rbar == 1
    assert scheduled[1][1].wait & 2


def test_warpsync_drains_all_outstanding_scoreboards():
    scheduled = schedule([_load(20,10),_load(24,12),_load(26,14),
                          Out("WARPSYNC R31",False,[31],[])])
    assert scheduled[-1][1].wait == 0x3f


def test_generated_warpsync_register_template_covers_the_recorded_sweep():
    encoder = Encoder("sm_80")
    for register in (0,2,4,8,15,31,64,127,254):
        assert isinstance(encoder.encode(f"WARPSYNC R{register}"),int)


@pytest.mark.parametrize("instruction", [
    "shfl.bfly.b32 %r1,%r0,1,31;",
    "shfl.sync.idx.b32 %r1,%r0,1,31,0xffff;",
    "shfl.sync.bfly.b32 %r1,%r0,1,15,0xffffffff;",
    "shfl.sync.bfly.b32 %r1,%r0,1,31,0xffff;",
    "@%p shfl.sync.bfly.b32 %r1,%r0,1,31,0xffffffff;",
])
def test_unsupported_shuffle_forms_fail_closed(instruction):
    kernel = parse_module(".version 8.7\n.target sm_80\n.address_size 64\n"
        ".visible .entry bad() {.reg .u32 %r<2>; .reg .pred %p;"
        "mov.u32 %r0,%tid.x; setp.eq.u32 %p,%r0,0;"+instruction+"ret;}").kernels["bad"]
    with pytest.raises(ra.NotSupported):
        native_generate(kernel,"sm_80","opt")
