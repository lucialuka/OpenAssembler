"""Contract and semantic checks for the standalone SM80 GEMM source emitter."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from hashlib import sha256
import random
import struct

import pytest

from openassembler.kernels import GemmPlan, generate
from openassembler.ptx.parse import Addr, ImmI, Var, parse_module


# SHA256 of the complete PTX bytes used by the passing phase-three A100
# experiments. Golden constants keep these tests independent of exp/ files.
_GOLDEN = {
    ("f16",4096,"stagepair"): "25df2a8950638940c87ab9460c01dd92883610f0e93b26159e410e9f8b6adb6a",
    ("f16",4096,"combined"): "83de480dac0f33e28677d5b779fb7253cfe2a78a25a05309a4a910a32f5948b5",
    ("f16",16384,"stagepair"): "6b63b83c78663c9c74cdb7e9bbaa9984ba4395cdd8f8b43f483ec687c270c920",
    ("f16",16384,"combined"): "9ac76ad843b01ed51dd0e2efde6285ddb52d33f5d4a949d268fae6b1417e1c32",
    ("bf16",4096,"stagepair"): "8966bd075c2cc65bf433744ed12e606743d5898be29fffae8c38c71ec3a76550",
    ("bf16",4096,"combined"): "237f31cfd6a1586847150e2072b6bf6004ee8f2d46d2d99b3fb2428c4ea21490",
    ("bf16",16384,"stagepair"): "11a4203c26ddb9e64a741abb52d01c44a078f6e9c8b7e3a86d541f9a9f8d3fc9",
    ("bf16",16384,"combined"): "c0f526ec1e7559a11725f31fad8bbbfea197f0d8706e38fa9668dbd3318b0042",
}


@pytest.mark.parametrize("m,n",[(4096,4096),(4096,16384),(16384,4096)])
@pytest.mark.parametrize("dtype",["f16","bf16"])
@pytest.mark.parametrize("schedule",["stagepair","combined"])
def test_source_is_byte_exact_to_gpu_validated_schedule(m,n,dtype,schedule):
    source = generate(GemmPlan(m,n,4096,dtype,schedule))
    assert sha256(source.encode()).hexdigest() == _GOLDEN[dtype,n,schedule]
    kernel = parse_module(source).kernels["gemm"]
    assert kernel.params == [("u64","A"),("u64","B"),("u64","C")]
    assert all(ins.mods[:6] == ("sync","aligned","m16n8k16","row","col","f32")
               for ins in kernel.body if ins.op == "mma")


@pytest.mark.parametrize("change",[
    {"m":0},{"n":-128},{"k":64},{"m":129},{"n":255},{"k":192},
    {"m":128.0},{"n":True},{"k":"128"},{"dtype":"f32"},{"dtype":"fp16"},
    {"output_dtype":"f16"},{"layout":"rrr"},{"arch":"sm_90"},
    {"schedule":"automatic"},{"m":1<<32},{"n":65536*128},
])
def test_invalid_or_unimplemented_contract_is_rejected(change):
    args = dict(m=128,n=128,k=128)
    args.update(change)
    with pytest.raises(ValueError):
        GemmPlan(**args)


@pytest.mark.parametrize("change",[
    {"cta_order":"unknown"}, {"epilogue":"strassen"},
    {"cta_order":"morton8","m":128,"n":1024},
    {"fragment_schedule":"unknown"},
    {"fragment_schedule":"load_next","m":2048,"n":8192,"k":1024},
    {"fragment_schedule":"load_next","pointer_mode":"direct"},
    {"fragment_schedule":"load_next","pointer_advance":"early"},
])
def test_new_plan_options_reject_unsupported_modes_and_incomplete_morton_groups(change):
    args=dict(m=128,n=128,k=128)
    args.update(change)
    with pytest.raises(ValueError):
        GemmPlan(**args)


def test_morton8_emitted_coordinate_program_is_a_full_grid_bijection():
    from openassembler.ptx.interp import Launch, run_kernel

    plan=GemmPlan(2048,3072,128,cta_order="morton8")
    source=generate(plan)
    prefix=source.split("and.b32 %r3,%r0,31;",1)[0]+"ret;\n}\n"
    module=parse_module(prefix)
    mapped=[]
    tile_m,tile_n=plan.m//128,plan.n//128
    for by in range(tile_n):
        for bx in range(tile_m):
            interp=run_kernel(module,"gemm",Launch(block=(32,1,1),grid=(tile_m,tile_n,1),
                              ctaid=(bx,by,0),params={"A":0x1000,"B":0x2000,"C":0x3000}),None)
            regs=interp.warps[0][0].regs
            mapped.append((regs["%r1"]//128,regs["%r2"]//128))
    expected=[]
    for by in range(tile_n):
        for bx in range(tile_m):
            linear=by*tile_m+bx
            group=linear>>6
            local=linear&63
            local_m=sum(((local>>(2*bit))&1)<<bit for bit in range(3))
            local_n=sum(((local>>(2*bit+1))&1)<<bit for bit in range(3))
            expected.append(((group%(tile_m//8))*8+local_m,
                             (group//(tile_m//8))*8+local_n))
    assert mapped==expected
    assert len(set(mapped))==tile_m*tile_n


def test_grouped_n8_emits_public_mapping_and_launch_contract():
    plan = GemmPlan(16384, 4096, 4096, cta_order="grouped_n8")
    source = generate(plan)
    assert "shr.u32 %r700,%r1,3;" in source
    assert "and.b32 %r701,%r1,7;" in source
    assert "mul.lo.u32 %r702,%r2,8;" in source
    assert plan.launch == {"grid": [1024, 4, 1], "threads": 128, "smem": 65536}
    assert plan.metadata["cta_order"] == "grouped_n8"

    from openassembler.ptx.interp import Launch, run_kernel
    from openassembler.ptx.parse import parse_module
    prefix = source.split("and.b32 %r3,%r0,31;", 1)[0] + "ret;\n}\n"
    module = parse_module(prefix)
    tile_m, tile_n = plan.m // 128, plan.n // 128
    mapped = []
    grid_x, grid_y, _ = plan.launch["grid"]
    for by in range(grid_y):
        for bx in range(grid_x):
            result = run_kernel(module, "gemm", Launch(block=(32, 1, 1),
                grid=(grid_x, grid_y, 1), ctaid=(bx, by, 0),
                params={"A": 0x1000, "B": 0x2000, "C": 0x3000}), None)
            regs = result.warps[0][0].regs
            mapped.append((regs["%r1"] // 128, regs["%r2"] // 128))
    assert set(mapped) == {(m, n) for m in range(tile_m) for n in range(tile_n)}
    assert len(mapped) == len(set(mapped)) == tile_m * tile_n


@pytest.mark.parametrize("n", [128, 896])
def test_grouped_n8_rejects_incomplete_n_groups(n):
    with pytest.raises(ValueError, match="grouped_n8"):
        GemmPlan(128, n, 128, cta_order="grouped_n8")


def test_plan_is_immutable_and_launch_metadata_is_independent():
    plan = GemmPlan(256,384,256,"bf16")
    assert plan.launch == {"grid":[2,3,1],"threads":128,"smem":65536}
    assert plan.metadata["accumulation"] == plan.output_dtype == "f32"
    assert plan.metadata["shared_swizzle"] == [7,7]
    launch = plan.launch
    launch["grid"][0] = 17
    assert plan.launch["grid"] == [2,3,1]
    with pytest.raises(FrozenInstanceError):
        plan.k = 512
    with pytest.raises(TypeError):
        generate({"m":128,"n":128,"k":128})


def _trace_pipeline(plan,tid):
    """Execute scalar PTX and track copy tile identity through both stages.

    This checks real emitted address/control-flow instructions, not a second
    copy of the generator's loop. Tensor arithmetic itself is covered by the
    complete golden-source match and native tensor-backend emulator tests.
    """
    kernel = parse_module(generate(plan)).kernels["gemm"]
    a,b,c = 0,1<<48,2<<48
    regs = {"%tid.x":tid,"%ctaid.x":plan.m//128-1,"%ctaid.y":plan.n//128-1}
    params = {"A":a,"B":b,"C":c}
    stages, pending, consumed, reads, stores = {},[],[],[],[]
    compute_tile = None
    visible = False
    barriers = 0
    steps = 0
    def value(arg):
        if isinstance(arg,ImmI): return arg.value
        if isinstance(arg,Var): return regs[arg.name]
        if isinstance(arg,Addr): return regs[arg.base]+arg.off
        raise AssertionError(arg)
    pc = 0
    while pc < len(kernel.body):
        steps += 1
        assert steps < 500000
        ins = kernel.body[pc];pc += 1
        op,args = ins.op,ins.args
        if ins.guard is not None and not (regs[ins.guard[0]] ^ ins.guard[1]):
            continue
        if op == "ld" and "param" in ins.mods:
            regs[args[0].name] = params[args[1].base]
        elif op == "mov" and "f32" not in ins.mods:
            regs[args[0].name] = value(args[1])
        elif op in ("add","mul","mad","and","xor","shr"):
            values = [value(arg) for arg in args[1:]]
            if op == "add": result = values[0]+values[1]
            elif op == "mul": result = values[0]*values[1]
            elif op == "mad": result = values[0]*values[1]+values[2]
            elif op == "and": result = values[0]&values[1]
            elif op == "xor": result = values[0]^values[1]
            else: result = values[0]>>values[1]
            width = 64 if "u64" in ins.mods or "wide" in ins.mods else 32
            regs[args[0].name] = result & ((1<<width)-1)
        elif op == "setp":
            assert ins.mods == ("lt","u32")
            regs[args[0].name] = value(args[1]) < value(args[2])
        elif op == "bra":
            pc = kernel.labels[args[0].name]
        elif op == "cp" and "global" in ins.mods:
            shared, glob = value(args[0]),value(args[1])
            which = int(glob >= b)
            base,rows = (b,plan.n) if which else (a,plan.m)
            index = (glob-base)//2
            assert 0 <= index < rows*plan.k and glob % 16 == 0
            row,col = divmod(index,plan.k)
            assert col % 8 == 0 and col+8 <= plan.k
            stage = shared//32768
            assert stage in (0,1) and shared % 16 == 0
            assert (shared%32768)//16384 == which
            pending.append((stage,col//64))
        elif op == "cp" and "commit_group" in ins.mods:
            assert len(pending) == 16 and len(set(pending)) == 1
            stage,tile = pending[0]
            stages[stage] = (tile,False)
            pending.clear()
        elif op == "cp" and "wait_group" in ins.mods:
            assert value(args[0]) == 0
            stages = {stage:(tile,True) for stage,(tile,_) in stages.items()}
            visible = False
            compute_tile = None
        elif op == "bar":
            barriers += 1
            visible = True
        elif op == "ldmatrix":
            assert visible
            stage = value(args[1])//32768
            tile,ready = stages[stage]
            assert ready
            if compute_tile is None:
                compute_tile = tile; consumed.append(tile)
            assert tile == compute_tile
        elif op == "mma":
            assert compute_tile is not None
            reads.append(compute_tile)
        elif op == "st":
            assert ins.mods == ("global","v2","f32")
            index = (value(args[0])-c)//4
            assert 0 <= index < plan.m*plan.n-1 and index % 2 == 0
            stores += [index,index+1]
        elif op == "ret":
            break
        else:
            assert op == "mov" and "f32" in ins.mods,(op,ins.mods)
    assert consumed == list(range(plan.k//64))
    assert barriers == 2*(plan.k//64)
    assert len(reads) == 128*(plan.k//64)
    assert len(stores) == 128 and len(set(stores)) == 128
    return stores


@pytest.mark.parametrize("k",[128,256,384,4096])
@pytest.mark.parametrize("schedule",["stagepair","combined"])
def test_emitted_pipeline_waits_and_consumes_every_in_bounds_tile(k,schedule):
    # Last CTA with lanes on fragment boundaries checks high addresses and
    # both halves of the B x4 address map without requiring a GPU toolkit.
    plan = GemmPlan(256,384,k,schedule=schedule)
    for tid in (0,15,16,31,32,64,127):
        _trace_pipeline(plan,tid)


@pytest.mark.parametrize("schedule",["stagepair","combined"])
def test_vector_output_covers_the_complete_last_cta_once(schedule):
    plan = GemmPlan(128,128,128,schedule=schedule)
    stores = [index for tid in range(128) for index in _trace_pipeline(plan,tid)]
    assert sorted(stores) == list(range(128*128))


@pytest.mark.parametrize("dtype,k",[("f16",128),("bf16",128),("f16",384),("bf16",384)])
def test_native_stagepair_backend_matches_complete_integer_oracle(dtype,k):
    from openassembler.backend.tensor import generate as compile_tensor
    from openassembler.emu.machine import Launch,Machine,Memory
    from openassembler.emu.program import load_listing
    from tests.backend_harness import listing

    plan = GemmPlan(128,128,k,dtype)
    kernel = parse_module(generate(plan)).kernels["gemm"]
    code,labels,words,exits,registers = compile_tensor(kernel,"sm_80","safe")
    rng = random.Random(198)
    av = [rng.randrange(-2,3) for _ in range(plan.m*k)]
    bv = [rng.randrange(-2,3) for _ in range(plan.n*k)]
    expected = [sum(av[row*k+i]*bv[col*k+i] for i in range(k))
                for row in range(plan.m) for col in range(plan.n)]
    def pack(values):
        if dtype == "f16":
            return b"".join(struct.pack("<e",value) for value in values)
        return b"".join(struct.pack("<H",struct.unpack("<I",struct.pack("<f",value))[0]>>16)
                        for value in values)
    a,b,c = 0x100000,0x200000,0x300000
    program = load_listing(listing(code,labels))
    memory = Memory();memory.write_bytes(a,pack(av));memory.write_bytes(b,pack(bv))
    memory.write_bytes(c,b"\x00\x00\xc0\x7f"*(plan.m*plan.n))
    Machine(program,Launch(grid=(1,1,1),block=(128,1,1),
                           params=struct.pack("<QQQ",a,b,c),smem_static=65536,
                           poison=0xDEADBEEF),memory).run(max_steps=6000000)
    got = [struct.unpack("<f",struct.pack("<I",memory.read(c+4*i,4)))[0]
           for i in range(plan.m*plan.n)]
    assert got == expected
    assert len(words) >= len(code) and len(words) % 8 == 0 and exits
    assert registers <= 255
    assert sum(text.startswith("STG.E.64") for text,ctl in code) == 64


def test_native_fused_bias_residual_relu_matches_full_integer_oracle():
    from openassembler.backend.tensor import generate as compile_tensor
    from openassembler.emu.machine import Launch,Machine,Memory
    from openassembler.emu.program import load_listing
    from tests.backend_harness import listing

    plan=GemmPlan(128,128,128,"f16",epilogue="bias_residual_relu")
    kernel=parse_module(generate(plan)).kernels["gemm"]
    code,labels,words,exits,registers=compile_tensor(kernel,"sm_80","safe")
    av=[(i%5)-2 for i in range(plan.m*plan.k)]
    bv=[(i%7)-3 for i in range(plan.n*plan.k)]
    bias=[(i%9)-4 for i in range(plan.n)]
    residual=[(i%11)-5 for i in range(plan.m*plan.n)]
    expected=[]
    for row in range(plan.m):
        for col in range(plan.n):
            value=sum(av[row*plan.k+i]*bv[col*plan.k+i] for i in range(plan.k))
            expected.append(max(0,value+bias[col]+residual[row*plan.n+col]))
    pack_half=lambda values:b"".join(struct.pack("<e",value) for value in values)
    pack_float=lambda values:b"".join(struct.pack("<f",value) for value in values)
    a,b,c,bi,res=0x100000,0x200000,0x300000,0x400000,0x500000
    memory=Memory()
    memory.write_bytes(a,pack_half(av));memory.write_bytes(b,pack_half(bv))
    memory.write_bytes(bi,pack_float(bias));memory.write_bytes(res,pack_float(residual))
    memory.write_bytes(c,b"\x00\x00\xc0\x7f"*(plan.m*plan.n))
    Machine(load_listing(listing(code,labels)),Launch(grid=(1,1,1),block=(128,1,1),
           params=struct.pack("<QQQQQ",a,b,c,bi,res),smem_static=65536,poison=0xDEADBEEF),memory).run(max_steps=6000000)
    got=[struct.unpack("<f",struct.pack("<I",memory.read(c+4*i,4)))[0]
         for i in range(plan.m*plan.n)]
    assert got==expected
    assert len(words)>=len(code) and len(words)%8==0 and exits and registers<=255
