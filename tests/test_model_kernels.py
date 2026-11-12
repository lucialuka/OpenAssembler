"""Independent math checks on both PTX and OA scheduled SASS, including tails."""
import itertools
import math
import struct

import pytest

from openassembler.build import generate
from openassembler.emu.machine import Launch as SLaunch, Machine, Memory
from openassembler.emu.program import load_listing
from openassembler.modelfront.kernels import emit_kernel, prepare
from openassembler.ptx.interp import Launch as PLaunch, run_kernel
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing


def spec(*shape):
    return {"shape": list(shape), "dtype": "f32"}


def run(specification, data, *, scalar_rate=0.125):
    text = ".version 8.7\n.target sm_80\n.address_size 64\n" + specification.ptx
    module = parse_module(text)
    code, labels, _, _, _ = generate(module.kernels[specification.name], "sm_80", "opt")
    prog = load_listing(listing(code, labels))
    addresses = {f"a{i}": 0x10000*(i+1) for i in range(len(data))}
    addresses["output"] = 0x80000
    params = b"".join(struct.pack("<Q", ptr) for ptr in addresses.values())
    if specification.op == "sgd":
        addresses["rate"] = struct.unpack("<I", struct.pack("<f", scalar_rate))[0]
        params += struct.pack("<f", scalar_rate)
    n = math.prod(specification.output)
    results = []
    for mode in ("ptx", "sass", "poisoned"):
        memory = Memory()
        for i, values in enumerate(data):
            memory.write_bytes(addresses[f"a{i}"], struct.pack("<"+"f"*len(values), *values))
        memory.write_bytes(addresses["output"], b"\xa5"*(4*n+16))
        for x, y, z in itertools.product(*(range(g) for g in specification.grid)):
            if mode == "ptx":
                run_kernel(module, specification.name, PLaunch(block=(specification.threads,1,1),
                           grid=specification.grid, ctaid=(x,y,z), params=addresses), memory)
            else:
                Machine(prog, SLaunch(block=(specification.threads,1,1), grid=specification.grid,
                        ctaid=(x,y,z), params=params, poison=0xDEADBEEF if mode=="poisoned" else None), memory).run()
        got = [struct.unpack("<f", struct.pack("<I", memory.read(addresses["output"]+4*i,4)))[0] for i in range(n)]
        assert bytes(memory.read(addresses["output"]+4*n+j,1) for j in range(16)) == b"\xa5"*16
        for i, values in enumerate(data):
            assert bytes(memory.read(addresses[f"a{i}"]+j,1) for j in range(4*len(values))) == struct.pack("<"+"f"*len(values), *values)
        results.append(got)
    assert results[1] == results[0]
    assert results[2] == results[0]
    return results[0]


@pytest.mark.parametrize("op,attrs,expected", [
    ("relu", {}, lambda a,b:[max(x,0) for x in a]),
    ("relu_backward", {}, lambda a,b:[x if y>0 else 0 for x,y in zip(a,b)]),
    ("add", {}, lambda a,b:[x+y for x,y in zip(a,b)]),
    ("mul", {}, lambda a,b:[x*y for x,y in zip(a,b)]),
    ("add", {"scalar":0.25}, lambda a,b:[x+.25 for x in a]),
    ("mul", {"scalar":-.5}, lambda a,b:[x*-.5 for x in a]),
    ("copy", {}, lambda a,b:a),
    ("fill", {"value":-.25}, lambda a,b:[-.25]*len(a)),
    ("sgd", {}, lambda a,b:[x-.125*y for x,y in zip(a,b)]),
])
def test_elementwise_crosses_cta_tail_and_matches_independent_math(op,attrs,expected):
    a=[(i%13-6)*.25 for i in range(131)]
    b=[(i%9-4)*.5 for i in range(131)]
    data=[] if op=="fill" else [a] if op in ("relu","copy") or "scalar" in attrs else [a,b]
    k=emit_kernel(op,[spec(131)]*len(data),spec(131),attrs)
    assert run(k,data)==expected(a,b)


def test_linear_forward_dx_dw_and_bias_gradient():
    m,n,k=3,5,7
    x=[(i%7-3)*.25 for i in range(m*k)]
    w=[(i%9-4)*.5 for i in range(n*k)]
    bias=[i*.25 for i in range(n)]
    gy=[(i%5-2)*.25 for i in range(m*n)]
    forward=[sum(x[r*k+j]*w[c*k+j] for j in range(k))+bias[c] for r in range(m) for c in range(n)]
    dx=[sum(gy[r*n+j]*w[j*k+c] for j in range(n)) for r in range(m) for c in range(k)]
    dw=[sum(gy[j*n+r]*x[j*k+c] for j in range(m)) for r in range(n) for c in range(k)]
    db=[sum(gy[r*n+c] for r in range(m)) for c in range(n)]
    assert run(emit_kernel("linear",[spec(m,k),spec(n,k),spec(n)],spec(m,n)),[x,w,bias])==forward
    assert run(emit_kernel("linear_dx",[spec(m,n),spec(n,k)],spec(m,k)),[gy,w])==dx
    assert run(emit_kernel("linear_dw",[spec(m,n),spec(m,k)],spec(n,k)),[gy,x])==dw
    assert run(emit_kernel("bias_grad",[spec(m,n)],spec(n)),[gy])==db


@pytest.mark.parametrize('size', [112, 136, 513])
def test_ordered_mse_preserves_scalar_fp32_sum_in_ptx_sass_and_poisoned_execution(size):
    a = [(i % 19 - 9)*.125 for i in range(size)]
    b = [(i % 7 - 3)*.0625 for i in range(size)]
    rounded = lambda value: struct.unpack('<f', struct.pack('<f', value))[0]
    expected = 0.0
    for x,y in zip(a,b):
        delta = rounded(x-y)
        expected = rounded(expected + rounded(delta*delta))
    expected = rounded(expected * rounded(1/size))
    kernel = emit_kernel('mse', [spec(size)]*2, spec())
    assert kernel.threads == 32
    assert run(kernel,[a,b]) == [expected]


@pytest.mark.parametrize('size', [1, 33, 111, 137, 514, 16385])
def test_mse_unmeasured_or_rejected_sizes_keep_the_serial_policy(size):
    kernel = emit_kernel('mse', [spec(size)]*2, spec())
    assert kernel.threads == 1 and 'shfl' not in kernel.ptx


def test_transpose_and_reshape_no_runtime_division():
    values=list(range(15))
    kernel=emit_kernel("transpose2d",[spec(3,5)],spec(5,3))
    assert "div." not in kernel.ptx and "rem." not in kernel.ptx
    assert run(kernel,[values])==[values[r*5+c] for c in range(5) for r in range(3)]
    assert run(emit_kernel("copy",[spec(3,5)],spec(15)),[values])==values


def test_loss_and_gradient_include_tail_and_upstream():
    a=[i*.25-1 for i in range(13)]
    b=[.5]*13
    assert run(emit_kernel("mse",[spec(13)]*2,spec()),[a,b])[0] == pytest.approx(sum((x-y)**2 for x,y in zip(a,b))/13,rel=1e-6)
    assert run(emit_kernel("mse_grad",[spec(13),spec(13),spec()],spec(13)),[a,b,[.5]]) == pytest.approx([(x-y)/13 for x,y in zip(a,b)],rel=1e-6)


@pytest.mark.parametrize("op,ins,out,attrs",[
    ("linear",[spec(3,5),spec(7,4)],spec(3,7),{}),
    ("linear",[spec(3,5),spec(7,5)],spec(3,7),{"k":4}),
    ("add",[spec(3,5),spec(5)],spec(3,5),{}),
    ("mul",[spec(5)],spec(5),{}),
    ("transpose2d",[spec(3,5)],spec(3,5),{}),
    ("copy",[spec(3,5)],spec(14),{}),
    ("fill",[],spec(5),{"value":float("nan")}),
    ("linear",[spec(65536,1),spec(1,1)],spec(65536,1),{}),
])
def test_unsupported_contracts_reject(op,ins,out,attrs):
    with pytest.raises(ValueError):
        emit_kernel(op,ins,out,attrs)


def test_invalid_plan_use_def_rejects():
    with pytest.raises(ValueError,match="undefined"):
        prepare({"schema":1,"inputs":[],"parameters":[],"buffers":{"z":spec(3)},
                 "steps":[{"kernel":"relu","inputs":["missing"],"output":"z"}],"outputs":["z"]})


@pytest.mark.parametrize("width",[1,3,33,513])
def test_cooperative_softmax_and_backward_row_column_tails(width):
    rows=5
    x=[(i%17-8)*.125 for i in range(rows*width)]
    gy=[(i%11-5)*.125 for i in range(rows*width)]
    expected=[]
    for row in range(rows):
        values=x[row*width:(row+1)*width];nums=[math.exp(v-max(values)) for v in values]
        expected.extend(n/sum(nums) for n in nums)
    y=run(emit_kernel("softmax",[spec(rows,width)],spec(rows,width)),[x])
    assert y==pytest.approx(expected,rel=2e-6,abs=2e-7)
    expected_grad=[]
    for row in range(rows):
        yy=y[row*width:(row+1)*width];g=gy[row*width:(row+1)*width]
        total=sum(a*b for a,b in zip(yy,g));expected_grad.extend(a*(b-total) for a,b in zip(yy,g))
    assert run(emit_kernel("softmax_backward",[spec(rows,width)]*2,spec(rows,width)),[gy,y])==pytest.approx(expected_grad,rel=2e-5,abs=2e-7)


def test_tanh_gelu_stable_backward_including_saturation_and_tail():
    x=[-12,-8,-4,-3,-2,-.752,-.75,-.748,-1e-6,0,1e-6,.75,2,3,4,8,12]*9
    gy=[(-1 if i%2 else 1)*(1+(i%3)*.5) for i in range(len(x))]
    expected=[];dx=[]
    for v,g in zip(x,gy):
        u=math.sqrt(2/math.pi)*(v+.044715*v**3);t=math.tanh(u)
        expected.append(.5*v*(1+t))
        dx.append(g*(.5*(1+t)+.5*v*(1-t*t)*math.sqrt(2/math.pi)*(1+3*.044715*v*v)))
    assert run(emit_kernel("gelu",[spec(len(x))],spec(len(x))),[x])==pytest.approx(expected,rel=1e-4,abs=2e-6)
    assert run(emit_kernel("gelu_backward",[spec(len(x))]*2,spec(len(x))),[gy,x])==pytest.approx(dx,rel=1e-4,abs=2e-6)


@pytest.mark.parametrize("width",[1,3,33,513])
def test_layernorm_full_affine_derivatives_with_tails(width):
    rows=5
    x=[1000+(i%13-6)*.125 for i in range(rows*width)]
    gy=[(i%11-5)*.125 for i in range(rows*width)]
    weight=[.5+(i%5)*.125 for i in range(width)]
    bias=[(i%7-3)*.125 for i in range(width)]
    expected=[];inverse=[]
    for row in range(rows):
        values=x[row*width:(row+1)*width];mean=sum(values)/width;var=sum((v-mean)**2 for v in values)/width
        inv=1/math.sqrt(var+1e-5);inverse.append(inv);expected.extend((v-mean)*inv for v in values)
    y=run(emit_kernel("layernorm",[spec(rows,width)],spec(rows,width),{"epsilon":1e-5}),[x])
    assert y==pytest.approx(expected,rel=2e-5,abs=1e-5)
    affine=run(emit_kernel("channel_affine",[spec(rows,width),spec(width),spec(width)],spec(rows,width)),[y,weight,bias])
    assert affine==pytest.approx([v*weight[i%width]+bias[i%width] for i,v in enumerate(y)],rel=2e-6,abs=1e-6)
    dn=run(emit_kernel("channel_mul",[spec(rows,width),spec(width)],spec(rows,width)),[gy,weight])
    assert dn==pytest.approx([v*weight[i%width] for i,v in enumerate(gy)],rel=2e-6,abs=1e-6)
    dx=[]
    for row in range(rows):
        yy=y[row*width:(row+1)*width];gg=dn[row*width:(row+1)*width]
        a=sum(gg)/width;b=sum(g*v for g,v in zip(gg,yy))/width
        dx.extend(inverse[row]*(g-a-v*b) for g,v in zip(gg,yy))
    assert run(emit_kernel("layernorm_backward",[spec(rows,width)]*2,spec(rows,width),{"epsilon":1e-5}),[dn,x])==pytest.approx(dx,rel=2e-4,abs=2e-5)
    assert run(emit_kernel("channel_grad",[spec(rows,width)]*2,spec(width)),[gy,y])==pytest.approx([sum(gy[r*width+c]*y[r*width+c] for r in range(rows)) for c in range(width)],rel=2e-5,abs=2e-6)
    assert run(emit_kernel("channel_sum",[spec(rows,width)],spec(width)),[gy])==pytest.approx([sum(gy[r*width+c] for r in range(rows)) for c in range(width)],rel=2e-6,abs=1e-6)
