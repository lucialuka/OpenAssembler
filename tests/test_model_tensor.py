"""Explicit tensor recipe contracts and exact FP32-to-half packing semantics."""
from __future__ import annotations
import struct
import pytest

from openassembler.build import generate
from openassembler.emu.machine import Launch,Machine,Memory
from openassembler.emu.program import load_listing
from openassembler.modelfront.kernels_tensor import MATH_MODE,_pack,emit_tensor_recipe
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing


def spec(shape):return {'shape':list(shape),'dtype':'f32'}


def test_tensor_linear_requires_explicit_precision_and_supported_full_tiles():
    with pytest.raises(ValueError,match='explicit math_mode'):
        emit_tensor_recipe('linear',[spec((128,128)),spec((128,128))],spec((128,128)))
    with pytest.raises(ValueError,match='multiple'):
        emit_tensor_recipe('linear',[spec((127,128)),spec((128,128))],spec((127,128)),{'math_mode':MATH_MODE})
    for op,left,right,out in [('linear',(128,512),(256,512),(128,256)),
                               ('linear_dx',(128,256),(256,512),(128,512)),
                               ('linear_dw',(128,256),(128,512),(256,512))]:
        recipe=emit_tensor_recipe(op,[spec(left),spec(right)],spec(out),{'math_mode':MATH_MODE})
        assert recipe.signature['math_mode']==MATH_MODE
        assert len(recipe.stages)==3
        assert all(x['dtype']=='f16' for x in recipe.workspaces.values())
        assert recipe.stages[-1].gemm_plan.output_dtype=='f32'
        assert recipe.stages[-1].launch['smem'] in (32768,65536)


@pytest.mark.parametrize('transpose',[False,True])
def test_pack_half_exact_rounding_boundaries_and_signs_in_emulator(transpose):
    bits=[0,0x80000000,1,0x80000001,0x33000000,0x33000001,0x33800000,0x387fffff,
          0x38800000,0x3f801000,0x3f803000,0xbf801000,0x477fe000,0x477ff000,
          0x7f800000,0xff800000,0x7fc00001,0xffc00001,0x3f800000,0xbf800000,
          0x38000000,0xb8000000,0x007fffff,0x807fffff,0x3f000000,0x40000000,
          0x42000000,0xc2000000,0x477fefff,0xc77fefff,0x477ff001,0xc77ff001]
    shape=(4,8)
    stage=_pack(shape,transpose,'a0','output')
    code,labels,*_=generate(parse_module(stage.ptx).kernels[stage.name],'sm_80','opt')
    p=load_listing(listing(code,labels));source=0x100000;output=0x200000
    memory=Memory();memory.write_bytes(source,struct.pack('<32I',*bits))
    for row in range(shape[1] if transpose else shape[0]):
        Machine(p,Launch(block=(128,1,1),grid=tuple(stage.launch['grid']),ctaid=(0,row,0),
                         params=struct.pack('<QQ',source,output),poison=0xdeadbeef),memory).run()
    expected=[]
    for row in range(shape[1] if transpose else shape[0]):
        for col in range(shape[0] if transpose else shape[1]):
            raw=bits[col*shape[1]+row if transpose else row*shape[1]+col]
            value=struct.unpack('<f',struct.pack('<I',raw))[0]
            if raw&0x7fffffff>0x7f800000:h=(raw>>16&0x8000)|0x7e00
            else:
                try:h=int.from_bytes(struct.pack('<e',value),'little')
                except OverflowError:h=(raw>>16&0x8000)|0x7c00
            expected.append(h)
    assert [memory.read(output+2*i,2) for i in range(32)]==expected
