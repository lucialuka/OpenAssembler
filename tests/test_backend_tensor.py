"""SM80 tensor selector, scheduler and bundled encoder regression evidence."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
from pathlib import Path
import struct
import sys

import pytest

from openassembler import oracle
from openassembler.backend import ra, tensor
from openassembler.backend.compile import Out
from openassembler.backend.tensor_encoding import TensorEncoder
from openassembler.emu.machine import Launch, Machine, Memory
from openassembler.emu.program import load_listing
from openassembler.isa import encode
from openassembler.isa.word import control, control_valid
from openassembler.kernels.gemm import GemmPlan, generate as gemm_source
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing


def kernel(source):
    return parse_module(source).kernels['gemm']


def packed_warp(dtype='f16'):
    one,two={'f16':('0x3c003c00','0x40004000'),
             'bf16':('0x3f803f80','0x40004000')}[dtype]
    return f''' .version 8.7
.target sm_80
.address_size 64
.visible .entry gemm(.param .u64 Out) {{
 .reg .b32 %r<8>; .reg .b64 %d<3>; .reg .f32 %c<4>;
 ld.param.u64 %d0,[Out]; mov.u32 %r0,%tid.x;
 mul.wide.u32 %d1,%r0,16; add.u64 %d2,%d0,%d1;
 mov.b32 %r1,{one}; mov.b32 %r2,{one};
 mov.b32 %r3,{one}; mov.b32 %r4,{one};
 mov.b32 %r5,{two}; mov.b32 %r6,{two};
 mov.f32 %c0,0f00000000; mov.f32 %c1,0f00000000;
 mov.f32 %c2,0f00000000; mov.f32 %c3,0f00000000;
 mma.sync.aligned.m16n8k16.row.col.f32.{dtype}.{dtype}.f32
 {{%c0,%c1,%c2,%c3}},{{%r1,%r2,%r3,%r4}},{{%r5,%r6}},{{%c0,%c1,%c2,%c3}};
 st.global.v2.f32 [%d2],{{%c0,%c1}};
 st.global.v2.f32 [%d2+8],{{%c2,%c3}};
 ret;
}}'''


@pytest.mark.parametrize('dtype',['f16','bf16'])
def test_straight_line_mma_native_pair_stores_and_poison(dtype):
    code,labels,words,exits,regs=tensor.generate(kernel(packed_warp(dtype)),'sm_80','safe')
    program=load_listing(listing(code,labels));output=0x100000
    for poison in (None,0xDEADBEEF):
        memory=Memory()
        Machine(program,Launch(block=(32,1,1),params=struct.pack('<Q',output),poison=poison),memory).run()
        actual=[struct.unpack('<f',struct.pack('<I',memory.read(output+4*i,4)))[0] for i in range(128)]
        assert actual==[32.0]*128
    assert sum(text.startswith('STG.E.64') for text,ctl in code)==2
    hmma=next(i for i,(text,ctl) in enumerate(code) if text.startswith('HMMA'))
    assert code[hmma][1].stall==8
    assert code[hmma+1][0]=='NOP' and code[hmma+1][1].stall==15
    assert code[hmma+1][1].yield_
    assert all(control_valid(control(word)) for word in words)


@pytest.mark.parametrize('shape,expected',[((4096,4096,4096),{
    'f16':'7db39cc746df3370ccb8158a8394ffa886024bc229a77fee0f53b63b7c475ddd',
    'bf16':'1573a311b93c91342d9c4437ef954624d350a07a07d2d2700ce1c31f2cef41d3'}),
    ((4096,16384,4096),{
    'f16':'a84a7705738b89ae2aa2cfc903db92c0ef34bc7b88726ffc75c95f066927a144',
    'bf16':'40927857547bb35ddae11575264f1f4c300bd4a8772a1c4cfe82c07518257db4'}),
    ((16384,4096,4096),{
    'f16':'7db39cc746df3370ccb8158a8394ffa886024bc229a77fee0f53b63b7c475ddd',
    'bf16':'1573a311b93c91342d9c4437ef954624d350a07a07d2d2700ce1c31f2cef41d3'})])
@pytest.mark.parametrize('dtype',['f16','bf16'])
def test_production_words_match_all_six_gpu_validated_phase3_kernels(shape,expected,dtype):
    # Fixed hashes cover every encoded instruction, control word, label offset
    # and padding word from the independently retained phase3 cubin listings.
    source=gemm_source(GemmPlan(*shape,dtype,schedule='stagepair'))
    code,labels,words,exits,regs=tensor.generate(kernel(source),'sm_80','safe')
    assert hashlib.sha256(oracle.words_to_bytes(words)).hexdigest()==expected[dtype]
    assert len(words)==1600 and regs==216
    assert all(control_valid(ctl) for text,ctl in code)


def test_no_oracle_corpus_bootstrap_writes_or_import_path_mutation(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('runtime discovery/write was called')
    monkeypatch.setattr(oracle,'run_tool',forbidden)
    monkeypatch.setattr(oracle,'disasm_words',forbidden)
    monkeypatch.setattr(encode,'field_probe',forbidden)
    monkeypatch.setattr(encode,'synthesize',forbidden)
    monkeypatch.setattr(encode.Encoder,'__init__',forbidden)
    monkeypatch.setattr(Path,'write_text',forbidden)
    monkeypatch.setattr(Path,'write_bytes',forbidden)
    def no_corpus_read(self,*args,**kwargs):
        raise AssertionError(f'tensor path should need no runtime file: {self}')
    monkeypatch.setattr(Path,'read_text',no_corpus_read)
    before=list(sys.path)
    source=gemm_source(GemmPlan(128,128,256,'f16'))
    k=kernel(source)
    outputs=[tensor.generate(k,'sm_80',level) for level in ('safe','mined','sched','opt')]
    assert all(result==outputs[0] for result in outputs)
    assert sys.path==before


def test_frozen_per_call_policy_and_no_external_state_leaks():
    config=tensor.TensorConfig()
    with pytest.raises(FrozenInstanceError):config.issue_gap=4
    with pytest.raises(ValueError):tensor.TensorConfig(issue_gap=4)
    descriptor=tensor.policy();descriptor['copy_group_scoreboard']=5
    assert tensor.policy()['copy_group_scoreboard']==0


def test_detection_and_strict_rejection():
    scalar=kernel(packed_warp().split(' mma.sync')[0]+' ret; }')
    assert not tensor.needs_tensor(scalar)
    with pytest.raises(ra.NotSupported,match='does not require'):tensor.generate(scalar,'sm_80','safe')
    k=kernel(packed_warp())
    assert tensor.needs_tensor(k)
    for arch in ('sm_86','sm_90'):
        with pytest.raises(ra.NotSupported,match='only sm_80'):tensor.generate(k,arch,'safe')
    for change in ('m16n8k8','col.row','f32.tf32.tf32.f32'):
        source=packed_warp().replace('m16n8k16',change) if change=='m16n8k8' else (
            packed_warp().replace('row.col','col.row') if change=='col.row' else
            packed_warp().replace('f32.f16.f16.f32',change))
        bad=kernel(source);assert tensor.needs_tensor(bad)
        with pytest.raises(ra.NotSupported):tensor.generate(bad,'sm_80','safe')


def test_native_b_and_store_ir_preserve_all_implicit_source_reads():
    selector=tensor.TensorSelector(kernel(gemm_source(GemmPlan(128,128,256,'f16'))))
    ops=ra.coalesce(ra.dce(selector.run()));phys=ra.allocate(ops,selector.r1)
    assert phys[selector.r1]==1
    groups=ra._groups(ops)
    for group in set(groups.values()):
        assert phys[group[0]]%len(group)==0
        assert [phys[v] for v in group]==list(range(phys[group[0]],phys[group[0]]+len(group)))
    for op in ops:
        if op.fmt.startswith('HMMA'):
            assert len(op.uses)==4 and phys[op.uses[3]]==phys[op.uses[1]]+1
        if op.fmt.startswith('STG.E.64'):
            assert len(op.uses)==3 and phys[op.uses[2]]==phys[op.uses[1]]+1


def test_scheduler_keeps_every_ldsm_address_war_and_async_group_edge():
    body=[Out('LDSM.16.M88.4 R20, [R10]',True,[10],[20,21,22,23]),
          Out('LDSM.16.M88.4 R24, [R10]',True,[10],[24,25,26,27]),
          Out('MOV R10, R11',False,[11],[10]),Out('LDGDEPBAR'),
          Out('DEPBAR.LE SB0, 0x0')]
    code=tensor.tensor_schedule(body,{})
    assert code[0][1].wbar==1 and code[1][1].wbar==2
    assert code[0][1].rbar==7 and code[1][1].rbar==7
    assert code[2][1].wait&0b110==0b110
    assert code[3][1].wbar==0
    # Recycling an in-flight ordinary scoreboard must retire its previous use.
    loads=[Out(f'LDSM.16.M88.4 R{20+4*i}, [R10]',True,[10],list(range(20+4*i,24+4*i))) for i in range(6)]
    recycled=tensor.tensor_schedule(loads,{})
    assert recycled[5][1].wbar==1 and recycled[5][1].wait&(1<<1)
    with pytest.raises(ra.NotSupported,match='without'):tensor.tensor_schedule([Out('DEPBAR.LE SB0, 0x0')],{})


@pytest.mark.parametrize('text',[
    'LDGSTS.E.BYPASS.128 [R0+0x80000], [R2.64]',
    'LDGSTS.E.BYPASS.128 [R0], [R2.64+0x800]',
    'LDGSTS.E.BYPASS.128 [R0+0x1], [R2.64]',
    'LDGSTS.E.BYPASS.128 [R0], [R3.64]',
    'LDGSTS.E.BYPASS.128.ZFILL [R0], [R2.64]',
    'DEPBAR.LE SB1, 0x0','DEPBAR.LE SB0, 0x3'])
def test_encoder_refuses_unproven_forms(text):
    with pytest.raises(encode.EncodeError):TensorEncoder().encode(text)


@pytest.mark.oracle
def test_bundled_tensor_words_roundtrip_against_nvdisasm():
    if not oracle.available('nvdisasm'):pytest.skip(oracle.NEEDS_TOOLKIT)
    enc=TensorEncoder()
    texts=['HMMA.16816.F32 R20, R40, R62, R20',
           'HMMA.16816.F32.BF16 R20, R40, R62, R20',
           'LDSM.16.M88.4 R200, [R12+0x800]',
           'LDSM.16.M88.2 R202, [R14]',
           'LDSM.16.MT88.2 R202, [R14]',
           'LDGSTS.E.BYPASS.128 [R18+0x4000], [R34.64-0x10]',
           'LDGSTS.E.BYPASS.128 [R254+0x7fff0], [R252.64+0x7f0]',
           'DEPBAR.LE SB0, 0x0','DEPBAR.LE SB0, 0x1','DEPBAR.LE SB0, 0x2']
    words=[enc.encode(text) for text in texts]
    # nvdisasm renders negative LDGSTS displacements as "+-0x10".
    assert [text.replace('+-','-') for text in oracle.disasm_words(words)]==texts
