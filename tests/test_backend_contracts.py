"""Independent semantic and negative-admission tests for scalar PTX modifiers."""
import random
import struct
import pytest

from openassembler.backend import ra
from openassembler.build import generate
from openassembler.emu.machine import Launch, Machine, Memory
from openassembler.emu.program import load_listing
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing, pack_u32

HEADER = '.version 8.7\n.target sm_80\n.address_size 64\n'


def kernel_source(op, *, guarded=False):
    compare = op.startswith('setp')
    instruction = (f'{op} %p,%a,%b; selp.u32 %v,1,0,%p;' if compare else
                   ('setp.eq.u32 %p,%a,%a; @%p ' if guarded else '') + f'{op} %v,%a,%b;')
    return HEADER + '''.entry probe(.param .u64 out,.param .u32 a,.param .u32 b) {
.reg .u64 %d; .reg .b32 %a,%b,%v; .reg .pred %p;
ld.param.u64 %d,[out]; ld.param.u32 %a,[a]; ld.param.u32 %b,[b];
''' + instruction + '\nst.global.u32 [%d],%v; ret; }\n'


def execute(text, level, pairs):
    kernel = parse_module(text).kernels['probe']
    code, labels, words, _, _ = generate(kernel, 'sm_80', level)
    assert words  # Encoding every selected instruction is part of this gate.
    program = load_listing(listing(code, labels))
    for a,b in pairs:
        outputs=[]
        for poison in (None, 0xdeadbeef):
            memory=Memory()
            Machine(program, Launch(block=(1,1,1),params=struct.pack('<QII',0x100000000,a,b),poison=poison),memory).run()
            outputs.append(memory.read(0x100000000,4))
        assert outputs[0] == outputs[1]
        yield outputs[0]


@pytest.mark.parametrize('level',['safe','mined','sched','opt'])
@pytest.mark.parametrize('op',['add','sub'])
def test_signed_saturation_boundaries_random_and_guarded(level,op):
    edges=[0,1,0xffffffff,0x7fffffff,0x80000000,0x7ffffffe,0x80000001]
    rng=random.Random(19011)
    pairs=[(a,b) for a in edges for b in edges]+[(rng.getrandbits(32),rng.getrandbits(32)) for _ in range(128)]
    signed=lambda x:x if x<0x80000000 else x-2**32
    expected=[min(2**31-1,max(-2**31,signed(a)+(signed(b) if op=='add' else -signed(b)))) & 0xffffffff for a,b in pairs]
    for guarded in (False,True):
        assert list(execute(kernel_source(op+'.sat.s32',guarded=guarded),level,pairs))==expected
    text=kernel_source(op+'.sat.s32').replace(
        f'{op}.sat.s32 %v,%a,%b;',f'mov.u32 %v,99; setp.eq.u32 %p,%a,0; @%p {op}.sat.s32 %v,%a,%b;')
    assert list(execute(text,level,pairs))==[value if a==0 else 99 for (a,b),value in zip(pairs,expected)]
    text=kernel_source(op+'.sat.s32').replace(f'{op}.sat.s32 %v,%a,%b;',
                                            f'{op}.sat.s32 %a,%a,%b; mov.b32 %v,%a;')
    assert list(execute(text,level,pairs))==expected


@pytest.mark.parametrize('level',['safe','mined','sched','opt'])
@pytest.mark.parametrize('ftz',[False,True])
def test_float_predicate_subnormal_zero_normal_inf_nan(level,ftz):
    bits=[0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,
          0x3f800000,0xbf800000,0x7f800000,0xff800000,0x7fc00001]
    pairs=[(a,b) for a in bits for b in bits]
    def number(x):
        if ftz and x & 0x7f800000 == 0:x &= 0x80000000
        return struct.unpack('<f',struct.pack('<I',x))[0]
    expected=[int(number(a)==number(b)) for a,b in pairs]
    assert list(execute(kernel_source('setp.eq.'+('ftz.' if ftz else '')+'f32'),level,pairs))==expected


@pytest.mark.parametrize('instruction',[
    'add.cc.u32 %v,%a,%b;', 'add.sat.u32 %v,%a,%b;', 'sub.cc.u64 %v,%a,%b;',
    'mad.hi.s32 %v,%a,%b,%a;', 'add.rz.f32 %v,%a,%b;', 'add.fiction.u32 %v,%a,%b;',
    'setp.eq.ftz.u32 %p,%a,%b;', 'setp.eq.and.f32 %p,%a,%b,%p;',
    'cvta.to.shared.u64 %d,%d;', 'ld.shared.volatile.u32 %v,[%a];',
    'ld.global.v4.u64 {%a,%b,%v,%a},[%d];', 'ld.global.v4.u32 {%a,%b},[%d];',
    'add.u32 %v,%a,[%d];',
])
def test_unimplemented_modifier_type_width_or_operand_rejects_with_line(instruction):
    text=kernel_source('add.u32').replace('add.u32 %v,%a,%b;',instruction)
    kernel=parse_module(text).kernels['probe']
    with pytest.raises(ra.NotSupported,match='PTX line .*contract|PTX line .*requires|PTX line .*scalar'):
        ra.compile_kernel(kernel)


def test_parsed_instruction_lines_are_source_lines():
    text=HEADER+'.entry k() {\n.reg .u32 %r;\n\nL: add.cc.u32 %r,%r,1;\nret;\n}'
    kernel=parse_module(text).kernels['k']
    assert [(i.op,i.line) for i in kernel.body]==[('add',7),('ret',8)]
