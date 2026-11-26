"""Frozen-pass controls with new limits/counts and exact finite FP32 FMA."""
import random
import struct
from probes import HEADER,SENTINEL,source,expected as integer_expected


def cases():
    for name,count,limit,width,predicates in(
        ('heldout_scalar64',64,16,1,False),
        ('heldout_quad160',160,64,4,False),
        ('heldout_pred40',40,32,1,True),
    ):
        yield dict(name=name,count=count,limit=limit,width=width,predicates=predicates,local=False,loop=False,output_words=1,
                   domain='heldout_parameters',source=source(name,count,limit,width,predicates))
    count=96;name='heldout_float96'
    lines=[HEADER+f'''.visible .entry {name}(.param .u64 output,.param .u64 input,.param .u32 n)
.maxnreg 32
{{
.reg .u64 %out,%input,%offset,%src,%dst;
.reg .u32 %t,%ct,%i,%n;
.reg .f32 %sum,%r<{count}>;
.reg .pred %end;
ld.param.u64 %out,[output];ld.param.u64 %input,[input];ld.param.u32 %n,[n];
mov.u32 %t,%tid.x;mov.u32 %ct,%ctaid.x;mad.lo.u32 %i,%ct,128,%t;
setp.ge.u32 %end,%i,%n;@%end bra END;
mul.wide.u32 %offset,%i,{count*4};add.u64 %src,%input,%offset;
''']
    for i in range(count):lines.append(f'ld.global.f32 %r{i},[%src+{4*i}];')
    lines.append('mov.f32 %sum,0.0;')
    for i in range(count):lines.append(f'fma.rn.f32 %sum,%r{i},1.5,%sum;')
    lines+=['mul.wide.u32 %offset,%i,4;add.u64 %dst,%out,%offset;st.global.f32 [%dst],%sum;','END:ret;}']
    yield dict(name=name,count=count,limit=32,width=1,predicates=False,local=False,loop=False,output_words=1,
               domain='heldout_exact_f32',dtype='f32',source='\n'.join(lines)+'\n')


def values(seed,case,lanes):
    rng=random.Random(seed)
    if case.get('dtype')=='f32':
        # Integer inputs and half-integer intermediates are exactly representable
        # throughout this range. The reference needs no implementation FP model.
        return[struct.unpack('<I',struct.pack('<f',rng.randrange(-1024,1025)))[0]for _ in range(lanes*case['count'])]
    return[rng.getrandbits(32)for _ in range(lanes*case['count'])]


def expected(data,case,n,lanes):
    if case.get('dtype')!='f32':return integer_expected(data,case,n,lanes)
    result=[SENTINEL]*lanes
    for lane in range(n):
        total=sum(int(struct.unpack('<f',struct.pack('<I',word))[0])for word in data[lane*case['count']:(lane+1)*case['count']])
        result[lane]=struct.unpack('<I',struct.pack('<f',total*1.5))[0]
    return result
