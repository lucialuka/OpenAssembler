"""Independent generated expressions and fixed exceptional/precedence controls."""
from fractions import Fraction
from pathlib import Path
import random
import struct
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'float_literals'))
from rounding_probe import rounded_bits,rational
from reference import generated

HEADER='.version 8.7\n.target sm_80\n.address_size 64\n'
SENTINEL=0x13579bdf


def source(name,slots,body):
    return HEADER+f'''.visible .entry {name}(.param .u64 output,.param .u64 input,.param .u32 n){{
.reg .u64 %out,%input,%offset,%src,%dst;
.reg .b64 %wide;
.reg .u32 %t,%ct,%i,%n,%iter,%tmp;
.reg .f32 %a,%v;
.reg .pred %end,%even,%again;
ld.param.u64 %out,[output];ld.param.u64 %input,[input];ld.param.u32 %n,[n];
mov.u32 %t,%tid.x;mov.u32 %ct,%ctaid.x;mad.lo.u32 %i,%ct,(64*2),%t;
setp.ge.u32 %end,%i,%n;@%end bra END;
mul.wide.u32 %offset,%i,(2*2);add.u64 %src,%input,%offset;
mul.wide.u32 %offset,%i,({slots}*8);add.u64 %dst,%out,%offset;
ld.global.f32 %a,[%src];and.b32 %tmp,%i,(1<<0);setp.eq.u32 %even,%tmp,(1-1);
{body}
END:ret;
}}
'''


def cases():
    corpus=list(generated(count=128))
    for kind in('s','u','f'):
        group=[row for row in corpus if row['kind']==kind]
        name='expressions_'+kind
        body=[f'mov.b64 %wide,{row["expression"]};st.global.b64 [%dst+{i*8}],%wide;'for i,row in enumerate(group)]
        yield dict(name=name,kind='constants64',width=8,slots=len(group),expressions=[row['expression']for row in group],bits=[row['bits']for row in group],source=source(name,len(group),'\n'.join(body)))
    fixed=[('((1.0e20+1.0)-1.0e20)',0),('1.0/3.0',0x3fd5555555555555),
        ('(-0.0)+(-0.0)',0x8000000000000000),('1.0e308*2.0',0x7ff0000000000000),
        ('1.0e-307/1.0e308',0),('0d7ff8000000000000',0x7ff8000000000000),
        ('0d7ff0000000000001+1.0',0x7ff8000000000001),('0dfff8000000000001+1.0',0xfff8000000000001),
        ('1<<64',1),('1<<-1',0x8000000000000000),('(-2&-1)>0',0),
        ('((.u64)-1)>>63',1),('-8 % 3',2),('8 % -3',8),('0xffffffffffffffff+1',0),
        ('18446744073709551616+1',1),('(1.5<2.0)?3:4',3),('0?3:4',4),
        ('(0123+1)',84),('(0b1010+1)',11)]
    body=[f'mov.b64 %wide,({text});st.global.b64 [%dst+{i*8}],%wide;'for i,(text,_)in enumerate(fixed)]
    yield dict(name='expressions_edges',kind='constants64',width=8,slots=len(fixed),expressions=[text for text,_ in fixed],bits=[bits for _,bits in fixed],source=source('expressions_edges',len(fixed),'\n'.join(body)))
    floats=[row for row in corpus if row['kind']=='f'][:48]
    body=[f'mov.f32 %v,{row["expression"]};st.global.f32 [%dst+{i*8}],%v;'for i,row in enumerate(floats)]
    yield dict(name='expressions_narrow',kind='constants32',width=4,slots=len(floats),expressions=[row['expression']for row in floats],bits=[rounded_bits(rational(row['bits'],64),32)for row in floats],source=source('expressions_narrow',len(floats),'\n'.join(body)))
    body='''mov.f32 %v,%a;@%even mov.f32 %v,(-(1.5+1.0));st.global.f32 [%dst],%v;
mov.u32 %iter,(3-3);
LOOP:add.rn.f32 %v,%v,(1.0/2.0);add.u32 %iter,%iter,(8/4-1);
setp.lt.u32 %again,%iter,(1+2);@%again bra LOOP;
st.global.f32 [%dst+8],%v;
@!%even st.global.b64 [%dst+16],((1.0e20+1.0)-1.0e20);
mov.b64 %wide,{((1<<4)|3),((.u64)-1)>>32};st.global.b64 [%dst+24],%wide;'''
    yield dict(name='expressions_guards',kind='guards',width=8,slots=4,source=source('expressions_guards',4,body))


def values(seed,lanes):
    rng=random.Random(seed)
    return[struct.unpack('<I',struct.pack('<f',(-1 if rng.randrange(2)else 1)*2.0**rng.randrange(-4,5)))[0]for _ in range(lanes)]


def expected(data,case,n,lanes):
    output=[SENTINEL]*(lanes*case['slots']*2)
    for lane in range(n):
        base=lane*case['slots']*2
        if case['kind'].startswith('constants'):
            for i,bits in enumerate(case['bits']):
                output[base+2*i]=bits&0xffffffff
                if case['width']==8:output[base+2*i+1]=bits>>32
        else:
            value=Fraction(-5,2)if lane%2==0 else rational(data[lane],32)
            output[base]=rounded_bits(value,32)
            output[base+2]=rounded_bits(value+Fraction(3,2),32)
            if lane%2:output[base+4]=output[base+5]=0
            output[base+6]=19;output[base+7]=0xffffffff
    return output
