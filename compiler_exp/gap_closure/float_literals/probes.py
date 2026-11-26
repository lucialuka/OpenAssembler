"""Independent literal bits, exact arithmetic and guarded-loop hardware sources."""
from fractions import Fraction
import random
import struct
from rounding_probe import rounded_bits,rational

HEADER='.version 8.7\n.target sm_80\n.address_size 64\n'
SENTINEL=0x13579bdf


def bits64(text):
    if text.lower().startswith('0d'):return int(text[2:],16)
    if text.lower().startswith('0f'):return int(text[2:],16)
    return rounded_bits(Fraction(text),64,text.startswith('-'))


def bits32(text):
    if text.lower().startswith('0f'):return int(text[2:],16)
    bits=bits64(text);exponent=bits>>52&2047;fraction=bits&((1<<52)-1);sign=(bits>>63)<<31
    if exponent==2047:return sign|0x7f800000|((fraction>>29)|0x400000 if fraction else 0)
    return rounded_bits(rational(bits,64),32,bool(bits>>63))


def literals():
    values=['0.0','-0.0','+1.5','.5','2.','1e-3','1E+3','0e9999','-0e-9999','1.0000000596046448',
            '1.0000001788139343','3.4028235677973366e38','1.401298464324817e-45','7.006492321624085e-46',
            '2.2250738585072014e-308','1.7976931348623157e308']
    rng=random.Random(8127)
    values.extend(f'{rng.randrange(-10**17,10**17)}.{rng.randrange(10**12):012d}e{rng.randrange(-300,289):+d}'for _ in range(64))
    values.extend('0d'+bits for bits in('7ff0000000000000','fff0000000000000','7ff8000000000000','fff8000000000000',
        '7ff0000000000001','7ffabcdef1234567','fffabcdef1234567','0000000000000001','8000000000000000'))
    values.extend('0f'+bits for bits in('00000000','80000000','00000001','7fc12345','7f800001','ffcabcde'))
    return values


def source(name,slots,body):
    return HEADER+f'''.visible .entry {name}(.param .u64 output,.param .u64 input,.param .u32 n){{
.reg .u64 %out,%input,%offset,%src,%dst;
.reg .b64 %wide;
.reg .u32 %t,%ct,%i,%n,%tmp,%iter;
.reg .f32 %a,%b,%v,%sum;
.reg .pred %end,%even,%p,%again;
ld.param.u64 %out,[output];ld.param.u64 %input,[input];ld.param.u32 %n,[n];
mov.u32 %t,%tid.x;mov.u32 %ct,%ctaid.x;mad.lo.u32 %i,%ct,128,%t;
setp.ge.u32 %end,%i,%n;@%end bra END;
mul.wide.u32 %offset,%i,4;add.u64 %src,%input,%offset;
mul.wide.u32 %offset,%i,{slots*8};add.u64 %dst,%out,%offset;
ld.global.f32 %a,[%src];and.b32 %tmp,%i,1;setp.eq.u32 %even,%tmp,0;
{body}
END:ret;
}}
'''


def cases():
    values=literals();body=[]
    for i,text in enumerate(values):body.append(f'mov.f32 %v,{text};st.global.f32 [%dst+{i*8}],%v;')
    yield dict(name='decimal32',kind='constants32',width=4,slots=len(values),literals=values,source=source('decimal32',len(values),'\n'.join(body)))
    wide=values[:16]+['0d7ff8000000000001','0dfff0000000000000','0d0000000000000001','0d8000000000000000',
                    '0f3fc00000','0f7fc12345','0f80000000','0f7f800001']
    types=['f64'if i%2==0 or text.startswith('0f')else'b64'for i,text in enumerate(wide)]
    body=[f'mov.{dtype} %wide,{text};st.global.b64 [%dst+{i*8}],%wide;'for i,(text,dtype)in enumerate(zip(wide,types))]
    yield dict(name='decimal64',kind='constants64',width=8,slots=len(wide),literals=wide,source=source('decimal64',len(wide),'\n'.join(body)))
    stores=[('f64','1.5'),('b64','-2.5'),('b64','0d7ff8000000000001'),('f64','0f7fc12345'),('f64','0f80000000')]
    body=[f'st.global.{dtype} [%dst+{i*8}],{text};'for i,(dtype,text)in enumerate(stores)]
    yield dict(name='decimal_stores64',kind='constants64',width=8,slots=len(stores),literals=[text for _,text in stores],source=source('decimal_stores64',len(stores),'\n'.join(body)))
    operations=[('add','add.rn.f32 %v,%a,1.5;'),('sub','sub.rn.f32 %v,1.5,%a;'),('mul','mul.rn.f32 %v,%a,.5;'),
                ('fma','fma.rn.f32 %v,1.5,%a,-.5;'),('mad','mad.rn.f32 %v,%a,1.5,-.5;'),
                ('min','min.f32 %v,%a,1.5;'),('max','max.f32 %v,1.5,%a;'),
                ('neg','neg.f32 %v,1.5;'),('abs','abs.f32 %v,-1.5;'),
                ('select_max','setp.lt.f32 %p,%a,1.5;selp.f32 %v,1.5,%a,%p;'),
                ('select_min','setp.lt.f32 %p,1.5,%a;selp.f32 %v,1.5,%a,%p;'),
                ('div','div.rn.f32 %v,%a,2.0;'),('reciprocal','rcp.rn.f32 %v,2.0;'),
                ('sqrt','sqrt.rn.f32 %v,4.0;'),('store','st.global.f32 [%dst+112],1.5;')]
    body=[]
    for i,(kind,text)in enumerate(operations):
        body.append(text+(f'st.global.f32 [%dst+{i*8}],%v;'if kind!='store'else''))
    yield dict(name='decimal_math',kind='math',width=4,slots=len(operations),operations=[kind for kind,_ in operations],source=source('decimal_math',len(operations),'\n'.join(body)))
    body='''mov.f32 %v,%a;@%even mov.f32 %v,-2.5;st.global.f32 [%dst],%v;
mov.f32 %sum,%a;mov.u32 %iter,0;
LOOP:add.rn.f32 %sum,%sum,.5;add.u32 %iter,%iter,1;setp.lt.u32 %again,%iter,3;@%again bra LOOP;
st.global.f32 [%dst+8],%sum;
@!%even st.global.f32 [%dst+16],1.0000000596046448;'''
    yield dict(name='decimal_guard_loop',kind='guards',width=4,slots=3,source=source('decimal_guard_loop',3,body))


def values(seed,lanes):
    rng=random.Random(seed)
    return[struct.unpack('<I',struct.pack('<f',(-1 if rng.randrange(2)else 1)*2.0**rng.randrange(-4,5)))[0]for _ in range(lanes)]


def expected(data,case,n,lanes):
    output=[SENTINEL]*(lanes*case['slots']*2)
    for lane in range(n):
        base=lane*case['slots']*2
        if case['kind'].startswith('constants'):
            for i,text in enumerate(case['literals']):
                bits=bits32(text)if case['width']==4 else bits64(text)
                output[base+2*i]=bits&0xffffffff
                if case['width']==8:output[base+2*i+1]=bits>>32
        else:
            a=rational(data[lane],32)
            if case['kind']=='math':
                ref={'add':a+Fraction(3,2),'sub':Fraction(3,2)-a,'mul':a/2,'fma':a*Fraction(3,2)-Fraction(1,2),
                     'mad':a*Fraction(3,2)-Fraction(1,2),'min':min(a,Fraction(3,2)),'max':max(a,Fraction(3,2)),
                     'neg':Fraction(-3,2),'abs':Fraction(3,2),'select_max':max(a,Fraction(3,2)),
                     'select_min':min(a,Fraction(3,2)),'div':a/2,'reciprocal':Fraction(1,2),'sqrt':Fraction(2),'store':Fraction(3,2)}
                for i,kind in enumerate(case['operations']):output[base+2*i]=rounded_bits(ref[kind],32)
            else:
                output[base]=rounded_bits(Fraction(-5,2),32)if lane%2==0 else data[lane]
                output[base+2]=rounded_bits(a+Fraction(3,2),32)
                if lane%2:output[base+4]=bits32('1.0000000596046448')
    return output
