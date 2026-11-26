"""Direct scalar function sources and independent exact byte references."""
from pathlib import Path
import random,struct
HERE=Path(__file__).resolve().parent
HEADER='.version 8.7\n.target sm_80\n.address_size 64\n'
MASK=(1<<64)-1
SENTINEL=0x13579bdf2468ace0


def cases():
    source=(HERE/'functions.ptx').read_text()
    yield dict(name='nested_wide',source=source,entry='probe',mode='wide')
    for mode in('repeated','guarded','local','loop','void','float','multi_return'):
        if mode=='repeated':
            function='''.func (.param .b32 result) helper(.param .b32 x){.reg .u32 %x,%v;ld.param.u32 %x,[x];mad.lo.u32 %v,%x,17,9;st.param.b32 [result],%v;ret;}'''
            body='''{.param .b32 a;.param .b32 r;st.param.b32 [a],%lo;call.uni (r),helper,(a);ld.param.u32 %lo,[r];}
{.param .b32 a;.param .b32 r;st.param.b32 [a],%lo;call.uni (r),helper,(a);ld.param.u32 %hi,[r];}
mov.b64 %result,{%lo,%hi};'''
        elif mode=='guarded':
            function='''.func (.param .b64 result) helper(.param .b64 x){.reg .b64 %x,%v;ld.param.u64 %x,[x];add.u64 %v,%x,0x123456789abcdef;st.param.b64 [result],%v;ret;}'''
            body='''{.param .b64 a;.param .b64 r;st.param.b64 [a],%wide;st.param.b64 [r],%wide;
and.b32 %lo,%lo,1;setp.ne.u32 %take,%lo,0;@%take call (r),helper,(a);ld.param.b64 %result,[r];}'''
        elif mode=='local':
            function='''.func (.param .b32 result) helper(.param .b32 x){.local .align 16 .b8 scratch[32];.reg .u32 %x,%v,%a;
ld.param.u32 %x,[x];mov.u32 %a,scratch;st.local.u32 [%a],%x;add.u32 %v,%x,7;st.local.u32 [%a+4],%v;
and.b32 %x,%x,1;shl.b32 %x,%x,2;add.u32 %a,%a,%x;ld.local.u32 %v,[%a];st.param.b32 [result],%v;ret;}'''
            body='''{.param .b32 a;.param .b32 r;st.param.b32 [a],%lo;call.uni (r),helper,(a);ld.param.u32 %lo,[r];}
{.param .b32 a;.param .b32 r;st.param.b32 [a],%lo;call.uni (r),helper,(a);ld.param.u32 %hi,[r];}
mov.b64 %result,{%lo,%hi};'''
        elif mode=='loop':
            function='''.func (.param .b32 result) helper(.param .b32 x){.reg .u32 %x,%v,%j;.reg .pred %p;
ld.param.u32 %x,[x];mov.u32 %v,0;mov.u32 %j,0;LOOP:add.u32 %v,%v,%x;add.u32 %j,%j,1;setp.lt.u32 %p,%j,3;@%p bra LOOP;st.param.b32 [result],%v;ret;}'''
            body='''{.param .b32 a;.param .b32 r;st.param.b32 [a],%lo;call.uni (r),helper,(a);ld.param.u32 %lo,[r];}mov.b64 %result,{%lo,%hi};'''
        elif mode=='void':
            function='''.func helper(.param .b64 p,.param .b64 v){.reg .u64 %p,%v;ld.param.u64 %p,[p];ld.param.u64 %v,[v];st.global.u64 [%p],%v;ret;}'''
            body='''{.param .b64 a;.param .b64 b;st.param.b64 [a],%dst;st.param.b64 [b],%wide;@%active call.uni helper,(a,b);}mov.b64 %result,%wide;'''
        elif mode=='float':
            function='''.func (.param .b32 result) helper(.param .b32 x){.reg .f32 %x,%v;ld.param.f32 %x,[x];add.rn.f32 %v,%x,1.0;mul.rn.f32 %v,%v,2.0;st.param.f32 [result],%v;ret;}'''
            body='''{.param .b32 a;.param .b32 r;st.param.b32 [a],%lo;call.uni (r),helper,(a);ld.param.b32 %lo,[r];}mov.b64 %result,{%lo,%hi};'''
        else:
            function='''.func (.param .b32 a,.param .b32 b) helper(.param .b32 x){.reg .u32 %x,%y;ld.param.u32 %x,[x];add.u32 %y,%x,7;st.param.b32 [a],%x;st.param.b32 [b],%y;ret;}'''
            body='''{.param .b32 a;.param .b32 b;.param .b32 x;st.param.b32 [x],%lo;call.uni (a,b),helper,(x);ld.param.b32 %lo,[a];ld.param.b32 %hi,[b];}mov.b64 %result,{%lo,%hi};'''
        # Multiple returns are tested first against the toolkit and remain a
        # rejected boundary if the actual PTX ABI only permits one return slot.
        text=HEADER+function+'''
.visible .entry probe(.param .u64 out,.param .u64 input,.param .u32 n){
.reg .u64 %out,%in,%off,%src,%dst,%wide,%result;.reg .u32 %t,%c,%i,%n,%lo,%hi;.reg .pred %active,%take;
ld.param.u64 %out,[out];ld.param.u64 %in,[input];ld.param.u32 %n,[n];
mov.u32 %t,%tid.x;mov.u32 %c,%ctaid.x;mad.lo.u32 %i,%c,128,%t;setp.lt.u32 %active,%i,%n;
mul.wide.u32 %off,%i,8;add.u64 %src,%in,%off;add.u64 %dst,%out,%off;ld.global.b64 %wide,[%src];mov.b64 {%lo,%hi},%wide;
'''+body+'''@%active st.global.b64 [%dst],%result;ret;}'''
        yield dict(name=mode,source=text,entry='probe',mode=mode)


def values(seed,lanes,mode):
    rng=random.Random(seed);edge=[0,1,MASK,1<<63,(1<<63)-1,0xffffffff,1<<32,0x123456789abcdef0]
    if mode=='float':
        return[(rng.getrandbits(32)<<32)|struct.unpack('<I',struct.pack('<f',float((i%33)-16)/4))[0]for i in range(lanes)]
    return[edge[i]if i<len(edge)else rng.getrandbits(64)for i in range(lanes)]


def expected(data,case,n):
    out=[SENTINEL]*len(data)
    for i,x in enumerate(data[:n]):
        lo=x&0xffffffff;hi=x>>32;mode=case['mode']
        if mode=='wide':value=(x+0x123456789abcdef)&MASK if((lo+7)&1)else x^0xfedcba9876543210
        elif mode=='repeated':
            low=(lo*17+9)&0xffffffff;value=low|(((low*17+9)&0xffffffff)<<32)
        elif mode=='guarded':value=(x+0x123456789abcdef)&MASK if lo&1 else x
        elif mode=='local':
            low=(lo+(7 if lo&1 else 0))&0xffffffff;high=(low+(7 if low&1 else 0))&0xffffffff;value=low|(high<<32)
        elif mode=='loop':value=((lo*3)&0xffffffff)|(hi<<32)
        elif mode=='void':value=x
        elif mode=='float':
            f=struct.unpack('<f',struct.pack('<I',lo))[0];value=struct.unpack('<I',struct.pack('<f',(f+1)*2))[0]|(hi<<32)
        else:value=lo|(((lo+7)&0xffffffff)<<32)
        out[i]=value
    return struct.pack('<'+str(len(out))+'Q',*out)
