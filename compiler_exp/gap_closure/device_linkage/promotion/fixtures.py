"""Source-library resolution hypotheses with independent complete word references."""
import random,struct
HEADER='.version 8.7\n.target sm_80\n.address_size 64\n'
SENTINEL=0x13579bdf2468ace0


def function(name='helper',body='mad.lo.u32 %v,%x,17,9;',linkage='visible',dtype='b32'):
    return('.'+linkage+' 'if linkage else'')+f'''.func (.param .{dtype} result) {name}(.param .{dtype} input){{
.reg .b32 %x,%v;ld.param.b32 %x,[input];{body}st.param.b32 [result],%v;ret;
}}
'''


def root():
    return HEADER+'''.extern .func (.param .b32 result) helper(.param .b32 input);
.visible .entry probe(.param .u64 out,.param .u64 input,.param .u32 n){
.reg .u64 %out,%in,%off,%src,%dst,%wide,%result;.reg .u32 %t,%c,%i,%n,%lo,%hi;.reg .pred %active;
ld.param.u64 %out,[out];ld.param.u64 %in,[input];ld.param.u32 %n,[n];mov.u32 %t,%tid.x;mov.u32 %c,%ctaid.x;mad.lo.u32 %i,%c,128,%t;
mul.wide.u32 %off,%i,8;add.u64 %src,%in,%off;add.u64 %dst,%out,%off;ld.global.b64 %wide,[%src];mov.b64 {%lo,%hi},%wide;
{.param .b32 a;.param .b32 r;st.param.b32 [a],%lo;call.uni (r),helper,(a);ld.param.b32 %lo,[r];}
mov.b64 %result,{%lo,%hi};setp.lt.u32 %active,%i,%n;@%active st.global.b64 [%dst],%result;ret;}
'''


def cases():
    yield dict(name='direct',root=root(),libraries=[HEADER+function()])
    local=function(body=""".local .align 16 .b8 scratch[32];.reg .u32 %addr;
mov.u32 %addr,scratch;st.local.u32 [%addr],%x;add.u32 %v,%x,7;st.local.u32 [%addr+4],%v;
and.b32 %x,%x,1;shl.b32 %x,%x,2;add.u32 %addr,%addr,%x;ld.local.u32 %v,[%addr];""")
    yield dict(name='local_frame',root=root(),libraries=[HEADER+local])
    wide_root=root().replace('(.param .b32 result) helper(.param .b32 input)','(.param .b64 result) helper(.param .b64 input)').replace('.param .b32 a;.param .b32 r;st.param.b32 [a],%lo;call.uni (r),helper,(a);ld.param.b32 %lo,[r];','.param .b64 a;.param .b64 r;st.param.b64 [a],%wide;call.uni (r),helper,(a);ld.param.b64 %result,[r];').replace('mov.b64 %result,{%lo,%hi};','')
    wide='.visible .func (.param .b64 result) helper(.param .b64 input){.reg .b64 %x,%v;ld.param.u64 %x,[input];add.u64 %v,%x,0x123456789abcdef;st.param.b64 [result],%v;ret;}'
    yield dict(name='wide_slot',root=wide_root,libraries=[HEADER+wide])

    first=HEADER+'''.extern .func (.param .b32 r) tail(.param .b32 a);
'''+function(body='''.param .b32 a;.param .b32 r;st.param.b32 [a],%x;call.uni (r),tail,(a);ld.param.b32 %v,[r];mad.lo.u32 %v,%v,17,9;''')
    yield dict(name='transitive',root=root(),libraries=[first,HEADER+function('tail','mad.lo.u32 %v,%x,3,5;')])
    first=HEADER+'''.extern .func (.param .b32 r) tail(.param .b32 a);
'''+function(body='''.param .b32 a;.param .b32 r;st.param.b32 [a],%x;call.uni (r),internal,(a);ld.param.b32 %v,[r];st.param.b32 [a],%v;call.uni (r),tail,(a);ld.param.b32 %v,[r];''')+function('internal','add.u32 %v,%x,3;',linkage='')
    second=HEADER+function('tail','''.param .b32 a;.param .b32 r;st.param.b32 [a],%x;call.uni (r),internal,(a);ld.param.b32 %v,[r];''')+function('internal','add.u32 %v,%x,5;',linkage='')
    prototype='.func (.param .b32 result) internal(.param .b32 input);\n'
    yield dict(name='private_names',root=root(),libraries=[first.replace(HEADER,HEADER+prototype,1),second.replace(HEADER,HEADER+prototype,1)])
    yield dict(name='weak_strong',root=root(),libraries=[HEADER+function(body='add.u32 %v,%x,1;',linkage='weak'),HEADER+function(body='add.u32 %v,%x,9;')])
    weak=HEADER+function(body='add.u32 %v,%x,7;',linkage='weak')
    yield dict(name='identical_weak',root=root(),libraries=[weak,weak])
    yield dict(name='debug_files',root=root().replace('.extern .func','.file 1 "root.cu"\n.extern .func'),libraries=[(HEADER+'.file 1 "library.cu"\n'+function(body='.loc 1 1 1\nmad.lo.u32 %v,%x,17,9;'))])
    yield dict(name='version_mix',root=root().replace('.version 8.7','.version 8.0'),libraries=[HEADER+function()])
    collision=HEADER+function('unused','mov.b32 %v,%x;',linkage='')+function(body='add.u32 %v,%x,11;')
    yield dict(name='unused_private',root=root(),libraries=[collision])


def values(seed,lanes):
    rng=random.Random(seed);edge=[0,1,(1<<64)-1,1<<63,(1<<63)-1,0xffffffff,1<<32,0x123456789abcdef0]
    return[edge[i]if i<len(edge)else rng.getrandbits(64)for i in range(lanes)]


def expected(data,case,n):
    result=[SENTINEL]*len(data)
    for i,value in enumerate(data[:n]):
        lo=value&0xffffffff;mode=case['name']
        if mode=='wide_slot':
            result[i]=(value+0x123456789abcdef)&((1<<64)-1);continue
        if mode=='local_frame':lo=(lo+(7 if lo&1 else 0))&0xffffffff
        elif mode=='transitive':lo=(((lo*3+5)&0xffffffff)*17+9)&0xffffffff
        elif mode=='private_names':lo=(lo+8)&0xffffffff
        elif mode=='weak_strong':lo=(lo+9)&0xffffffff
        elif mode=='identical_weak':lo=(lo+7)&0xffffffff
        elif mode=='unused_private':lo=(lo+11)&0xffffffff
        else:lo=(lo*17+9)&0xffffffff
        result[i]=(value&0xffffffff00000000)|lo
    return struct.pack('<'+str(len(result))+'Q',*result)
