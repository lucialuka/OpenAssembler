"""Per-thread local arrays, symbols/pointers, predicates, loops and layout holes."""
import random
import struct

HEADER='.version 8.7\n.target sm_80\n.address_size 64\n'
N=259
THREADS=128
GRID=(3,1,1)
LANES=384
SENTINEL=0x13579bdf


def cases():
    for dtype in('b32','u32','s32','f32'):
        for addressing in('symbol','pointer'):
            name=f'local_{dtype}_{addressing}'
            lines=[HEADER+f'''.visible .entry {name}(.param .u64 output,.param .u64 input,.param .u32 n) {{
.local .align 4 .b8 prefix[4]; .local .align 16 .b8 slots[32]; .local .align 8 .b8 suffix[8];
.reg .u64 %out,%input,%offset,%src,%dst;
.reg .b32 %t,%ct,%i,%n,%base,%address,%slot,%value,%initial,%loaded,%result,%iter;
.reg .pred %end,%even,%loop;
ld.param.u64 %out,[output];ld.param.u64 %input,[input];ld.param.u32 %n,[n];
mov.u32 %t,%tid.x;mov.u32 %ct,%ctaid.x;mad.lo.u32 %i,%ct,128,%t;
setp.ge.u32 %end,%i,%n;@%end bra DONE;
mul.wide.u32 %offset,%i,4;add.u64 %src,%input,%offset;ld.global.b32 %initial,[%src];
st.local.b32 [prefix],%initial;st.local.b32 [suffix],%initial;st.local.b32 [suffix+4],%initial;
mov.u32 %base,slots;mov.u32 %iter,0;
LOOP:
''']
            for slot in range(8):
                lines.append(f'add.u32 %value,%initial,{slot};xor.b32 %value,%value,%iter;')
                if addressing=='symbol':addr=f'[slots+{slot*4}]'
                else:
                    lines.append(f'add.u32 %address,%base,{slot*4};');addr='[%address]'
                lines.append(f'st.local.{dtype} {addr},%value;')
            lines+=['add.u32 %iter,%iter,1;setp.lt.u32 %loop,%iter,3;@%loop bra LOOP;',
                    'and.b32 %slot,%i,7;shl.b32 %slot,%slot,2;add.u32 %address,%base,%slot;',
                    f'ld.local.{dtype} %result,[%address];',
                    'and.b32 %slot,%i,1;setp.eq.u32 %even,%slot,0;',
                    f'@%even st.local.{dtype} [slots],%initial;',
                    'mov.b32 %loaded,0x89abcdef;',f'@%even ld.local.{dtype} %loaded,[slots];',
                    'xor.b32 %result,%result,%loaded;',
                    'ld.local.b32 %value,[prefix];xor.b32 %result,%result,%value;',
                    'ld.local.b32 %value,[suffix+4];xor.b32 %result,%result,%value;',
                    'mul.wide.u32 %offset,%i,4;add.u64 %dst,%out,%offset;st.global.b32 [%dst],%result;',
                    'DONE: ret; }']
            yield dict(name=name,dtype=dtype,addressing=addressing,source='\n'.join(lines)+'\n')


def values(seed):
    rng=random.Random(seed)
    return [rng.getrandbits(32)for _ in range(LANES)]


def expected(input_values,n=N):
    return [(((input_values[i]+i%8)&0xffffffff)^2)^(input_values[i]if i%2==0 else 0x89abcdef)
            if i<n else SENTINEL for i in range(LANES)]
