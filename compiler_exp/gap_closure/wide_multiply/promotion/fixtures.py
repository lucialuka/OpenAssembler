"""Independent arbitrary-precision references for next low64 product cycle."""
import random,struct
SENTINEL=0x13579bdf2468ace0
CHOICE=0x8123456776543210
MASK=(1<<64)-1

def cases():
 for dtype in('u64','s64'):
  for mode in('register','immediate','aliased','aliased_right','false_guard','negated_guard','parameter','left_immediate','both_immediate'):
   left,right,dest='%a','%b','%r';initial='mov.b64 %r,0xdeadbeef13579bdf;'
   if mode in('immediate','both_immediate'):right='-17'
   if mode in('left_immediate','both_immediate'):left='0x80000000ffffffff'
   if mode=='parameter':left='%choice'
   if mode in('aliased','aliased_right'):dest='%a'if mode=='aliased'else'%b';initial=''
   guard='@%writer 'if mode=='false_guard'else'@!%writer 'if mode=='negated_guard'else''
   mutations=''.join(f'mov.b64 {reg},0;'for reg in('%a','%b','%choice')if reg!=dest)
   source=f'''.version 8.7
.target sm_80
.address_size 64
.entry probe(.param .u64 out,.param .u64 input,.param .u32 n,.param .b64 choice){{
.reg .u64 %out,%in,%off,%src,%dst;.reg .b64 %a,%b,%r,%choice;
.reg .u32 %i,%n,%lo,%block;.reg .pred %writer,%active;
ld.param.u64 %out,[out];ld.param.u64 %in,[input];ld.param.u32 %n,[n];ld.param.b64 %choice,[choice];
mov.u32 %i,%tid.x;mov.u32 %block,%ctaid.x;mad.lo.u32 %i,%block,128,%i;
mul.wide.u32 %off,%i,16;add.u64 %src,%in,%off;ld.global.b64 %a,[%src];ld.global.b64 %b,[%src+8];
and.b32 %lo,%i,2;setp.ne.u32 %writer,%lo,0;
{initial}{guard}mul.lo.{dtype} {dest},{left},{right};{mutations}
mul.wide.u32 %off,%i,8;add.u64 %dst,%out,%off;setp.lt.u32 %active,%i,%n;
@%active st.global.b64 [%dst],{dest};ret;}}'''
   yield dict(name=dtype+'_'+mode,dtype=dtype,mode=mode,entry='probe',source=source)

def values(seed,lanes=128):
 rng=random.Random(seed);edges=[0,MASK,1<<63,1,0xffffffff,0x100000000,0x7fffffffffffffff,0xffffffff00000000,0x80000000ffffffff,0x7fffffff80000000]
 return[edges[i]if i<len(edges)else rng.getrandbits(64)for i in range(lanes*2)]

def expected(data,case,n,lanes=128):
 out=[SENTINEL]*lanes
 for i in range(n):
  left,right=data[2*i:2*i+2];mode=case['mode']
  if mode in('immediate','both_immediate'):right=-17
  if mode in('left_immediate','both_immediate'):left=0x80000000ffffffff
  if mode=='parameter':left=CHOICE
  write=bool(i&2)
  if mode=='negated_guard':write=not write
  out[i]=0xdeadbeef13579bdf if mode in('false_guard','negated_guard')and not write else(left*right)&MASK
 return struct.pack('<'+str(lanes)+'Q',*out)
