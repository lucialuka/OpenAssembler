"""Original typed/effectful chains and independently reconstructed complete references."""
import random,struct
MASK=(1<<64)-1;SENTINEL=0x13579bdf
HEADER='from typing import Final\nfrom openassembler.pyfront import *\n'
BODIES={
 'ordinary':('i32','    if -3<x<7:\n        value+=1\n'),
 'negated':('i32','    if not (-3<=x<=7):\n        value+=2\n'),
 'or_chain':('i32','    if -3<x<0 or 1<=x<=7:\n        value+=3\n'),
 'and_chain':('i32','    if -3<x<7 and 0<=x<3:\n        value+=4\n'),
 'while_chain':('i32','    step=0\n    while 0<=step<3 and -3<x<7:\n        value+=5\n        step+=1\n'),
 'mixed_ops':('i32','    if -3<=x!=3<7:\n        value+=6\n'),
 'length_four':('i32','    if -7<-3<x<3<7:\n        value+=7\n'),
 'context_bridge':('i64','    if 0<1<x:\n        value+=8\n'),
 'unsigned':('u32','    if u32(0)<=x<u32(2147483648):\n        value+=9\n'),
 'signed_wide':('i64','    if i64(-1099511627776)<x<i64(1099511627776):\n        value+=10\n'),
 'unsigned_wide':('u64','    if u64(0)<=x<u64(9223372036854775808):\n        value+=11\n'),
 'special_ordered':('f32','    if -2.0<x<3.0:\n        value+=12\n'),
 'special_ne':('f32','    if -2.0!=x!=3.0:\n        value+=13\n'),
 'special_negated':('f32','    if not (-2.0<=x<=3.0):\n        value+=14\n'),
 'nested_helper':('i64','    if i64(-3)<bump(x)<i64(7):\n        value+=15\n'),
 'return_helper':('i32','    if -3<x<7:\n        return value+16\n'),
 'atomic_middle':('i32','    if -1<atomic_add(count,i,1)<1:\n        value+=17\n'),
 'atomic_last':('i32','    if x<0<atomic_add(count,i,1) or 0<=x<3:\n        value+=18\n'),
 'atomic_multi':('i32','    if -1<atomic_add(count,i,1)<atomic_add(count,i,1)<2:\n        value+=19\n'),
 'load_snapshot':('i32','    if -1<atomic_add(count,i,1)<count[i]<2:\n        value+=20\n'),
}
NAMES=tuple(BODIES)
def configurations():
 constants=HEADER+'ZERO:Final=0\n'
 helpers=HEADER+'from . import constants\nraise RuntimeError("helper must never execute")\n@device\ndef bump(x:i64)->i64:\n    x+=i64(1)\n    return x\n'
 entry=HEADER+'from library import helpers\nfrom library import constants\nraise RuntimeError("entry must never execute")\n'
 single=constants+helpers[helpers.index('@device'):]
 for name,(dtype,body)in BODIES.items():
  if name.startswith('atomic_')or name=='load_snapshot':
   kernel=f'@kernel\ndef {name}(a:Array({dtype}),out:Array(i32),count:Array(i32),n:i32):\n    i=ctaid_x()*128+tid_x()\n    if i<n:\n        x=a[i]\n        value=11+i32(constants.ZERO)\n'+''.join('    '+line+'\n'for line in body.splitlines())+'        out[i]=value\n'
  else:
   fn=f'@device\ndef helper_{name}(x:{dtype})->i32:\n    value=11\n'+body+'    return value\n';helpers+=fn;single+=fn
   kernel=f'@kernel\ndef {name}(a:Array({dtype}),out:Array(i32),count:Array(i32),n:i32):\n    i=ctaid_x()*128+tid_x()\n    if i<n:\n        out[i]=helpers.helper_{name}(a[i])+i32(constants.ZERO)\n'
  entry+=kernel;single+=kernel.replace('helpers.','').replace('constants.','')
 base={'entry.py':entry,'library/__init__.py':'raise RuntimeError("package must never execute")\n','library/constants.py':constants,'library/helpers.py':helpers}
 yield dict(name='imported',files=base,entry='entry.py',explicit=False,remote=False)
 yield dict(name='same_file',files={'entry.py':single},entry='entry.py',explicit=False,remote=False)
 packaged={k:v for k,v in base.items()if k!='entry.py'};packaged.update({'kernels/__init__.py':'raise RuntimeError("entry package must never execute")\n','kernels/entry.py':entry})
 yield dict(name='remote_package',files=packaged,entry='kernels/entry.py',explicit=False,remote=True)
def write_tree(root,case):
 for name,text in case['files'].items():
  path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
 return root/case['entry']
def signed(bits,width):return bits-(1<<width)if bits&(1<<(width-1))else bits
def reference(name,x):
 value=11;count=0
 if name=='ordinary':yes=-3<x and x<7;gain=1
 elif name=='negated':yes=not(-3<=x and x<=7);gain=2
 elif name=='or_chain':yes=(-3<x and x<0)or(1<=x and x<=7);gain=3
 elif name=='and_chain':yes=(-3<x and x<7)and(0<=x and x<3);gain=4
 elif name=='while_chain':yes=-3<x and x<7;gain=15
 elif name=='mixed_ops':yes=-3<=x and x!=3 and 3<7;gain=6
 elif name=='length_four':yes=-7<-3 and-3<x and x<3 and 3<7;gain=7
 elif name=='context_bridge':yes=0<1 and 1<x;gain=8
 elif name=='unsigned':yes=0<=x and x<2147483648;gain=9
 elif name=='signed_wide':yes=-1099511627776<x and x<1099511627776;gain=10
 elif name=='unsigned_wide':yes=0<=x and x<9223372036854775808;gain=11
 elif name=='special_ordered':yes=-2.0<x and x<3.0;gain=12
 elif name=='special_ne':yes=-2.0!=x and x!=3.0;gain=13
 elif name=='special_negated':yes=not(-2.0<=x and x<=3.0);gain=14
 elif name=='nested_helper':middle=signed((x+1)&MASK,64);yes=-3<middle and middle<7;gain=15
 elif name=='return_helper':yes=-3<x and x<7;gain=16
 elif name=='atomic_middle':yes=True;gain=17;count=1
 elif name=='atomic_last':yes=0<=x and x<3;gain=18;count=1 if x<0 else 0
 elif name=='atomic_multi':yes=True;gain=19;count=2
 elif name=='load_snapshot':yes=True;gain=20;count=1
 else:raise AssertionError(name)
 return value+(gain if yes else 0),count
def raw_input(name,seed,lanes):
 dtype=BODIES[name][0];rng=random.Random(seed)
 if dtype=='f32':
  edges=[0,0x80000000,0x7f800000,0xff800000,0x7fc00001,0xffc12345,0x3f800000,0xbf800000,0xc0000000,0x40400000,1,0x80000001]
  bits=edges+[struct.unpack('<I',struct.pack('<f',rng.randint(-128,128)/8))[0]for _ in range(lanes-len(edges))];return struct.pack('<'+str(lanes)+'I',*bits)
 width=64 if dtype.endswith('64')else 32;mask=(1<<width)-1;edges=[0,1,mask,1<<(width-1),(1<<(width-1))-1,3,7,19,mask-2,mask-6,2,4]
 return struct.pack('<'+str(lanes)+('Q'if width==64 else'I'),*(edges+[rng.getrandbits(width)for _ in range(lanes-len(edges))]))
def values(name,seed,lanes):
 dtype=BODIES[name][0];fmt='f'if dtype=='f32'else'Q'if dtype.endswith('64')else'I';data=struct.unpack('<'+str(lanes)+fmt,raw_input(name,seed,lanes))
 return[signed(x,64 if dtype.endswith('64')else 32)if dtype.startswith('i')else x for x in data]
def initial(name,lanes):return struct.pack('<'+str(lanes)+'I',*([SENTINEL]*lanes))
def expected(name,seed,n,lanes):
 return struct.pack('<'+str(lanes)+'I',*[reference(name,x)[0]if i<n else SENTINEL for i,x in enumerate(values(name,seed,lanes))])
def expected_counter(name,seed,n,lanes):
 return struct.pack('<'+str(lanes)+'I',*[reference(name,x)[1]if i<n else 0 for i,x in enumerate(values(name,seed,lanes))])
