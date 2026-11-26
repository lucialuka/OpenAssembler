"""Original scalar operand selection, Boolean truth and independent raw-bit effects."""
import random,struct
MASK=(1<<64)-1;SENTINEL=0x13579bdf
HEADER='from typing import Final\nfrom openassembler.pyfront import *\n'
BODIES={
 'operand_and':('i32','    return x and 7\n'),
 'operand_or':('i32','    return x or 9\n'),
 'operand_nested':('i32','    return (x and 7) or 9\n'),
 'truth_flags':('i32','    flag=0<x<7\n    inverse=not (flag or False)\n    enabled=bool(x) and True\n    return i32(flag)+i32(inverse)*2+i32(enabled)*4\n'),
 'literal_left':('i64','    return 0 or x\n'),
 'literal_right':('i64','    return x or 0\n'),
 'signed_wide':('i64','    return x and -7\n'),
 'unsigned_wide':('u64','    return x or 18446744073709551615\n'),
 'bool_literals':('i32','    first=True\n    second=False\n    return i32(first and not second)+i32(second or first)*2\n'),
 'not_wide':('i64','    return i64(not x)\n'),
 'float_and':('f32','    return x and -0.0\n'),
 'float_or':('f32','    return x or -0.0\n'),
 'float_truth':('f32','    return float(i32(bool(x)))\n'),
 'float_not':('f32','    flag=x!=0.0\n    return float(i32(not flag))\n'),
 'predicate_while':('i32','    flag=x<3\n    step=0\n    value=0\n    again=flag and step<3\n    while again:\n        step+=1\n        value+=1\n        flag=step<2\n        again=flag and step<3\n    return value\n'),
 'nested_helper':('i64','    return pick(x) or 9\n'),
 'atomic_and':('i32','    return x and atomic_add(count,i,1)\n'),
 'atomic_or':('i32','    return x or atomic_add(count,i,1)\n'),
 'atomic_nested':('i32','    return (x and atomic_add(count,i,1)) or atomic_add(count,i,1)\n'),
 'atomic_bool':('i32','    flag=bool(atomic_add(count,i,1))\n    second=not (flag or bool(atomic_add(count,i,1)))\n    return i32(flag)+i32(second)*2\n'),
}
NAMES=tuple(BODIES)
def configurations():
 constants=HEADER+'ZERO:Final=0\n'
 helpers=HEADER+'from . import constants\nraise RuntimeError("helper must never execute")\n@device\ndef pick(x:i64)->i64:\n    return x and 7\n'
 entry=HEADER+'from library import helpers\nfrom library import constants\nraise RuntimeError("entry must never execute")\n';single=constants+helpers[helpers.index('@device'):]
 for name,(dtype,body)in BODIES.items():
  if name.startswith('atomic_'):
   lines=body.replace('    return ','    result=').splitlines();kernel=f'@kernel\ndef {name}(a:Array({dtype}),out:Array({dtype}),count:Array(i32),n:i32):\n    i=ctaid_x()*128+tid_x()\n    if i<n:\n        unused=i32(constants.ZERO)\n        x=a[i]\n'+''.join('    '+line+'\n'for line in lines)+'        out[i]=result\n'
  else:
   fn=f'@device\ndef helper_{name}(x:{dtype})->{dtype}:\n'+body;helpers+=fn;single+=fn
   kernel=f'@kernel\ndef {name}(a:Array({dtype}),out:Array({dtype}),count:Array(i32),n:i32):\n    i=ctaid_x()*128+tid_x()\n    if i<n:\n        unused=i32(constants.ZERO)\n        out[i]=helpers.helper_{name}(a[i])\n'
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
def width(name):return 8 if BODIES[name][0].endswith('64')else 4
def raw_input(name,seed,lanes):
 dtype=BODIES[name][0];rng=random.Random(seed)
 if dtype=='f32':
  edges=[0,0x80000000,0x7f800000,0xff800000,0x7fc12345,0xffc54321,0x3f800000,0xbf800000,0xc0000000,0x40400000,1,0x80000001]
  bits=edges+[struct.unpack('<I',struct.pack('<f',rng.randint(-128,128)/8))[0]for _ in range(lanes-len(edges))]
 else:
  size=width(name)*8;mask=(1<<size)-1;edges=[0,1,mask,1<<(size-1),(1<<(size-1))-1,3,7,19,mask-2,mask-6,2,4];bits=edges+[rng.getrandbits(size)for _ in range(lanes-len(edges))]
 return struct.pack('<'+str(lanes)+('Q'if width(name)==8 else'I'),*bits)
def truth(name,bits):return bool(struct.unpack('<f',bits.to_bytes(4,'little'))[0])if BODIES[name][0]=='f32'else bool(bits)
def reference(name,bits):
 dtype=BODIES[name][0];size=width(name)*8;mask=(1<<size)-1;x=bits-(1<<size)if dtype.startswith('i')and bits&(1<<(size-1))else bits;t=truth(name,bits);calls=0
 if name=='operand_and':result=7 if t else x
 elif name=='operand_or':result=x if t else 9
 elif name=='operand_nested':result=7 if t else 9
 elif name=='truth_flags':flag=0<x<7;result=int(flag)+2*int(not flag)+4*int(t)
 elif name in('literal_left','literal_right'):result=bits
 elif name=='signed_wide':result=-7 if t else x
 elif name=='unsigned_wide':result=x if t else mask
 elif name=='bool_literals':result=3
 elif name=='not_wide':result=int(not t)
 elif name=='float_and':return(0x80000000 if t else bits),0
 elif name=='float_or':return(bits if t else 0x80000000),0
 elif name=='float_truth':return(0x3f800000 if t else 0),0
 elif name=='float_not':return(0 if t else 0x3f800000),0
 elif name=='predicate_while':result=2 if x<3 else 0
 elif name=='nested_helper':result=7 if t else 9
 elif name=='atomic_and':result=0;calls=int(t)
 elif name=='atomic_or':result=x if t else 0;calls=int(not t)
 elif name=='atomic_nested':result=1 if t else 0;calls=2 if t else 1
 elif name=='atomic_bool':result=0;calls=2
 else:raise AssertionError(name)
 return result&mask,calls
def initial(name,lanes):return struct.pack('<'+str(lanes)+('Q'if width(name)==8 else'I'),*([SENTINEL]*lanes))
def expected(name,seed,n,lanes):
 fmt='<'+str(lanes)+('Q'if width(name)==8 else'I');bits=struct.unpack(fmt,raw_input(name,seed,lanes));return struct.pack(fmt,*[reference(name,x)[0]if i<n else SENTINEL for i,x in enumerate(bits)])
def expected_counter(name,seed,n,lanes):
 fmt='<'+str(lanes)+('Q'if width(name)==8 else'I');bits=struct.unpack(fmt,raw_input(name,seed,lanes));return struct.pack('<'+str(lanes)+'I',*[reference(name,x)[1]if i<n else 0 for i,x in enumerate(bits)])
