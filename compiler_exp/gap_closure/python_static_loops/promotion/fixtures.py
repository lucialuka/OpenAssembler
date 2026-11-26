"""Original finite tuple-loop sources and independently computed complete references."""
import random,struct
MASK=(1<<64)-1
SENTINEL=0x13579bdf2468ace0
HEADER='from typing import Final\nfrom openassembler.pyfront import *\n'
BODIES={
 'plain':('i64','(3,-5,7)','    value=i64(0)\n    for c in VALUES:\n        value=value*x+i64(c)\n'),
 'empty':('i64','()','    value=x\n    for c in VALUES:\n        value+=i64(c)\n    else:\n        value+=i64(17)\n'),
 'break_loop':('i64','(3,-5,7)','    value=i64(0)\n    for c in VALUES:\n        if x<i64(c):\n            break\n        value=value*x+i64(c)\n'),
 'continue_loop':('i64','(3,-5,7)','    value=i64(0)\n    for c in VALUES:\n        if x<i64(c):\n            continue\n        value=value*x+i64(c)\n'),
 'nested':('i64','(3,-5,7)','    value=x\n    for c in VALUES:\n        for d in (1,2):\n            if c<0 and d==2:\n                break\n            value+=i64(c*d)\n'),
 'range_inside':('i64','(3,-5,7)','    value=x\n    for c in VALUES:\n        for d in range(3):\n            if d==1:\n                continue\n            value+=i64(c*d)\n'),
 'tuple_inside_range':('i64','(3,-5,7)','    value=x\n    for step in range(2):\n        for c in VALUES:\n            value+=i64(c+step)\n'),
 'target_after':('i64','(3,-5,7)','    value=x\n    for c in VALUES:\n        value+=i64(c)\n    value+=i64(c)\n'),
 'target_mutation':('i64','(3,-5,7)','    value=x\n    for c in VALUES:\n        value+=i64(c)\n        c=99\n    value+=i64(c)\n'),
 'break_target':('i64','(3,-5,7)','    value=x\n    for c in VALUES:\n        if x<i64(c):\n            break\n        value+=i64(c)\n    value+=i64(c)\n'),
 'empty_target':('i64','()','    value=x\n    c=11\n    for c in VALUES:\n        value+=i64(c)\n    value+=i64(c)\n'),
 'loop_else':('i64','(3,-5,7)','    value=x\n    for c in VALUES:\n        if x<i64(c):\n            break\n        value+=i64(c)\n    else:\n        value+=i64(17)\n'),
 'nested_else':('i64','(3,-5,7)','    value=x\n    for c in VALUES:\n        for d in (1,2):\n            if x<i64(c*d):\n                break\n            value+=i64(d)\n        else:\n            value+=i64(7)\n    else:\n        value+=i64(19)\n'),
 'sliced':('i64','(3,-5,7)','    value=x\n    for c in VALUES[::-1]:\n        value=value*i64(3)+i64(c)\n'),
 'literal_tuple':('i64','(3,-5,7)','    value=x\n    for c in (2,3,5):\n        value*=i64(c)\n'),
 'wide_signed':('i64','(-9223372036854775808,1099511627777,9223372036854775807)','    value=x\n    for c in VALUES:\n        value+=c\n'),
 'wide_unsigned':('u64','(0,9223372036854775808,18446744073709551615)','    value=x\n    for c in VALUES:\n        value+=c\n'),
 'wide_target':('i64','(3,-5,7)','    value=x\n    c=i64(11)\n    for c in VALUES:\n        value+=c\n    value+=c\n'),
 'floating':('f32','(0.5,-0.25,2.0)','    value=x\n    for c in VALUES:\n        value=value*c+c\n'),
 'mixed_exact':('f32','(2,0.5,-1)','    value=x\n    for c in VALUES:\n        value=value*c+c\n'),
}
NAMES=tuple(BODIES)

def configurations():
 constants=HEADER+'\n'.join('VALUES_'+name.upper()+':Final='+values for name,(_,values,_)in BODIES.items())+'\n'
 helpers=HEADER+'from . import constants as constants\nraise RuntimeError("helper module must never execute")\n'
 entry=HEADER+'from library import helpers\nraise RuntimeError("entry must never execute")\n'
 single=constants
 for name,(dtype,values,body)in BODIES.items():
  fn='@device\ndef helper_'+name+'(x:'+dtype+')->'+dtype+':\n'+body.replace('VALUES','constants.VALUES_'+name.upper())+'    return value\n'
  helpers+=fn
  kernel=f'@kernel\ndef {name}(a:Array({dtype}),out:Array({dtype}),n:i32):\n    i=ctaid_x()*128+tid_x()\n    if i<n:\n        out[i]=helpers.helper_{name}(a[i])\n'
  entry+=kernel;single+=fn.replace('constants.','')+kernel.replace('helpers.','')
 base={'entry.py':entry,'library/__init__.py':'raise RuntimeError("package must never execute")\n','library/constants.py':constants,'library/helpers.py':helpers}
 yield dict(name='imported',files=base,entry='entry.py',explicit=False,remote=False)
 yield dict(name='same_file',files={'entry.py':single},entry='entry.py',explicit=False,remote=False)
 packaged={k:v for k,v in base.items()if k!='entry.py'};packaged.update({'kernels/__init__.py':'raise RuntimeError("entry package must never execute")\n','kernels/entry.py':entry})
 yield dict(name='remote_package',files=packaged,entry='kernels/entry.py',explicit=False,remote=True)

def write_tree(root,case):
 for name,text in case['files'].items():
  path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
 return root/case['entry']

def raw_input(name,seed,lanes):
 if name in('floating','mixed_exact'):return struct.pack('<'+str(lanes)+'f',*[(i-63)/8 for i in range(lanes)])
 rng=random.Random(seed);edges=[0,1,MASK,1<<63,(1<<63)-1,3,7,0xffffffff]
 data=edges+[rng.getrandbits(64)for _ in range(lanes-len(edges))]
 return struct.pack('<'+str(lanes)+'Q',*data)

def initial(name,lanes):
 return struct.pack('<'+str(lanes)+('f'if name in('floating','mixed_exact')else'Q'),*([-12345.0 if name in('floating','mixed_exact')else SENTINEL]*lanes))

def reference(name,bits):
 x=bits if name=='wide_unsigned'or name in('floating','mixed_exact')else bits-(1<<64)if bits&(1<<63)else bits
 if name=='plain':return(3*x-5)*x+7
 if name=='empty':return x+17
 if name=='break_loop':
  value=0
  for c in(3,-5,7):
   if x<c:break
   value=value*x+c
  return value
 if name=='continue_loop':
  value=0
  for c in(3,-5,7):
   if x<c:continue
   value=value*x+c
  return value
 if name=='nested':return x+3*3-5+3*7
 if name=='range_inside':return x+2*(3-5+7)
 if name=='tuple_inside_range':return x+2*(3-5+7)+3
 if name=='target_after'or name=='wide_target':return x+12
 if name=='target_mutation':return x+104
 if name=='empty_target':return x+11
 if name in('break_target','loop_else'):
  value=x
  for c in(3,-5,7):
   if x<c:break
   value+=c
  else:
   if name=='loop_else':value+=17
  return value+c if name=='break_target'else value
 if name=='nested_else':
  value=x
  for c in(3,-5,7):
   for d in(1,2):
    if x<c*d:break
    value+=d
   else:value+=7
  return value+19
 if name=='sliced':return((x*3+7)*3-5)*3+3
 if name=='literal_tuple':return x*30
 if name=='wide_signed':return x+1099511627776
 if name=='wide_unsigned':return x+(1<<63)-1
 if name in('floating','mixed_exact'):
  import struct
  round32=lambda value:struct.unpack('<f',struct.pack('<f',value))[0]
  value=x
  for c in((0.5,-0.25,2.0)if name=='floating'else(2,0.5,-1)):value=round32(round32(value*c)+c)
  return value
 raise AssertionError(name)

def expected(name,seed,n,lanes):
 floating=name in('floating','mixed_exact');fmt='f'if floating else'Q';data=struct.unpack('<'+str(lanes)+fmt,raw_input(name,seed,lanes));out=[-12345.0 if floating else SENTINEL]*lanes
 for i in range(n):
  value=reference(name,data[i]);out[i]=value if floating else value&MASK
 return struct.pack('<'+str(lanes)+fmt,*out)
