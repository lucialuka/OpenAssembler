"""Original static-value source graphs and independent reference formulas."""
import random,struct
HEADER='from openassembler.pyfront import *\n'
MASK=(1<<64)-1
SENTINEL=0x13579bdf2468ace0

def configurations():
 values='''from typing import Final as Frozen
import typing as types
raise RuntimeError('static declarations are source-only')
GAIN: Frozen[int] = 3
BIAS: types.Final[int] = -(1 << 32) + 7
COEFFICIENTS: Frozen[tuple[int,int,int]] = (3,5,7)
GRID: Frozen[tuple[tuple[int,int],tuple[int,int]]] = ((11,13),(17,19))
SCALE: Frozen[float] = (1.0 / 8.0) * 4.0
NEGZERO: Frozen[float] = -0.0
SHIFT: Frozen[int] = (5 << 2) - 15
COUNT: Frozen[int] = len(COEFFICIENTS)
'''
 helper=HEADER+'''from .constants import GAIN as gain, BIAS, SCALE, COEFFICIENTS as coefficients
from . import constants as configuration
raise RuntimeError('helper module must not execute')
@device
def signed_affine(x:i64)->i64:
    return x*gain + BIAS
@device
def unsigned_table(x:u64)->u64:
    return x*u64(coefficients[1]) + u64(coefficients[-1])
@device
def nested_table(x:u64)->u64:
    return x+u64(configuration.GRID[-1][0] + configuration.COEFFICIENTS[1:][0])
@device
def floating_scale(x:f32)->f32:
    return x*SCALE
@device
def integer_expression(x:i64)->i64:
    return (x << configuration.SHIFT) + i64(configuration.COUNT)
'''
 entry=HEADER+'''from library.helpers import signed_affine,unsigned_table,nested_table,floating_scale,integer_expression
from library import constants as configuration
raise RuntimeError('entry must not execute')
'''
 modes={'signed':'signed_affine(a[i])','unsigned':'unsigned_table(a[i])','nested':'nested_table(a[i])','floating':'floating_scale(a[i])','shifted':'integer_expression(a[i])','shared_helper':'signed_affine(a[i])','direct_constant':'a[i]+i64(configuration.COEFFICIENTS[-1])','negative_zero':'configuration.NEGZERO','local_shadow':'signed_affine(a[i])'}
 for name,expr in modes.items():
  dtype='f32'if name in('floating','negative_zero')else'u64'if name in('unsigned','nested')else'i64'
  body='''    i=ctaid_x()*128+tid_x()
    if i<n:
'''
  if name=='local_shadow':body+='        configuration = a[i]\n        out[i] = signed_affine(configuration)\n'
  else:body+='        out[i] = '+expr+'\n'
  entry+=f'@kernel\ndef {name}(a:Array({dtype}),out:Array({dtype}),n:i32):\n'+body
 base={'entry.py':entry,'library/__init__.py':"raise RuntimeError('package source-only')\n",'library/constants.py':values,'library/helpers.py':helper}
 yield dict(name='imported',entry='entry.py',files=base,explicit=False,remote=False)
 reexport=dict(base);reexport['library/__init__.py']+='from .constants import GAIN, BIAS, SCALE, COEFFICIENTS\n'
 reexport['library/helpers.py']=helper.replace('from .constants import GAIN as gain, BIAS, SCALE, COEFFICIENTS as coefficients','from . import GAIN as gain, BIAS, SCALE, COEFFICIENTS as coefficients')
 yield dict(name='reexport',entry='entry.py',files=reexport,explicit=False,remote=False)
 single=HEADER+values+helper.replace(HEADER,'').replace('from .constants import GAIN as gain, BIAS, SCALE, COEFFICIENTS as coefficients\n','').replace('from . import constants as configuration\n','').replace('gain','GAIN').replace('coefficients','COEFFICIENTS').replace('configuration.','')
 single+=entry.replace(HEADER,'').replace('from library.helpers import signed_affine,unsigned_table,nested_table,floating_scale,integer_expression\n','').replace('from library import constants as configuration\n','').replace('configuration.','')
 # Preserve local shadow control while removing its unrelated module alias.
 single=single.replace('configuration = a[i]','configuration = a[i]')
 yield dict(name='same_file',entry='entry.py',files={'entry.py':single},explicit=False,remote=False)
 packaged=dict(base);packaged.pop('entry.py');packaged['kernels/__init__.py']="raise RuntimeError('entry package source-only')\n";packaged['kernels/entry.py']=entry
 yield dict(name='remote_package',entry='kernels/entry.py',files=packaged,explicit=False,remote=True)

NAMES=('signed','unsigned','nested','floating','shifted','shared_helper','direct_constant','negative_zero','local_shadow')

def write_tree(root,case):
 for name,text in case['files'].items():
  path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
 return root/case['entry']

def raw_input(name,seed,lanes):
 if name in('floating','negative_zero'):return struct.pack('<'+str(lanes)+'f',*[(i-63)/8 for i in range(lanes)])
 rng=random.Random(seed);edges=[0,MASK,1<<63,(1<<63)-1,0xffffffff,0x100000000,0x123456789abcdef0,0xfedcba9876543210]
 data=edges+[rng.getrandbits(64)for _ in range(lanes-len(edges))]
 return struct.pack('<'+str(lanes)+'Q',*data)

def initial(name,lanes):
 return struct.pack('<'+str(lanes)+('f'if name in('floating','negative_zero')else'Q'),*([-12345.0 if name in('floating','negative_zero')else SENTINEL]*lanes))

def expected(name,seed,n,lanes):
 floating=name in('floating','negative_zero');fmt='f'if floating else'Q'
 data=struct.unpack('<'+str(lanes)+fmt,raw_input(name,seed,lanes));out=[-12345.0 if floating else SENTINEL]*lanes
 for i in range(n):
  x=data[i]
  if name in('signed','shared_helper','local_shadow'):value=x*3-(1<<32)+7
  elif name=='unsigned':value=x*5+7
  elif name=='nested':value=x+22
  elif name=='floating':value=x*0.5
  elif name=='negative_zero':value=-0.0
  elif name=='shifted':value=(x<<5)+3
  else:value=x+7
  out[i]=value if floating else value&MASK
 return struct.pack('<'+str(lanes)+fmt,*out)
