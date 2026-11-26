"""Original source-tree variants and independent full-bit numerical references."""
from pathlib import Path
import random,struct
HERE=Path(__file__).resolve().parent
MASK=(1<<64)-1
SENTINEL=0x13579bdf2468ace0

def configurations():
 base={str(p.relative_to(HERE/'corpus')):p.read_text()for p in sorted((HERE/'corpus').rglob('*.py'))}
 yield dict(name='direct',files=base,entry='entry.py',explicit=False,remote=False)
 reexport=dict(base);reexport['arithmetic/__init__.py']+='\nfrom .transforms import signed_step\nfrom .base import affine\n'
 reexport['entry.py']=base['entry.py'].replace('from arithmetic.transforms import signed_step as repeated_product','from arithmetic import signed_step as repeated_product').replace('from arithmetic.base import affine','from arithmetic import affine')
 yield dict(name='reexport',files=reexport,entry='entry.py',explicit=False,remote=False)
 packaged={**base,'application/__init__.py':"raise RuntimeError('must not execute entry package')\n",'application/entry.py':base['entry.py']};packaged.pop('entry.py')
 yield dict(name='package_entry',files=packaged,entry='application/entry.py',explicit=False,remote=False)
 explicit={**base,'kernels/entry.py':base['entry.py']};explicit.pop('entry.py')
 yield dict(name='explicit_root',files=explicit,entry='kernels/entry.py',explicit=True,remote=False,directory=True)
 yield dict(name='remote_package',files=packaged,entry='application/entry.py',explicit=False,remote=True)

def write_tree(root,case):
 for name,text in case['files'].items():
  path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
 return root/case['entry']

def values(name,seed,lanes):
 if name=='floating':return [(i-63)/8 for i in range(lanes)],[(i%11-5)/4 for i in range(lanes)]
 rng=random.Random(seed);edges=[0,MASK,1<<63,(1<<63)-1,0xffffffff,0x100000000,0x123456789abcdef0,0xfedcba9876543210]
 left=[edges[i]if i<len(edges)else rng.getrandbits(64)for i in range(lanes)]
 right=[MASK,MASK,MASK,2,0xffffffff,0x8000000000000001,0,1]+[rng.getrandbits(64)for _ in range(lanes-8)]
 return left,right

def raw_data(name,seed,lanes):
 left,right=values(name,seed,lanes);fmt='f'if name=='floating'else'Q'
 return struct.pack('<'+str(lanes)+fmt,*left),struct.pack('<'+str(lanes)+fmt,*right)

def initial(name,lanes):
 return struct.pack('<'+str(lanes)+('f'if name=='floating'else'Q'),*([-12345.0 if name=='floating'else SENTINEL]*lanes))

def expected(name,seed,n,lanes):
 left,right=values(name,seed,lanes);out=[-12345.0 if name=='floating'else SENTINEL]*lanes
 for i in range(n):
  if name=='signed':out[i]=(left[i]*right[i]**3)&MASK
  elif name=='unsigned':out[i]=(left[i]*right[i]+left[i])&MASK
  else:out[i]=left[i]*2+right[i]
 return struct.pack('<'+str(lanes)+('f'if name=='floating'else'Q'),*out)
