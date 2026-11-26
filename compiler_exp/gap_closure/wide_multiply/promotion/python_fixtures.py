"""Typed Python products and independent modulo64 formulas, not a lowering mirror."""
from pathlib import Path
import struct
from fixtures import cases as ptx_cases,values,SENTINEL,CHOICE
MASK=(1<<64)-1

def cases():
 yield from ptx_cases()
 yield dict(name='cuda_multiply',mode='cuda',entry='multiply',cuda_source=Path(__file__).with_name('cuda.cu').read_text())
 for dtype in('i64','u64'):
  for mode in('arrays','parameter','literal','augmented','helper','loop','guarded','cast'):
   name='python_'+dtype+'_'+mode
   helper=f'@device\ndef product(x:{dtype},y:{dtype})->{dtype}:\n    return x*y\n\n'if mode=='helper'else''
   body='        v = a[i*2] * a[i*2+1]\n'
   if mode=='parameter':body='        v = a[i*2] * choice\n'
   if mode=='literal':body=f'        v = a[i*2] * {dtype}(-17)\n'
   if mode=='augmented':body='        v = a[i*2]\n        v *= a[i*2+1]\n'
   if mode=='helper':body='        v = product(a[i*2],a[i*2+1])\n'
   if mode=='loop':body='        v = a[i*2]\n        for step in range(3):\n            v *= a[i*2+1]\n'
   if mode=='guarded':body='        v = a[i*2]\n        if i % 2 == 0:\n            v *= choice\n'
   if mode=='cast':body=f'        v = {dtype}(i32(a[i*2])) * a[i*2+1]\n'
   source='from openassembler.pyfront import *\n'+helper+f'''@kernel
def {name}(out:Array({dtype}),a:Array({dtype}),n:i32,choice:{dtype}):
    i = ctaid_x()*128 + tid_x()
    if i < n:
'''+body+'        out[i] = v\n'
   yield dict(name=name,entry=name,mode='python',python_mode=mode,dtype=dtype,python_source=source)

def expected(data,case,n,lanes):
 if case['mode']not in('python','cuda'):
  from fixtures import expected as ptx_expected
  return ptx_expected(data,case,n,lanes)
 result=[SENTINEL]*lanes
 for i in range(n):
  if case['mode']=='cuda':value=data[i]*data[i+lanes]
  else:
   left,right=data[2*i:2*i+2];mode=case['python_mode']
   if mode in('parameter','guarded'):right=CHOICE
   if mode=='literal':right=-17
   if mode=='cast':
    left&=0xffffffff
    if left&(1<<31):left-=1<<32
   if mode=='loop':value=left*right**3
   elif mode=='guarded'and i&1:value=left
   else:value=left*right
  result[i]=value&MASK
 return struct.pack('<'+str(lanes)+'Q',*result)
