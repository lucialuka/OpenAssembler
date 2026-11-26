"""Shared-operand regression: register and scalar-parameter squares keep all bits."""
import struct
from python_fixtures import values,SENTINEL,CHOICE
MASK=(1<<64)-1

def cases():
 for dtype in('i64','u64'):
  for mode in('register','parameter'):
   name='python_'+dtype+'_square_'+mode
   expression='a[i*2]'if mode=='register'else'choice'
   source=f'''from openassembler.pyfront import *
@kernel
def {name}(out:Array({dtype}),a:Array({dtype}),n:i32,choice:{dtype}):
    i = ctaid_x()*128 + tid_x()
    if i < n:
        x = {expression}
        out[i] = x*x
'''
   yield dict(name=name,entry=name,mode='python',python_mode='square_'+mode,dtype=dtype,python_source=source)

def expected(data,case,n,lanes):
 result=[SENTINEL]*lanes
 for i in range(n):
  x=data[2*i]if case['python_mode']=='square_register'else CHOICE
  result[i]=pow(x,2,1<<64)
 return struct.pack('<'+str(lanes)+'Q',*result)
