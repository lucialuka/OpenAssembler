"""Independent rank 1/3/4 and observable out-of-bounds Boolean-load gates."""
import random,struct
from pathlib import Path
NAMES=('rank_one','rank_three','rank_four','read_zero')
PARAMS=('a','out','counter','flag','n')
ANN={
'rank_one':("Buffer(bool, 31, strides=(2,), access='read')",'Array(bool)'),
'rank_three':("Buffer(bool, (2,3,5), strides=(1,2,8), access='read')","Buffer(bool, (2,3,5), strides=(31,9,1))"),
'rank_four':("Buffer(bool, (2,2,3,5), strides=(0,0,5,1), access='read')","Buffer(bool, (2,2,3,5))"),
'read_zero':("Buffer(bool, (4,7), strides=(11,1), access='read')",'Array(bool)'),
}
BODIES={
'rank_one':'out[i] = a[i] and flag',
'rank_three':'row=i//15\ncol=(i//5)%3\nlast=i%5\nout[row,col,last]=a[row,col,last] or flag',
'rank_four':'first=i//30\nsecond=(i//15)%2\nthird=(i//5)%3\nlast=i%5\nout[first,second,third,last]=not a[first,second,third,last]',
'read_zero':'row=i//7-1\ncol=i%7\nout[i]=a[row,col] or flag',
}
def indent(s,n=4):return '\n'.join(' '*n+line for line in s.splitlines())
def source():
 text='from openassembler.pyfront import *\nraise RuntimeError("shape discovery must never execute")\n'
 for name in NAMES:
  a,out=ANN[name];text+=f'\n@kernel\ndef {name}(a:{a},out:{out},counter:Array(i32),flag:bool,n:i32):\n    i=tid_x()+ctaid_x()*ntid_x()\n    if i<n:\n'+indent(BODIES[name],8)+'\n'
 return text

def configurations():
 text=source()
 return [dict(name='same_file',entry='source.py',files={'source.py':text}),dict(name='imported',entry='pkg/kernels.py',files={'pkg/__init__.py':'raise RuntimeError("package must not execute")\n','pkg/kernels.py':text}),dict(name='remote_package',entry='pkg/kernels.py',explicit=True,files={'pkg/__init__.py':'raise RuntimeError("remote package must not execute")\n','pkg/kernels.py':text})]
def write_tree(root,case):
 for name,data in case['files'].items():
  path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(data)
 return root/case['entry']
def raw_input(seed,lanes,canonical=False):
 rng=random.Random(seed);edges=(0,1,2,127,128,254,255,0,1);data=bytes(edges[i]if i<len(edges)else rng.randrange(256)for i in range(lanes))
 return bytes(bool(value)for value in data)if canonical else data
def initial(lanes,canonical=False):
 data=bytes((0xd2+i*17)%256 for i in range(lanes));return bytes(bool(value)for value in data)if canonical else data
def initial_counter(lanes):return b''.join(struct.pack('<i',(i%5)-2)for i in range(lanes))
def references(name,seed,flag,n,lanes,canonical=False):
 inp=raw_input(seed,lanes,canonical);out=bytearray(initial(lanes,canonical))
 for i in range(n):
  if name=='rank_one':
   value=(inp[2*i]!=0 if i<31 else False)and flag;dest=i
  elif name=='rank_three':
   row,col,last=i//15,(i//5)%3,i%5
   if row>=2:continue
   value=inp[row+col*2+last*8]!=0 or flag;dest=row*31+col*9+last
  elif name=='rank_four':
   first,second,third,last=i//30,(i//15)%2,(i//5)%3,i%5
   if first>=2:continue
   value=not bool(inp[third*5+last]);dest=first*30+second*15+third*5+last
  elif name=='read_zero':
   row,col=i//7-1,i%7;value=(inp[row*11+col]!=0 if 0<=row<4 else False)or flag;dest=i
  else:raise AssertionError(name)
  out[dest]=int(value)
 return inp,bytes(out),initial_counter(lanes)
def values(flag,n,a=0x100010000,out=0x100000000,counter=0x100020000):return dict(a=a,out=out,counter=counter,flag=int(flag),n=n)
def pack(args):return struct.pack('<QQQB3xi',*(args[name]for name in PARAMS))
