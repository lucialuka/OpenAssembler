"""Original logical Boolean source contracts with independent numeric/effect bytes."""
import random,struct

HEADER='from typing import Final\nfrom openassembler.pyfront import *\n'
SENTINEL=0x13579bdf
HELPERS='''@device
def forward(value: bool)->bool:
    return value
@device
def flip(value:bool)->bool:
    value=not value
    return value
@device
def early(value:bool,x:i32)->bool:
    if value:
        return True
    if x<0:
        return False
    return x<7
'''
BODIES={
 'forward_flag':('i32','result=i32(first)'),
 'second_flag':('i32','result=i32(second)'),
 'both_flags':('i32','result=i32(first and second)'),
 'either_flag':('i32','result=i32(first or second)'),
 'not_flag':('i32','result=i32(not first)'),
 'local_reassign':('i32','first=not first\nresult=i32(first)+2*i32(second)'),
 'helper_copy':('i32','result=i32(helpers.flip(first))+2*i32(first)'),
 'early_helper':('i32','result=i32(helpers.early(first,x))'),
 'predicate_loop':('i32','flag=first\nstep=0\nwhile step<3:\n    flag=not flag\n    step+=1\nresult=i32(flag)'),
 'branch_neighbor':('i32','result=neighbor\nif first:\n    result=x\nif second:\n    result+=neighbor'),
 'wide_select':('i64','result=wide\nif first:\n    result=x\nif second:\n    result+=i64(neighbor)'),
 'wide_flags':('i64','result=i64(first)+2*i64(second)+wide'),
 'unsigned_flags':('u64','result=u64(first)+2*u64(second)'),
 'float_select':('f32','result=-0.0\nif first:\n    result=x'),
 'float_flags':('f32','result=float(i32(first and not second))'),
 'lazy_and':('i32','result=i32(first and bool(atomic_add(counter,i,1)))'),
 'lazy_or':('i32','result=i32(first or bool(atomic_add(counter,i,1)))'),
 'lazy_nested':('i32','result=i32((first and bool(atomic_add(counter,i,1)))or(second and bool(atomic_add(counter,i,1))))'),
 'forward_helper':('i32','result=i32(helpers.forward(second))'),
 'dimension_name':('i32','result=i32(n)+i32(second)*2'),
}
NAMES=tuple(BODIES)


def parameters(name):
 dtype=BODIES[name][0]
 flags=[('first','bool'),('second','bool')]
 if name=='dimension_name':flags=[('first','bool'),('n','bool'),('second','bool')]
 # Vary physical positions; first flags can be adjacent before/after wide slots.
 index=NAMES.index(name)
 arrays=[('a','Array('+dtype+')'),('out','Array('+dtype+')'),('counter','Array(i32)')]
 numeric=[('neighbor','i32'),('wide','i64'),('count'if name=='dimension_name'else'n','i32')]
 return flags+arrays+numeric if index%3==0 else arrays+flags+numeric if index%3==1 else [*arrays,flags[0],numeric[0],*flags[1:],*numeric[1:]]


def configurations():
 constants=HEADER+'ZERO:Final=0\n'
 helpers=HEADER+'raise RuntimeError("helper must never execute")\n'+HELPERS
 entry=HEADER+'from library import helpers\nfrom library import constants\nraise RuntimeError("entry must never execute")\n'
 single=constants+'raise RuntimeError("single must never execute")\n'+HELPERS
 for name,(dtype,body)in BODIES.items():
  params=','.join(key+':'+value for key,value in parameters(name));limit='count'if name=='dimension_name'else'n'
  kernel=f'@kernel\ndef {name}({params}):\n    i=ctaid_x()*128+tid_x()\n    if i<{limit}:\n        unused=i32(constants.ZERO)\n        x=a[i]\n'+''.join('        '+line+'\n'for line in body.splitlines())+'        out[i]=result\n'
  entry+=kernel;single+=kernel.replace('helpers.','').replace('constants.','')
 base={'entry.py':entry,'library/__init__.py':'raise RuntimeError("package must never execute")\n','library/constants.py':constants,'library/helpers.py':helpers}
 yield dict(name='imported',files=base,entry='entry.py',remote=False)
 yield dict(name='same_file',files={'entry.py':single},entry='entry.py',remote=False)
 packaged={k:v for k,v in base.items()if k!='entry.py'};packaged.update({'kernels/__init__.py':'raise RuntimeError("entry package must never execute")\n','kernels/entry.py':entry})
 yield dict(name='remote_package',files=packaged,entry='kernels/entry.py',remote=True)


def write_tree(root,case):
 for name,text in case['files'].items():
  path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
 return root/case['entry']


def width(name):return 8 if BODIES[name][0].endswith('64')else 4


def raw_input(name,seed,lanes):
 rng=random.Random(seed);size=width(name)*8;mask=(1<<size)-1
 if BODIES[name][0]=='f32':
  edges=[0,0x80000000,0x7f800000,0xff800000,0x7fc12345,0xffc54321,1,0x80000001,0x3f800000,0xbf800000,0x40e00000,0xc0e00000]
  values=edges+[struct.unpack('<I',struct.pack('<f',rng.randint(-128,128)/8))[0]for _ in range(lanes-len(edges))]
 else:values=[0,1,mask,1<<(size-1),(1<<(size-1))-1,3,7,19,mask-2,mask-6,2,4]+[rng.getrandbits(size)for _ in range(lanes-12)]
 return struct.pack('<'+str(lanes)+('Q'if size==64 else'I'),*values)


def values(name,first,second,n,a=0x100010000,out=0x100000000,counter=0x100020000):
 result=dict(a=a,out=out,counter=counter,first=bool(first),second=bool(second),neighbor=-19,wide=-0x123456789abcdef,n=n)
 if name=='dimension_name':result.update(n=bool(first),count=n)
 return result


def pack(name,args):
 result=bytearray()
 for key,dtype in parameters(name):
  size=8 if dtype.startswith('Array')or dtype.endswith('64')else 1 if dtype=='bool'else 4
  result+=bytes((-len(result))%size);result+=(int(args[key])&((1<<(8*size))-1)).to_bytes(size,'little')
 return bytes(result)


def reference(name,bits,first,second):
 size=width(name)*8;mask=(1<<size)-1;x=bits-(1<<size)if BODIES[name][0].startswith('i')and bits&(1<<(size-1))else bits
 first,second=bool(first),bool(second);calls=0
 if name=='forward_flag':result=int(first)
 elif name in('second_flag','forward_helper'):result=int(second)
 elif name=='both_flags':result=int(first and second)
 elif name=='either_flag':result=int(first or second)
 elif name in('not_flag','predicate_loop'):result=int(not first)
 elif name=='local_reassign':result=int(not first)+2*int(second)
 elif name=='helper_copy':result=int(not first)+2*int(first)
 elif name=='early_helper':result=int(first or (0<=x<7))
 elif name=='branch_neighbor':result=(x if first else -19)+(-19 if second else 0)
 elif name=='wide_select':result=(x if first else -0x123456789abcdef)+(-19 if second else 0)
 elif name=='wide_flags':result=int(first)+2*int(second)-0x123456789abcdef
 elif name=='unsigned_flags':result=int(first)+2*int(second)
 elif name=='float_select':result=bits if first else 0x80000000
 elif name=='float_flags':result=0x3f800000 if first and not second else 0
 elif name=='lazy_and':result=0;calls=int(first)
 elif name=='lazy_or':result=int(first);calls=int(not first)
 elif name=='lazy_nested':result=int(first and second);calls=int(first)+int(second)
 elif name=='dimension_name':result=int(first)+2*int(second)
 else:raise AssertionError(name)
 return result&mask,calls


def initial(name,lanes):return struct.pack('<'+str(lanes)+('Q'if width(name)==8 else'I'),*([SENTINEL]*lanes))


def expected(name,seed,first,second,n,lanes):
 fmt='<'+str(lanes)+('Q'if width(name)==8 else'I');raw=struct.unpack(fmt,raw_input(name,seed,lanes))
 return struct.pack(fmt,*[reference(name,b,first,second)[0]if i<n else SENTINEL for i,b in enumerate(raw)])


def expected_counter(name,seed,first,second,n,lanes):
 fmt='<'+str(lanes)+('Q'if width(name)==8 else'I');raw=struct.unpack(fmt,raw_input(name,seed,lanes))
 return struct.pack('<'+str(lanes)+'I',*[reference(name,b,first,second)[1]if i<n else 0 for i,b in enumerate(raw)])
