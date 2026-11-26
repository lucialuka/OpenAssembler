"""Original finite range/while control and independent complete Python references."""
import random,struct
MASK=(1<<64)-1;SENTINEL=0x13579bdf2468ace0
HEADER='from typing import Final\nfrom openassembler.pyfront import *\n'
BODIES={
 'positive_overflow':'''    value=i64(0)
    target=17
    for target in range(2147483646,2147483647,2):
        value+=i64(target)
    return value+i64(target)
''',
 'negative_overflow':'''    value=i64(0)
    target=17
    for target in range(-2147483647,-2147483648,-2):
        value+=i64(target)
    return value+i64(target)
''',
 'large_positive_step':'''    value=i64(0)
    target=17
    for target in range(-2147483648,2147483647,2147483647):
        value+=i64(target)
    return value+i64(target)
''',
 'minimum_step':'''    value=i64(0)
    target=17
    for target in range(2147483647,-2147483648,-2147483648):
        value+=i64(target)
    return value+i64(target)
''',
 'runtime_positive':'''    start=2147483643+i32(x&i64(3))
    stop=2147483647
    target=17
    value=i64(0)
    for target in range(start,stop,3):
        value+=i64(target)
    return value+i64(target)
''',
 'runtime_negative':'''    start=-2147483644-i32(x&i64(3))
    stop=-2147483648
    target=17
    value=i64(0)
    for target in range(start,stop,-3):
        value+=i64(target)
    return value+i64(target)
''',
 'overflow_else':'''    value=x
    for target in range(2147483646,2147483647,2):
        value+=i64(target)
    else:
        value+=i64(19)
    return value
''',
 'empty_else':'''    target=17
    value=x
    for target in range(3,3,2):
        value+=i64(99)
    else:
        value+=i64(target)
    return value
''',
 'break_else':'''    value=x
    target=17
    for target in range(2147483643,2147483647,3):
        if x&i64(1):
            break
        value+=i64(target)
    else:
        value+=i64(19)
    return value+i64(target)
''',
 'continue_else':'''    value=x
    target=17
    for target in range(-2147483644,-2147483648,-3):
        if x&i64(1):
            continue
        value+=i64(target)
    else:
        value+=i64(19)
    return value+i64(target)
''',
 'mutated_target':'''    value=x
    target=17
    for target in range(2147483643,2147483647,3):
        value+=i64(target)
        target=11
    return value+i64(target)
''',
 'bound_snapshot':'''    start=2147483643
    stop=2147483647
    value=x
    for target in range(start,stop,3):
        value+=i64(target)
        start=0
        stop=0
    return value+i64(start+stop)
''',
 'nested_else':'''    value=x
    for outer in range(3):
        for target in range(2147483646,2147483647,2):
            if outer==1 and x&i64(1):
                break
            value+=i64(target)
        else:
            value+=i64(7)
    else:
        value+=i64(19)
    return value
''',
 'else_break_outer':'''    value=x
    for outer in range(3):
        for target in range(3,3):
            value+=i64(99)
        else:
            value+=i64(7)
            if x&i64(1):
                break
    else:
        value+=i64(19)
    return value
''',
 'while_else':'''    value=x
    step=0
    while step<3:
        value+=i64(step)
        step+=1
    else:
        value+=i64(19)
    return value
''',
 'while_break':'''    value=x
    step=0
    while step<3:
        if step==1 and x&i64(1):
            break
        value+=i64(step)
        step+=1
    else:
        value+=i64(19)
    return value
''',
 'while_continue':'''    value=x
    step=0
    while step<3:
        step+=1
        if step==2:
            continue
        value+=i64(step)
    else:
        value+=i64(19)
    return value
''',
 'early_return':'''    for target in range(2147483643,2147483647,3):
        if x&i64(1):
            return x+i64(target)
    else:
        return x+i64(19)
''',
 'while_else_return':'''    step=0
    while step<3:
        if x&i64(1):
            return x+i64(step)
        step+=1
    else:
        return x+i64(19)
''',
 'else_continue_outer':'''    value=x
    for outer in (1,2,3):
        step=0
        while step<0:
            step+=1
        else:
            if outer==2:
                continue
        value+=i64(outer)
    return value
''',
}
NAMES=tuple(BODIES)
def configurations():
 constants=HEADER+'ZERO:Final=0\n'
 helpers=HEADER+'from . import constants\nraise RuntimeError("helper must never execute")\n'
 entry=HEADER+'from library import helpers\nfrom library import constants\nraise RuntimeError("entry must never execute")\n';single=constants
 for name,body in BODIES.items():
  fn='@device\ndef helper_'+name+'(x:i64)->i64:\n'+body;helpers+=fn
  kernel=f'@kernel\ndef {name}(a:Array(i64),out:Array(i64),n:i32):\n    i=ctaid_x()*128+tid_x()\n    if i<n:\n        out[i]=helpers.helper_{name}(a[i])+i64(constants.ZERO)\n'
  entry+=kernel;single+=fn+kernel.replace('helpers.','').replace('constants.','')
 base={'entry.py':entry,'library/__init__.py':'raise RuntimeError("package must never execute")\n','library/constants.py':constants,'library/helpers.py':helpers}
 yield dict(name='imported',files=base,entry='entry.py',explicit=False,remote=False)
 yield dict(name='same_file',files={'entry.py':single},entry='entry.py',explicit=False,remote=False)
 packaged={k:v for k,v in base.items()if k!='entry.py'};packaged.update({'kernels/__init__.py':'raise RuntimeError("entry package must never execute")\n','kernels/entry.py':entry})
 yield dict(name='remote_package',files=packaged,entry='kernels/entry.py',explicit=False,remote=True)
def write_tree(root,case):
 for name,text in case['files'].items():
  path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
 return root/case['entry']
def signed(x):return x-(1<<64)if x&(1<<63)else x
def sum_target(start,stop,step):
 values=list(range(start,stop,step));assert len(values)<8
 return sum(values)+(values[-1]if values else 17)
def reference(name,bits):
 x=signed(bits)
 if name=='positive_overflow':return sum_target(2147483646,2147483647,2)
 if name=='negative_overflow':return sum_target(-2147483647,-2147483648,-2)
 if name=='large_positive_step':return sum_target(-2147483648,2147483647,2147483647)
 if name=='minimum_step':return sum_target(2147483647,-2147483648,-2147483648)
 if name=='runtime_positive':return sum_target(2147483643+(x&3),2147483647,3)
 if name=='runtime_negative':return sum_target(-2147483644-(x&3),-2147483648,-3)
 if name=='overflow_else':return x+2147483646+19
 if name=='empty_else':return x+17
 if name=='break_else':return x+2147483643 if x&1 else x+2147483643+2*2147483646+19
 if name=='continue_else':return x-2147483647+19 if x&1 else x-2147483644-2*2147483647+19
 if name=='mutated_target':return x+2147483643+2147483646+11
 if name=='bound_snapshot':return x+2147483643+2147483646
 if name=='nested_else':return x+(2*2147483646+14 if x&1 else 3*2147483646+21)+19
 if name=='else_break_outer':return x+7 if x&1 else x+21+19
 if name=='while_else':return x+22
 if name=='while_break':return x if x&1 else x+22
 if name=='while_continue':return x+23
 if name=='early_return':return x+2147483643 if x&1 else x+19
 if name=='while_else_return':return x if x&1 else x+19
 if name=='else_continue_outer':return x+4
 raise AssertionError(name)
def raw_input(name,seed,lanes):
 rng=random.Random(seed);edges=[0,1,MASK,1<<63,(1<<63)-1,3,7,19,MASK-6,(1<<63)-7]
 return struct.pack('<'+str(lanes)+'Q',*(edges+[rng.getrandbits(64)for _ in range(lanes-len(edges))]))
def initial(name,lanes):return struct.pack('<'+str(lanes)+'Q',*([SENTINEL]*lanes))
def expected(name,seed,n,lanes):
 data=struct.unpack('<'+str(lanes)+'Q',raw_input(name,seed,lanes));return struct.pack('<'+str(lanes)+'Q',*[(reference(name,x)&MASK)if i<n else SENTINEL for i,x in enumerate(data)])
