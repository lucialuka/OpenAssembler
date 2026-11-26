"""Original source trees and independent complete-byte Boolean storage references."""
import random,struct
from pathlib import Path
NAMES=('copy','invert','lazy_and','lazy_or','helper','equal_flag','atomic_truth','constants','predicate_loop','alias_invert','buffer_padded','buffer_transpose','buffer_broadcast','buffer_bounds','shared_reverse','shared_mixed','shared_broadcast','shared_loop')
PARAMS=('a','out','counter','flag','n')
HELPERS='''from openassembler.pyfront import *
raise RuntimeError('Boolean storage helpers must not execute')
@device
def choose(value: bool, flag: bool) -> bool:
    if flag:
        return not value
    return value
'''
BODIES={
'copy':'out[i] = a[i]',
'invert':'out[i] = not a[i]',
'lazy_and':'out[i] = a[i] and bool(atomic_add(counter, i, 1))',
'lazy_or':'out[i] = a[i] or bool(atomic_add(counter, i, 1))',
'helper':'out[i] = choose(a[i], flag)',
'equal_flag':'out[i] = i32(a[i]) == i32(flag)',
'atomic_truth':'out[i] = bool(atomic_add(counter, i, 1))',
'constants':'if i % 2 == 0:\n    out[i] = True\nelse:\n    out[i] = False',
'predicate_loop':'value = a[i]\nfor j in range(3):\n    value = not value\nout[i] = value',
'alias_invert':'out[i] = not a[i]',
'buffer_padded':'row = i // 7\ncol = i % 7\nout[row, col] = a[row, col] or flag',
'buffer_transpose':'row = i // 7\ncol = i % 7\nout[row, col] = a[row, col] and flag',
'buffer_broadcast':'row = i // 7\ncol = i % 7\nout[row, col] = not a[row, col]',
'buffer_bounds':'row = i // 7 - 1\ncol = i % 7\nout[row, col] = a[row, col] or flag',
}
ANN={
'buffer_padded':("Buffer(bool, (4, 7), strides=(11, 1), access='read')","Buffer(bool, (4, 7), strides=(9, 1))"),
'buffer_transpose':("Buffer(bool, (4, 7), strides=(1, 5), access='read')","Buffer(bool, (4, 7))"),
'buffer_broadcast':("Buffer(bool, (4, 7), strides=(0, 1), access='read')","Buffer(bool, (4, 7))"),
'buffer_bounds':("Buffer(bool, (4, 7), access='read')","Buffer(bool, (4, 7))"),
}
SHARED={
'shared_reverse':'''s = shared_array(bool, 128)
s[t] = a[i]
syncthreads()
if i < n:
    out[i] = s[127 - t]''',
'shared_mixed':'''pad = shared_array(bool, 3)
wide = shared_array(i64, 128)
s = shared_array(bool, 128)
if t < 3:
    pad[t] = flag
wide[t] = i64(t)
s[t] = a[i]
syncthreads()
if i < n:
    out[i] = s[127 - t] and (wide[127 - t] >= 0)''',
'shared_broadcast':'''s = shared_array(bool, 1)
if t == 0:
    s[0] = flag
syncthreads()
if i < n:
    out[i] = s[0]''',
'shared_loop':'''s = shared_array(bool, 128)
s[t] = a[i]
syncthreads()
value = s[(t + 65) % 128]
for j in range(3):
    value = not value
if i < n:
    out[i] = value''',
}
def indent(s,n=4):return '\n'.join(' '*n+line for line in s.splitlines())
def source(imported=False):
    text=('from .helpers import choose\n'if imported else HELPERS)+'\nfrom openassembler.pyfront import *\nraise RuntimeError("Boolean storage source discovery must not execute")\n'
    for name in NAMES:
        a,out=ANN.get(name,('Array(bool)','Array(bool)'))
        text+=f'\n@kernel\ndef {name}(a: {a}, out: {out}, counter: Array(i32), flag: bool, n: i32):\n'
        text+='    t = tid_x()\n    i = t + ctaid_x() * ntid_x()\n'
        text+=indent(SHARED[name]if name in SHARED else 'if i < n:\n'+indent(BODIES[name]))+'\n'
    return text

def configurations():
    same=source();imported=source(True)
    return [dict(name='same_file',entry='source.py',files={'source.py':same}),dict(name='imported',entry='pkg/kernels.py',files={'pkg/__init__.py':'raise RuntimeError("package must not execute")\n','pkg/helpers.py':HELPERS,'pkg/kernels.py':imported}),dict(name='remote_package',entry='entry.py',explicit=True,files={'entry.py':'from pkg.kernels import *\nraise RuntimeError("entry must not execute")\n','pkg/__init__.py':'raise RuntimeError("package must not execute")\n','pkg/helpers.py':HELPERS,'pkg/kernels.py':imported})]
def write_tree(root,case):
    for name,data in case['files'].items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(data)
    return root/case['entry']
def raw_input(seed,lanes):
    rng=random.Random(seed);edges=(0,1,2,127,128,254,255,0,1)
    return bytes(edges[i]if i<len(edges)else rng.randrange(256)for i in range(lanes))
def initial(lanes):return bytes((0xd2+i*17)%256 for i in range(lanes))
def initial_counter(lanes):return b''.join(struct.pack('<i',(i%5)-2)for i in range(lanes))
def references(name,seed,flag,n,lanes):
    inp=raw_input(seed,lanes);out=bytearray(inp if name=='alias_invert'else initial(lanes));counter=bytearray(initial_counter(lanes))
    for i in range(n):
        truth=inp[i]!=0;old=(i%5)-2;dest=i;effect=0
        if name=='copy':value=truth
        elif name in('invert','alias_invert','predicate_loop'):value=not truth
        elif name=='lazy_and':value=truth and old!=0;effect=int(truth)
        elif name=='lazy_or':value=truth or old!=0;effect=int(not truth)
        elif name=='helper':value=not truth if flag else truth
        elif name=='equal_flag':value=truth==flag
        elif name=='atomic_truth':value=old!=0;effect=1
        elif name=='constants':value=i%2==0
        elif name in ANN:
            row,col=i//7,i%7
            if name=='buffer_bounds':row-=1
            if not 0<=row<4:continue
            astride={'buffer_padded':11,'buffer_transpose':1,'buffer_broadcast':0,'buffer_bounds':7}[name]
            cstride=5 if name=='buffer_transpose'else 1
            truth=inp[row*astride+col*cstride]!=0
            dest=row*(9 if name=='buffer_padded'else 7)+col
            value=truth or flag if name in('buffer_padded','buffer_bounds')else truth and flag if name=='buffer_transpose'else not truth
        elif name in('shared_reverse','shared_mixed'):value=inp[(i//128)*128+127-i%128]!=0
        elif name=='shared_broadcast':value=flag
        elif name=='shared_loop':value=not bool(inp[(i//128)*128+(i%128+65)%128])
        else:raise AssertionError(name)
        out[dest]=int(value)
        if effect:struct.pack_into('<i',counter,4*i,old+effect)
    return (bytes(out)if name=='alias_invert'else inp),bytes(out),bytes(counter)
def values(flag,n,a=0x100010000,out=0x100000000,counter=0x100020000):return dict(a=a,out=out,counter=counter,flag=int(flag),n=n)
def pack(args):return struct.pack('<QQQB3xi',*(args[name]for name in PARAMS))
