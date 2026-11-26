"""Original logical helper contracts and independent typed, bitwise references."""
import random
import struct

SENTINEL = 0x13579bdf
HEADER = 'from typing import Final\nfrom openassembler.pyfront import *\n'
HELPERS = '''@device
def identity(flag: bool) -> bool:
    return flag
@device
def flip(flag: "bool") -> "bool":
    return not flag
@device
def both(first: bool, second: bool) -> bool:
    return first and second
@device
def either(first: bool, second: bool) -> bool:
    return first or second
@device
def early(flag: bool, x: i32) -> bool:
    if flag:
        return True
    if x < 0:
        return False
    return x < 7
@device
def assign(flag: bool) -> bool:
    flag = not flag
    return flag
@device
def toggle(flag: bool, trips: i32) -> bool:
    step = 0
    while step < trips:
        flag = not flag
        step += 1
    return flag
@device
def loop_return(flag: bool, x: i32) -> bool:
    for step in range(3):
        if flag:
            return not flag
        if x < step:
            return True
    return flag
@device
def nested(flag: bool) -> bool:
    return flip(identity(flag))
@device
def pick(flag: bool, x: i64) -> i64:
    if flag:
        return x + i64(3)
    return i64(-7)
@device
def float_pick(flag: bool, x: f32) -> f32:
    if flag:
        return x
    return -0.0
@device
def wide_truth(x: i64) -> bool:
    return bool(x)
@device
def unsigned_truth(x: u64) -> bool:
    return not x
@device
def chain(x: i32) -> bool:
    return -3 < x < 7
'''
# Entries return numeric bytes only; no Boolean physical launch/storage ABI.
BODIES = {
    'identity_value': ('i32', 'result=i32(helpers.identity(bool(x)))'),
    'invert_value': ('i32', 'result=i32(helpers.flip(bool(x)))'),
    'and_values': ('i32', 'result=i32(helpers.both(bool(x),x<3))'),
    'or_values': ('i32', 'result=i32(helpers.either(bool(x),x<3))'),
    'early_value': ('i32', 'result=i32(helpers.early(x<0,x))'),
    'caller_copy': ('i32', 'flag=bool(x)\nresult=i32(helpers.assign(flag))+i32(flag)*2'),
    'predicate_loop': ('i32', 'result=i32(helpers.toggle(bool(x),3))'),
    'loop_return_value': ('i32', 'result=i32(helpers.loop_return(bool(x),x))'),
    'nested_value': ('i32', 'result=i32(helpers.nested(bool(x)))'),
    'wide_pick': ('i64', 'result=helpers.pick(bool(x),x)'),
    'float_select': ('f32', 'result=helpers.float_pick(bool(x),x)'),
    'wide_boolean': ('i64', 'result=i64(helpers.wide_truth(x))'),
    'unsigned_boolean': ('u64', 'result=u64(helpers.unsigned_truth(x))'),
    'chain_boolean': ('i32', 'result=i32(helpers.chain(x))'),
    'literal_boolean': ('i32', 'result=i32(helpers.identity(True))+2*i32(helpers.flip(False))'),
    'eager_arguments': ('i32', 'result=i32(helpers.either(bool(atomic_add(count,i,1)),bool(atomic_add(count,i,1))))'),
    'lazy_call': ('i32', 'result=i32(bool(x) and helpers.flip(bool(atomic_add(count,i,1))))'),
    'argument_order': ('i32', 'result=i32(helpers.either(bool(atomic_add(count,i,1)),bool(count[i])))'),
    'caller_loop': ('i32', 'flag=bool(x)\nstep=0\nwhile step<4:\n    flag=helpers.flip(flag)\n    step+=1\nresult=i32(flag)'),
    'wide_conversion': ('u64', 'result=u64(helpers.flip(bool(x)))'),
}
NAMES = tuple(BODIES)


def configurations():
    constants = HEADER+'ZERO:Final=0\n'
    helpers = HEADER+'from . import constants\nraise RuntimeError("helper must never execute")\n'+HELPERS
    entry = HEADER+'from library import helpers\nfrom library import constants\nraise RuntimeError("entry must never execute")\n'
    single = constants+'raise RuntimeError("same file must never execute")\n'+HELPERS
    for name, (dtype, body) in BODIES.items():
        kernel = (f'@kernel\ndef {name}(a:Array({dtype}),out:Array({dtype}),count:Array(i32),n:i32):\n'
                  '    i=ctaid_x()*128+tid_x()\n    if i<n:\n        unused=i32(constants.ZERO)\n        x=a[i]\n'
                  + ''.join('        '+line+'\n' for line in body.splitlines())+'        out[i]=result\n')
        entry += kernel
        single += kernel.replace('helpers.', '').replace('constants.', '')
    base = {'entry.py': entry, 'library/__init__.py': 'raise RuntimeError("package must never execute")\n',
            'library/constants.py': constants, 'library/helpers.py': helpers}
    yield dict(name='imported', files=base, entry='entry.py', explicit=False, remote=False)
    yield dict(name='same_file', files={'entry.py': single}, entry='entry.py', explicit=False, remote=False)
    packaged = {k: v for k, v in base.items() if k != 'entry.py'}
    packaged.update({'kernels/__init__.py': 'raise RuntimeError("entry package must never execute")\n', 'kernels/entry.py': entry})
    yield dict(name='remote_package', files=packaged, entry='kernels/entry.py', explicit=False, remote=True)


def write_tree(root, case):
    for name, text in case['files'].items():
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root/case['entry']


def width(name):
    return 8 if BODIES[name][0].endswith('64') else 4


def raw_input(name, seed, lanes):
    dtype = BODIES[name][0]
    rng = random.Random(seed)
    if dtype == 'f32':
        edges = [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc54321,
                 0x3f800000, 0xbf800000, 0xc0000000, 0x40400000, 1, 0x80000001]
        bits = edges+[struct.unpack('<I', struct.pack('<f', rng.randint(-128, 128)/8))[0] for _ in range(lanes-len(edges))]
    else:
        size = width(name)*8
        mask = (1 << size)-1
        edges = [0, 1, mask, 1 << (size-1), (1 << (size-1))-1, 3, 7, 19, mask-2, mask-6, 2, 4]
        bits = edges+[rng.getrandbits(size) for _ in range(lanes-len(edges))]
    return struct.pack('<'+str(lanes)+('Q' if width(name) == 8 else 'I'), *bits)


def reference(name, bits):
    dtype = BODIES[name][0]
    size = width(name)*8
    mask = (1 << size)-1
    x = bits-(1 << size) if dtype.startswith('i') and bits & (1 << (size-1)) else bits
    truth = bool(struct.unpack('<f', bits.to_bytes(4, 'little'))[0]) if dtype == 'f32' else bool(bits)
    calls = 0
    if name in ('identity_value', 'caller_loop', 'wide_boolean'):
        result = int(truth)
    elif name in ('invert_value', 'predicate_loop', 'nested_value', 'unsigned_boolean', 'wide_conversion'):
        result = int(not truth)
    elif name == 'and_values': result = int(truth and x < 3)
    elif name == 'or_values': result = int(truth or x < 3)
    elif name == 'early_value': result = int(x < 7)
    elif name == 'caller_copy': result = int(not truth)+2*int(truth)
    elif name == 'loop_return_value': result = int(not truth and x < 2)
    elif name == 'wide_pick': result = x+3 if truth else -7
    elif name == 'float_select': result = bits if truth else 0x80000000
    elif name == 'chain_boolean': result = int(-3 < x < 7)
    elif name == 'literal_boolean': result = 3
    elif name == 'eager_arguments': result = 1; calls = 2
    elif name == 'lazy_call': result = int(truth); calls = int(truth)
    elif name == 'argument_order': result = 1; calls = 1
    else: raise AssertionError(name)
    return result & mask, calls


def initial(name, lanes):
    return struct.pack('<'+str(lanes)+('Q' if width(name) == 8 else 'I'), *([SENTINEL]*lanes))


def expected(name, seed, n, lanes):
    fmt = '<'+str(lanes)+('Q' if width(name) == 8 else 'I')
    bits = struct.unpack(fmt, raw_input(name, seed, lanes))
    return struct.pack(fmt, *[reference(name, x)[0] if i < n else SENTINEL for i, x in enumerate(bits)])


def expected_counter(name, seed, n, lanes):
    fmt = '<'+str(lanes)+('Q' if width(name) == 8 else 'I')
    bits = struct.unpack(fmt, raw_input(name, seed, lanes))
    return struct.pack('<'+str(lanes)+'I', *[reference(name, x)[1] if i < n else 0 for i, x in enumerate(bits)])
