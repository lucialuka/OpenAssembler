"""Original pure return paths and independent whole-bit reference arithmetic."""
import random
import struct

MASK = (1 << 64) - 1
SENTINEL = 0x13579bdf2468ace0
HEADER = 'from typing import Final\nfrom openassembler.pyfront import *\n'
BODIES = {
    'classify': ('i64', '''    if x<i64(0):
        return -x
    if x==i64(0):
        return i64(17)
    return x+i64(3)
'''),
    'branches': ('i64', '''    if x<i64(0):
        return -x
    else:
        return x+i64(3)
'''),
    'live_init': ('i64', '''    if x<i64(0):
        return -x
    else:
        value=x+i64(7)
    return value
'''),
    'range_return': ('i64', '''    value=i64(0)
    for step in range(5):
        value+=i64(step)
        if value>x:
            return value
    return value+i64(11)
'''),
    'while_return': ('i64', '''    value=i64(0)
    step=0
    while step<5:
        value+=i64(step)
        if value>x:
            return value
        step+=1
    return value+i64(11)
'''),
    'nested_return': ('i64', '''    value=i64(0)
    for outer in constants.VALUES:
        for inner in (3,5):
            value+=i64(outer*inner)
            if value>x:
                return value
    return value+i64(19)
'''),
    'break_return': ('i64', '''    value=x
    for step in (3,7):
        if x<i64(step):
            break
        value+=i64(step)
    else:
        return value+i64(17)
    return value
'''),
    'continue_return': ('i64', '''    value=i64(0)
    for step in (1,2,3,4):
        if step==2:
            continue
        value+=i64(step)
        if x<value:
            return value
    return value+i64(23)
'''),
    'empty_else': ('i64', '''    for step in ():
        return x
    else:
        return x+i64(11)
'''),
    'finite_first': ('i64', '''    for step in (3,7):
        return x+i64(step)
'''),
    'loop_else_return': ('i64', '''    value=x
    for step in constants.VALUES:
        if x<i64(step):
            return value-i64(13)
        value+=i64(step)
    else:
        return value+i64(17)
'''),
    'nested_else_return': ('i64', '''    value=i64(0)
    for outer in constants.VALUES:
        for inner in (3,5):
            if x<i64(inner):
                break
            if x<i64(outer*inner):
                return value+i64(outer)
            value+=i64(inner)
        else:
            if x<i64(19):
                return value+i64(7)
    return value+i64(23)
'''),
    'caller_snapshot': ('i64', '''    x+=i64(7)
    if x<i64(0):
        return -x
    return x+i64(3)
'''),
    'nested_calls': ('i64', '''    value=choose(x)
    if value<i64(0):
        return choose(value+i64(7))
    return choose(value)+choose(x)
'''),
    'argument_mutation': ('i64', '''    value=choose(x)+choose(x+i64(1))
    if value<i64(0):
        return value-x
    return value+x
'''),
    'multiple_sites': ('i64', '''    if x<i64(0):
        return choose(x)+x
    return choose(x)-x
'''),
    'return_expression': ('i64', '''    if x<i64(0):
        return choose(x)*i64(3)+choose(x+i64(1))
    return choose(x)*i64(5)-choose(x-i64(1))
'''),
    'wide_unsigned': ('u64', '''    if x<u64(9223372036854775808):
        return x+u64(1099511627777)
    return x*u64(3)+u64(18446744073709551615)
'''),
    'floating': ('f32', '''    if x<0.0:
        return x*0.5-0.25
    return x*2.0+0.5
'''),
    'mixed_exact': ('f32', '''    value=x
    for coefficient in (2.0,0.5,-1.0):
        value=value*coefficient+coefficient
        if value<0.0:
            return value
    return value+0.25
'''),
}
NAMES = tuple(BODIES)
UTILITY = '''@device
def choose(x:i64)->i64:
    x+=i64(7)
    if x<i64(0):
        return -x
    return x+i64(3)
'''

def configurations():
    constants = HEADER + 'VALUES:Final=(1,2,3)\n'
    helpers = HEADER + 'from . import constants\nraise RuntimeError("helper module must never execute")\n' + UTILITY
    entry = HEADER + 'from library import helpers\nraise RuntimeError("entry must never execute")\n'
    single = constants + UTILITY
    for name, (dtype, body) in BODIES.items():
        fn = '@device\ndef helper_' + name + '(x:' + dtype + ')->' + dtype + ':\n' + body
        helpers += fn
        call = f'helpers.helper_{name}(a[i])'
        if name == 'caller_snapshot':
            call = 'helpers.helper_caller_snapshot(a[i])+a[i]+helpers.helper_caller_snapshot(a[i])'
        if name == 'multiple_sites':
            call = 'helpers.helper_multiple_sites(a[i])+helpers.helper_multiple_sites(a[i])'
        kernel = f'@kernel\ndef {name}(a:Array({dtype}),out:Array({dtype}),n:i32):\n    i=ctaid_x()*128+tid_x()\n    if i<n:\n        out[i]={call}\n'
        entry += kernel
        single += fn.replace('constants.', '') + kernel.replace('helpers.', '')
    base = {'entry.py': entry, 'library/__init__.py': 'raise RuntimeError("package must never execute")\n',
            'library/constants.py': constants, 'library/helpers.py': helpers}
    yield dict(name='imported', files=base, entry='entry.py', explicit=False, remote=False)
    yield dict(name='same_file', files={'entry.py': single}, entry='entry.py', explicit=False, remote=False)
    packaged = {k: v for k, v in base.items() if k != 'entry.py'}
    packaged.update({'kernels/__init__.py': 'raise RuntimeError("entry package must never execute")\n', 'kernels/entry.py': entry})
    yield dict(name='remote_package', files=packaged, entry='kernels/entry.py', explicit=False, remote=True)

def write_tree(root, case):
    for name, text in case['files'].items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root / case['entry']

def signed(x):
    bits = x & MASK
    return bits - (1 << 64) if bits & (1 << 63) else bits

def choose(x):
    value = signed(x + 7)
    return signed(-value if value < 0 else value + 3)

def reference(name, bits):
    x = bits if name in ('wide_unsigned', 'floating', 'mixed_exact') else signed(bits)
    if name == 'classify': return -x if x < 0 else 17 if x == 0 else x + 3
    if name == 'branches': return -x if x < 0 else x + 3
    if name == 'live_init': return -x if x < 0 else x + 7
    if name in ('range_return', 'while_return'):
        for value in (0, 1, 3, 6, 10):
            if value > x: return value
        return 21
    if name == 'nested_return':
        for value in (3, 8, 14, 24, 33, 48):
            if value > x: return value
        return 67
    if name == 'break_return': return x if x < 3 else x + 3 if x < 7 else x + 27
    if name == 'continue_return':
        for value in (1, 4, 8):
            if x < value: return value
        return 31
    if name == 'empty_else': return x + 11
    if name == 'finite_first': return x + 3
    if name == 'loop_else_return':
        value = x
        for c in (1, 2, 3):
            if x < c: return value - 13
            value += c
        return value + 17
    if name == 'nested_else_return':
        value = 0
        for outer in (1, 2, 3):
            for inner in (3, 5):
                if x < inner: break
                if x < outer * inner: return value + outer
                value += inner
            else:
                if x < 19: return value + 7
        return value + 23
    if name == 'caller_snapshot': return choose(x) + x + choose(x)
    if name == 'nested_calls':
        value = choose(x)
        return choose(signed(value + 7)) if value < 0 else choose(value) + choose(x)
    if name == 'argument_mutation':
        value = signed(choose(x) + choose(signed(x + 1)))
        return value - x if value < 0 else value + x
    if name == 'multiple_sites': return 2 * (choose(x) + x if x < 0 else choose(x) - x)
    if name == 'return_expression': return choose(x) * 3 + choose(signed(x + 1)) if x < 0 else choose(x) * 5 - choose(signed(x - 1))
    if name == 'wide_unsigned': return x + 1099511627777 if x < (1 << 63) else x * 3 + MASK
    if name == 'floating': return x * 0.5 - 0.25 if x < 0 else x * 2 + 0.5
    if name == 'mixed_exact':
        value = x
        for c in (2.0, 0.5, -1.0):
            value = value * c + c
            if value < 0: return value
        return value + 0.25
    raise AssertionError(name)

def raw_input(name, seed, lanes):
    if name in ('floating', 'mixed_exact'):
        return struct.pack('<' + str(lanes) + 'f', *[(i - 63) / 8 for i in range(lanes)])
    edges = [0, 1, MASK, 1 << 63, (1 << 63) - 1, 3, 7, 19, MASK - 6, (1 << 63) - 7]
    rng = random.Random(seed)
    return struct.pack('<' + str(lanes) + 'Q', *(edges + [rng.getrandbits(64) for _ in range(lanes - len(edges))]))

def initial(name, lanes):
    floating = name in ('floating', 'mixed_exact')
    return struct.pack('<' + str(lanes) + ('f' if floating else 'Q'), *([-12345.0 if floating else SENTINEL] * lanes))

def expected(name, seed, n, lanes):
    floating = name in ('floating', 'mixed_exact')
    dtype = '<' + str(lanes) + ('f' if floating else 'Q')
    data = struct.unpack(dtype, raw_input(name, seed, lanes))
    values = [reference(name, x) if i < n else -12345.0 if floating else SENTINEL for i, x in enumerate(data)]
    return struct.pack(dtype, *[x if floating else x & MASK for x in values])
