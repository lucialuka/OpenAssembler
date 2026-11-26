"""Original adjacent narrow scalar entries and independent whole-bit ABI references."""
import random
import struct

HEADER = '.version 8.0\n.target sm_80\n.address_size 64\n'
TYPES = ('b8', 'u8', 's8', 'b16', 'u16', 's16')
SENTINEL = 0x13579bdf


def definitions():
    for dtype in TYPES:
        width = int(dtype[1:])//8
        for phase in (range(4) if width == 1 else (0, 2)):
            yield dict(name=f'{dtype}_byte{phase}', dtype=dtype, phase=phase, mode='plain', offset=0)
        for mode in ('guard', 'loop', 'alias'):
            yield dict(name=f'{dtype}_{mode}', dtype=dtype, phase=1 if width == 1 else 2, mode=mode, offset=0)
        for offset in (0, 2, 4, 6):
            yield dict(name=f'{dtype}_slice{offset}', dtype=dtype, phase=0, mode='slice', offset=offset)


CASES = tuple(definitions())
NAMES = tuple(row['name'] for row in CASES)
assert len(NAMES) == 60


def fields(case):
    prefix = [('u8', 'pad'+str(i)) for i in range(case['phase'])]
    return prefix+[(('b64' if case['mode'] == 'slice' else case['dtype']), 'value'),
                   ('u8', 'neighbor'), ('u32', 'salt'), ('u64', 'out'), ('u16', 'last'),
                   ('s64', 'wide'), ('u32', 'n')]


def source(case):
    dtype, mode = case['dtype'], case['mode']
    address = 'value'+('+'+str(case['offset']) if case['offset'] else '')
    body = f''' ld.param.u64 %rd1,[out];
 ld.param.u32 %n,[n];
 mov.u32 %i,%tid.x;
 setp.ge.u32 %p,%i,%n;
 @%p bra done;
 mul.wide.u32 %rd2,%i,32;
 add.u64 %rd1,%rd1,%rd2;
 ld.param.u8 %neighbor,[neighbor];
 ld.param.u32 %salt,[salt];
 ld.param.u16 %last,[last];
 ld.param.u32 %lo,[wide];
 ld.param.u32 %hi,[wide+4];
'''
    if mode == 'guard':
        body += f''' mov.u32 %v,17;
 and.b32 %temp,%i,1;
 setp.ne.u32 %p,%temp,0;
 @%p ld.param.{dtype} %v,[{address}];
'''
    elif mode == 'loop':
        body += f''' mov.u32 %v,0;
 mov.u32 %step,0;
again:
 ld.param.{dtype} %temp,[{address}];
 add.u32 %v,%v,%temp;
 add.u32 %step,%step,1;
 setp.lt.u32 %p,%step,3;
 @%p bra again;
'''
    elif mode == 'alias':
        body += f''' mov.u32 %v,0xdeadbeef;
 ld.param.{dtype} %v,[{address}];
 mov.u32 %saved,%v;
 ld.param.u8 %v,[neighbor];
 add.u32 %v,%v,%saved;
'''
    else:
        body += f' ld.param.{dtype} %v,[{address}];\n'
    body += ''' st.global.u32 [%rd1],%v;
 st.global.u32 [%rd1+4],%neighbor;
 st.global.u32 [%rd1+8],%salt;
 st.global.u32 [%rd1+12],%last;
 st.global.u32 [%rd1+16],%lo;
 st.global.u32 [%rd1+20],%hi;
 add.u32 %temp,%v,%salt;
 st.global.u32 [%rd1+24],%temp;
 st.global.u32 [%rd1+28],%i;
done: ret;
'''
    params = ','.join('.param .'+dtype+' '+name for dtype, name in fields(case))
    return '.visible .entry '+case['name']+'('+params+') {\n .reg .pred %p;\n .reg .u32 %i,%n,%v,%neighbor,%salt,%last,%lo,%hi,%temp,%step,%saved;\n .reg .u64 %rd<3>;\n'+body+'}\n'


def module():
    return HEADER+''.join(source(case) for case in CASES)


def inputs(case, seed, sample, out=0x100000000, n=37):
    rng = random.Random(seed)
    bits = int(case['dtype'][1:])
    edge = [0, 1, (1 << (bits-1))-1, 1 << (bits-1), (1 << bits)-1, rng.getrandbits(bits)][sample]
    raw = rng.getrandbits(64)
    if case['mode'] == 'slice':
        shift = case['offset']*8
        raw = (raw & ~(((1 << bits)-1) << shift)) | (edge << shift)
    else:
        raw = edge
    values = {name: rng.getrandbits(int(dtype[1:])) for dtype, name in fields(case)}
    values.update(value=raw, neighbor=(0xa5+sample)&255, salt=0x91827364 ^ seed ^ sample,
                  last=0xc3d2 ^ sample, wide=0xfedcba9876543210 ^ seed, out=out, n=n)
    return values


def pack(case, values):
    result = bytearray()
    for dtype, name in fields(case):
        width = int(dtype[1:])//8
        result += bytes((-len(result)) % width)
        result += (values[name] & ((1 << (width*8))-1)).to_bytes(width, 'little')
    return bytes(result)


def expected(case, values, lanes):
    bits = int(case['dtype'][1:])
    raw = (values['value'] >> (case['offset']*8)) & ((1 << bits)-1)
    x = raw-(1 << bits) if case['dtype'].startswith('s') and raw & (1 << (bits-1)) else raw
    result = []
    for lane in range(lanes):
        if lane >= values['n']:
            result.extend([SENTINEL]*8)
            continue
        mode = case['mode']
        selected = (x if lane & 1 else 17) if mode == 'guard' else x*3 if mode == 'loop' else x+values['neighbor'] if mode == 'alias' else x
        result.extend(v & 0xffffffff for v in [selected, values['neighbor'], values['salt'], values['last'],
                                               values['wide'], values['wide'] >> 32,
                                               selected+values['salt'], lane])
    return struct.pack('<'+str(len(result))+'I', *result)


def initial(lanes):
    return struct.pack('<'+str(lanes*8)+'I', *([SENTINEL]*(lanes*8)))
