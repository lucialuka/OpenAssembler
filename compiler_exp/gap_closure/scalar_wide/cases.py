"""Independent modulo-integer/shift/comparison references for G11 wide scalar paths."""
import random
import struct

HEADER = '.version 8.7\n.target sm_80\n.address_size 64\n'
EDGES = [0, 1, 0xffffffff, 0x100000000, 0x100000001, 0x7fffffffffffffff,
         0x8000000000000000, 0xffffffffffffffff, 0xffffffff00000000, 0x8000000080000000]
COUNTS = list(range(66)) + [127, 128, 255, 256, 1024, 0x7fffffff, 0x80000000, 0xffffffff]
LITERALS = [0, 1, 7, 31, 32, 33, 63, 64, 65, 255, 256, 0xffffffff]
N = 527
THREADS = 576
GUARD = b'\xa5' * 256


def rows():
    rng = random.Random(0x473131)
    base = [(a, b, COUNTS[i % len(COUNTS)]) for i, (a, b) in enumerate((a, b) for a in EDGES for b in EDGES)]
    base += [(EDGES[i % len(EDGES)], rng.getrandbits(64), count) for i, count in enumerate(COUNTS)]
    base += [(rng.getrandbits(64), rng.getrandbits(64), rng.getrandbits(32)) for _ in range(90)]
    duplicated = [row for row in base for _ in range(2)]
    return (duplicated + [base[0]] * THREADS)[:THREADS]


ROWS = rows()
INPUT = b''.join(struct.pack('<QQI12x', a, b, count) for a, b, count in ROWS)


def source(body, alias=False):
    return HEADER + '''.visible .entry probe(.param .u64 output,.param .u64 input,.param .u32 n) {
.reg .u64 %input,%output,%offset,%src,%dst;
.reg .b64 %x,%y,%z; .reg .b32 %t,%n,%odd,%count,%a,%b,%r;
.reg .pred %end,%take,%p;
ld.param.u64 %input,[input]; ld.param.u64 %output,[output]; ld.param.u32 %n,[n];
mov.u32 %t,%tid.x; setp.ge.u32 %end,%t,%n; @%end bra DONE;
mul.wide.u32 %offset,%t,32; add.u64 %src,%input,%offset;
mul.wide.u32 %offset,%t,16; add.u64 %dst,%output,%offset;
ld.global.b64 %x,[%src]; ld.global.b64 %y,[%src+8]; ld.global.u32 %count,[%src+16];
ld.global.b32 %a,[%src]; ld.global.b32 %b,[%src+8];
and.b32 %odd,%t,1; setp.eq.u32 %take,%odd,0;
''' + body + '\nDONE: ret; }\n'


def signed(value, width):
    return value - (1 << width) if value >> (width - 1) else value


def expected(op, width, count, x, y):
    mask = (1 << width) - 1
    x, y = x & mask, y & mask
    name, *mods = op.split('.')
    if name == 'sub':
        return (x - y) & mask
    if name in ('shl', 'shr'):
        if name == 'shl':
            return (x << count) & mask if count < width else 0
        if mods[-1].startswith('s'):
            return (signed(x, width) >> min(count, width)) & mask
        return x >> min(count, width)
    comparison = {'lo': 'lt', 'ls': 'le', 'hi': 'gt', 'hs': 'ge'}.get(mods[0], mods[0])
    if mods[-1].startswith('s') and mods[0] not in ('lo', 'ls', 'hi', 'hs'):
        x, y = signed(x, width), signed(y, width)
    return int({'eq': x == y, 'ne': x != y, 'lt': x < y, 'le': x <= y, 'gt': x > y, 'ge': x >= y}[comparison])


def cases():
    specs = [(f'sub.{dtype}', 64, None) for dtype in ('s64', 'u64')]
    for dtype in ('b32', 'u32', 's32', 'b64', 'u64', 's64'):
        width = int(dtype[1:])
        for op in (('shl', 'shr') if dtype.startswith('b') else ('shr',)):
            specs += [(op + '.' + dtype, width, count) for count in [None, *LITERALS]]
    specs += [(f'setp.{comparison}.{dtype}', 64, None) for dtype in ('s64', 'u64')
              for comparison in ('eq', 'ne', 'lt', 'le', 'gt', 'ge')]
    specs += [(f'setp.{comparison}.u64', 64, None) for comparison in ('lo', 'ls', 'hi', 'hs')]
    for op, width, literal in specs:
        compare = op.startswith('setp')
        if compare:
            body = f'setp.eq.u32 %p,%t,%t; @%take {op} %p,%x,%y; selp.u32 %r,1,0,%p; st.global.u32 [%dst],%r;'
        else:
            value, other = ('%x', '%y') if width == 64 else ('%a', '%b')
            count_operand = '%count' if literal is None else str(literal)
            right = other if op.startswith('sub') else count_operand
            # Source/destination alias exercises home-register preservation.
            body = f'@%take {op} {value},{value},{right}; st.global.b{width} [%dst],{value};'
        output = bytearray([0x3c] * (THREADS * 16))
        for index, (x, y, count) in enumerate(ROWS[:N]):
            result = (expected(op, width, count if literal is None else literal, x, y)
                      if index % 2 == 0 else 1 if compare else x & ((1 << width) - 1))
            struct.pack_into('<I' if compare or width == 32 else '<Q', output, index * 16, result)
        yield {'name': op.replace('.', '_') + ('_variable' if literal is None else '_count_' + str(literal)),
               'source': source(body), 'input': INPUT, 'expected': bytes(output),
               'contract': {'op': op, 'width': width, 'literal': literal,
                            'count_policy': 'u32 count clamped at element width; signed right shifts sign-fill',
                            'predication': 'even lanes update aliased source; odd lanes retain prior value',
                            'natural_alignment': 8 if width == 64 else 4}}
    casts = {
        'cast_s64_s32_snapshot': ('cvt.s64.s32 %z,%a; @%take mov.b32 %a,99; st.global.b64 [%dst],%z;',
                                 64, lambda x: signed(x & 0xffffffff, 32) & ((1 << 64) - 1)),
        'cast_u64_u32_snapshot': ('cvt.u64.u32 %z,%a; @%take mov.b32 %a,99; st.global.b64 [%dst],%z;',
                                 64, lambda x: x & 0xffffffff),
        'cast_s32_s64_snapshot': ('cvt.s32.s64 %a,%x; @%take ld.global.b64 %x,[%src+8]; st.global.b32 [%dst],%a;',
                                 32, lambda x: x & 0xffffffff),
        'cast_u64_s64_snapshot': ('cvt.u64.s64 %z,%x; @%take ld.global.b64 %x,[%src+8]; st.global.b64 [%dst],%z;',
                                 64, lambda x: x),
        'cast_u32_s32_snapshot': ('cvt.u32.s32 %r,%a; @%take mov.b32 %a,99; st.global.b32 [%dst],%r;',
                                 32, lambda x: x & 0xffffffff),
    }
    for name, (body, width, reference) in casts.items():
        output = bytearray([0x3c] * (THREADS * 16))
        for index, (x, _, _) in enumerate(ROWS[:N]):
            struct.pack_into('<Q' if width == 64 else '<I', output, index * 16, reference(x))
        yield {'name': name, 'source': source(body), 'input': INPUT, 'expected': bytes(output),
               'contract': {'op': name, 'width': width, 'literal': None,
                            'count_policy': 'not a shift', 'predication': 'cast snapshots survive later guarded source mutation'}}
