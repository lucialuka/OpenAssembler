"""Independent bit references and legal typed PTX for scalar wide selection."""
import random
import struct
from pathlib import Path

HEADER = '.version 8.7\n.target sm_80\n.address_size 64\n'
SENTINEL = 0x13579bdf2468ace0
CHOICE = 0x8123456776543210
OLD = 0xdeadbeef13579bdf
MASK = (1 << 64) - 1


def cases():
    for dtype in ('b64', 'u64', 's64', 'f64'):
        for mode in ('register', 'immediate', 'aliased', 'aliased_right',
                     'false_guard', 'negated', 'parameter', 'left_immediate',
                     'both_immediate'):
            a, b, dest, predicate = '%a', '%b', '%r', '%take'
            initial = f'mov.b64 %r,{OLD:#x};'
            if mode in ('immediate', 'both_immediate'):
                b = '0dfff8123456789abc' if dtype == 'f64' else '-17'
            if mode in ('left_immediate', 'both_immediate'):
                a = '0d8000000000000000' if dtype == 'f64' else '0x8000000000000000'
            if mode in ('aliased', 'aliased_right'):
                dest = '%a' if mode == 'aliased' else '%b'
                initial = ''
            if mode == 'negated':
                predicate = '!%take'
            if mode == 'parameter':
                a = '%choice'
            instruction = f'selp.{dtype} {dest},{a},{b},{predicate};'
            if mode == 'false_guard':
                instruction = '@%writer ' + instruction
            value_type = 'f64' if dtype == 'f64' else 'b64'
            mutations = ''.join(f'mov.b64 {reg},0;' for reg in ('%a', '%b', '%choice') if reg != dest)
            source = HEADER + f'''.entry probe(.param .u64 out,.param .u64 input,.param .u32 n,.param .b64 choice){{
.reg .u64 %out,%in,%off,%src,%dst;
.reg .{value_type} %a,%b,%r,%choice;
.reg .u32 %i,%n,%lo,%block;.reg .pred %take,%writer,%active;
ld.param.u64 %out,[out];ld.param.u64 %in,[input];ld.param.u32 %n,[n];ld.param.b64 %choice,[choice];
mov.u32 %i,%tid.x;mov.u32 %block,%ctaid.x;mad.lo.u32 %i,%block,128,%i;
mul.wide.u32 %off,%i,16;add.u64 %src,%in,%off;ld.global.b64 %a,[%src];ld.global.b64 %b,[%src+8];
and.b32 %lo,%i,1;setp.ne.u32 %take,%lo,0;and.b32 %lo,%i,2;setp.ne.u32 %writer,%lo,0;
{initial}{instruction}{mutations}
mul.wide.u32 %off,%i,8;add.u64 %dst,%out,%off;setp.lt.u32 %active,%i,%n;
@%active st.global.b64 [%dst],{dest};ret;}}'''
            yield dict(name=dtype + '_' + mode, dtype=dtype, mode=mode,
                       source=source, entry='probe')


def expanded_cases():
    yield from cases()
    yield dict(name='cuda_noinline', mode='cuda', entry='entry',
               cuda_source=Path(__file__).with_name('noinline.cu').read_text())


def values(seed, lanes=128):
    rng = random.Random(seed)
    edges = [0, MASK, 1 << 63, 1, 0x7ff0000000000000, 0xfff0000000000000,
             0x7ff0123456789abc, 0xfff8123456789abc, 0x8000000000000001,
             0x00000000ffffffff, 0xffffffff00000000, 0x7fffffffffffffff]
    return [edges[i] if i < len(edges) else rng.getrandbits(64) for i in range(lanes * 2)]


def expected(data, case, n, lanes=128):
    output = [SENTINEL] * lanes
    for i in range(n):
        if case['mode'] == 'cuda':
            x = i
            for _ in range(3):
                x = (x * 1664525 + 1013904223) & 0xffffffff
            output[i] = ((data[i] + 0x123456789abcdef) if x & 1
                         else (data[i] ^ 0xfedcba9876543210)) & MASK
            continue
        left, right = data[2 * i:2 * i + 2]
        mode = case['mode']
        if mode in ('immediate', 'both_immediate'):
            right = 0xfff8123456789abc if case['dtype'] == 'f64' else MASK - 16
        if mode in ('left_immediate', 'both_immediate'):
            left = 1 << 63
        if mode == 'parameter':
            left = CHOICE
        take = bool(i & 1)
        if mode == 'negated':
            take = not take
        output[i] = OLD if mode == 'false_guard' and not i & 2 else left if take else right
    return struct.pack('<' + str(lanes) + 'Q', *output)
