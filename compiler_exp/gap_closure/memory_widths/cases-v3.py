"""Independent byte-level G12/G11 fixtures, natural alignment, tails and guards."""
import random
import struct

HEADER = '.version 8.7\n.target sm_80\n.address_size 64\n'
N = 67
THREADS = 96
SLOT = 16
GUARD = bytes([0xa5]) * 256


def source(body):
    return HEADER + '''.visible .entry probe(.param .u64 output,.param .u64 input,.param .u32 n) {
.reg .u64 %input,%output,%offset,%src,%dst; .reg .b64 %wide;
.reg .b32 %t,%n,%odd,%a,%b,%c,%d,%v,%count;
.reg .pred %end,%take,%p,%q,%r;
ld.param.u64 %input,[input]; ld.param.u64 %output,[output]; ld.param.u32 %n,[n];
mov.u32 %t,%tid.x; setp.ge.u32 %end,%t,%n; @%end bra DONE;
mul.wide.u32 %offset,%t,16; add.u64 %src,%input,%offset; add.u64 %dst,%output,%offset;
and.b32 %odd,%t,1; setp.eq.u32 %take,%odd,0;
''' + body + '\nDONE: ret; }\n'


def data():
    rng = random.Random(0x473132)
    edges = [0, 1, 0x7f, 0x80, 0xff, 0x7fff, 0x8000, 0xffff,
             0x7fffffff, 0x80000000, 0xffffffff, 0x7f800000, 0x7fc00001]
    return b''.join(struct.pack('<4I', *(edges[(t + i) % len(edges)] if t < len(edges)
                                      else rng.getrandbits(32) for i in range(4))) for t in range(THREADS))


def cases():
    payload = data()
    for dtype in ('b32', 'u32', 's32', 'f32'):
        body = f'''mov.b32 %a,0x13579bdf; mov.b32 %b,0x2468ace0;
@%take ld.global.v2.{dtype} {{%a,%b}},[%src];
st.global.v2.b32 [%dst],{{%a,%b}};'''
        expected = bytearray([0x3c] * len(payload))
        for t in range(N):
            expected[t * 16:t * 16 + 8] = payload[t * 16:t * 16 + 8] if t % 2 == 0 else struct.pack('<2I', 0x13579bdf, 0x2468ace0)
        yield {'name': 'load_v2_' + dtype, 'source': source(body), 'input': payload, 'expected': bytes(expected),
               'contract': {'op': 'ld.global.v2.' + dtype, 'bytes': 8, 'alignment': 8,
                            'predication': 'even lanes load; odd lanes preserve initial vector values'}}
        body = f'ld.global.v4.b32 {{%a,%b,%c,%d}},[%src]; @%take st.global.v4.{dtype} [%dst],{{%a,%b,%c,%d}};'
        expected = bytearray([0x3c] * len(payload))
        for t in range(0, N, 2):
            expected[t * 16:t * 16 + 16] = payload[t * 16:t * 16 + 16]
        yield {'name': 'store_v4_' + dtype, 'source': source(body), 'input': payload, 'expected': bytes(expected),
               'contract': {'op': 'st.global.v4.' + dtype, 'bytes': 16, 'alignment': 16,
                            'predication': 'even lanes store; odd lanes preserve poisoned bytes'}}
    for dtype, width in [('b8', 1), ('u8', 1), ('s8', 1), ('b16', 2), ('u16', 2), ('s16', 2),
                         ('b64', 8), ('u64', 8), ('s64', 8), ('f64', 8)]:
        wide = width == 8
        body = (f'ld.global.{"b64" if wide else "b32"} %{"wide" if wide else "a"},[%src];\n'
                f'@%take st.global.{dtype} [%dst+{width}],%{"wide" if wide else "a"};')
        expected = bytearray([0x3c] * len(payload))
        for t in range(0, N, 2):
            expected[t * 16 + width:t * 16 + width * 2] = payload[t * 16:t * 16 + width]
        yield {'name': 'store_' + dtype, 'source': source(body), 'input': payload, 'expected': bytes(expected),
               'contract': {'op': 'st.global.' + dtype, 'bytes': width, 'alignment': width,
                            'predication': 'even lanes store at naturally aligned nonzero offset'}}
    snapshots = {
        'pack_snapshot': ('''mov.b32 %a,0x11111111; mov.b32 %b,0x22222222;
mov.b64 %wide,{%a,%b}; @%take mov.b32 %a,0x33333333; st.global.u64 [%dst],%wide;''',
                          lambda t: struct.pack('<2I', 0x11111111, 0x22222222)),
        'unpack_snapshot': ('''ld.global.b64 %wide,[%src]; mov.b64 {%a,%b},%wide;
@%take ld.global.b64 %wide,[%src+8]; st.global.v2.u32 [%dst],{%a,%b};''',
                            lambda t: payload[t * 16:t * 16 + 8]),
        'wide_guarded_load': ('''mov.b64 %wide,0x2222222211111111;
@%take ld.global.b64 %wide,[%src]; st.global.u64 [%dst],%wide;''',
                              lambda t: payload[t * 16:t * 16 + 8] if t % 2 == 0 else struct.pack('<2I', 0x11111111, 0x22222222)),
    }
    for name, (body, reference) in snapshots.items():
        expected = bytearray([0x3c] * len(payload))
        for t in range(N):
            expected[t * 16:t * 16 + 8] = reference(t)
        yield {'name': name, 'source': source(body), 'input': payload, 'expected': bytes(expected),
               'contract': {'op': 'mov.b64 pack/unpack and scalar64 store', 'bytes': 8, 'alignment': 8,
                            'predication': name}}
    predicates = {
        'copy_snapshot': ('''setp.lt.u32 %p,%t,17; mov.pred %q,%p; setp.ge.u32 %p,%t,31;
selp.u32 %a,1,0,%q; st.global.u32 [%dst],%a;''', lambda t: t < 17),
        'guarded_copy': ('''setp.lt.u32 %p,%t,17; setp.ge.u32 %q,%t,31;
@%take mov.pred %q,%p; selp.u32 %a,1,0,%q; st.global.u32 [%dst],%a;''',
                         lambda t: t < 17 if t % 2 == 0 else t >= 31),
        'branch_join': ('''setp.lt.u32 %q,%t,17; @%take bra JOIN;
setp.ge.u32 %p,%t,31; mov.pred %q,%p;
JOIN: selp.u32 %a,1,0,%q; st.global.u32 [%dst],%a;''',
                        lambda t: t < 17 if t % 2 == 0 else t >= 31),
        'loop_carried': ('''setp.eq.u32 %p,%t,%t; mov.pred %q,%p; mov.u32 %count,0;
LOOP: not.pred %q,%q; add.u32 %count,%count,1; setp.lt.u32 %r,%count,3; @%r bra LOOP;
selp.u32 %a,1,0,%q; st.global.u32 [%dst],%a;''', lambda t: False),
        'self_guarded': ('''setp.lt.u32 %q,%t,17; setp.ge.u32 %p,%t,31;
@%q mov.pred %q,%p; selp.u32 %a,1,0,%q; st.global.u32 [%dst],%a;''', lambda t: False),
    }
    for name, (body, reference) in predicates.items():
        expected = bytearray([0x3c] * len(payload))
        for t in range(N):
            expected[t * 16:t * 16 + 4] = struct.pack('<I', int(reference(t)))
        yield {'name': 'predicate_' + name, 'source': source(body), 'input': payload, 'expected': bytes(expected),
               'contract': {'op': 'mov.pred and predicate assignment', 'bytes': 4, 'alignment': 4,
                            'predication': name}}
