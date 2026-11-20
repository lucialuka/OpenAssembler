"""Independent integer/FP-bit expectations for accepted scalar PTX contracts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from openassembler.build import generate
from openassembler.ptx.parse import parse_module
from openassembler.emu.machine import Launch, Machine, Memory
from openassembler.emu.program import load_listing
from tools.bench_build import listing

HEADER = '.version 8.7\n.target sm_80\n.address_size 64\n'


def source(op):
    compare = op.startswith('setp')
    body = (f'{op} %p, %a, %b; selp.u32 %v,1,0,%p;' if compare else f'{op} %v, %a, %b;')
    return HEADER + '''.visible .entry probe(.param .u64 out,.param .u32 a,.param .u32 b) {
.reg .u64 %d; .reg .b32 %a,%b,%v; .reg .pred %p;
ld.param.u64 %d,[out]; ld.param.u32 %a,[a]; ld.param.u32 %b,[b];
''' + body + '\nst.global.u32 [%d],%v; ret; }\n'


def signed(x):
    return x if x < 0x80000000 else x - 0x100000000


def expected(op, a, b):
    if op.startswith(('add','sub')):
        value = signed(a) + signed(b) if op.startswith('add') else signed(a) - signed(b)
        if '.sat.' in op:
            value = min(0x7fffffff, max(-0x80000000, value))
        return value & 0xffffffff
    def number(x):
        if '.ftz.' in op and x & 0x7f800000 == 0:
            x &= 0x80000000
        return struct.unpack('<f',struct.pack('<I',x))[0]
    a,b = number(a),number(b)
    return int(a == b)


def cases():
    edges = [0,1,0xffffffff,0x7fffffff,0x80000000,0x7ffffffe,0x80000001]
    rng = random.Random(20261119)
    pairs = [(a,b) for a in edges for b in edges] + [(rng.getrandbits(32),rng.getrandbits(32)) for _ in range(128)]
    for op in ('add.sat.s32','sub.sat.s32','add.s32','sub.s32'):
        yield op,pairs
    bits=[0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,
          0x3f800000,0xbf800000,0x7f800000,0xff800000,0x7fc00001]
    for op in ('setp.eq.ftz.f32','setp.eq.f32'):
        yield op,[(a,b) for a in bits for b in bits]


def run(output):
    rows=[]
    for op,pairs in cases():
        text=source(op);kernel=parse_module(text).kernels['probe']
        for level in ('safe','mined','sched','opt'):
            try:
                code,labels,words,_,registers=generate(kernel,'sm_80',level)
                program=load_listing(listing(code,labels));bad=[]
                for a,b in pairs:
                    truth=expected(op,a,b)
                    for poison in (None,0xdeadbeef):
                        memory=Memory();memory.write(0x100000000,4,0xa5a5a5a5)
                        Machine(program,Launch(block=(1,1,1),params=struct.pack('<QII',0x100000000,a,b),poison=poison),memory).run()
                        actual=memory.read(0x100000000,4)
                        if actual!=truth:bad.append({'a':a,'b':b,'poison':poison,'actual':actual,'expected':truth})
                rows.append({'op':op,'level':level,'pairs':len(pairs),'poisons':2,'mismatches':len(bad),
                             'first_mismatches':bad[:8],'registers':registers,'instructions':len(code),
                             'source_sha256':hashlib.sha256(text.encode()).hexdigest(),
                             'native_words_sha256':hashlib.sha256(b''.join(w.to_bytes(16,'little') for w in words)).hexdigest()})
            except Exception as exc:rows.append({'op':op,'level':level,'error':str(exc)})
    report={'schema':1,'completed_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'Independent bit/integer formulas vs encoded instruction stream emulation; no hardware timing',
            'results':rows,'passed':all(r.get('mismatches')==0 for r in rows)}
    output=Path(output)
    if output.exists():raise ValueError('immutable record exists')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(output,'passed',report['passed'], 'mismatches',sum(r.get('mismatches',0) for r in rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');args=parser.parse_args();run(args.output)
