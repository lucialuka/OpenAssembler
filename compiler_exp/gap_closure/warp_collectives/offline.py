"""Compare independent lane arithmetic with candidate PTX/SASS execution."""
import hashlib
import json
from pathlib import Path
import random
import struct
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(HERE)]
from cases import cases,expected
from candidate import install
from openassembler.build import generate
from openassembler.ptx.parse import parse_module
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from openassembler.ptx.interp import run_kernel,Launch as PTXLaunch
from tests.backend_harness import listing


if __name__=='__main__':
    install();rng=random.Random(15052);values=[rng.getrandbits(32)for _ in range(128)]
    rows=[]
    for case in cases():
        module=parse_module(case['source'])
        for level in('safe','mined','sched','opt'):
            try:
                c,l,w,_,_=generate(module.kernels[case['name']],'sm_80',level)
                for delta in (0,1,17,31,32,33,0xffffffff):
                    reference=expected(case,values,delta)
                    for engine in ('sass','poisoned','ptx'):
                        memory=Memory();memory.write_bytes(0x100010000,struct.pack('<128I',*values))
                        if engine=='ptx':run_kernel(module,case['name'],PTXLaunch(block=(128,1,1),params={'output':0x100000000,'input':0x100010000,'delta':delta}),memory)
                        else:Machine(load_listing(listing(c,l)),Launch(block=(128,1,1),params=struct.pack('<QQI',0x100000000,0x100010000,delta),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
                        actual=[memory.read(0x100000000+i*4,4)for i in range(128)]
                        assert actual==reference,(case['name'],level,delta,engine,actual[:32],reference[:32])
                rows.append(dict(name=case['name'],level=level,passed=True,native_sha256=hashlib.sha256(b''.join(x.to_bytes(16,'little')for x in w)).hexdigest()))
            except Exception as exc:rows.append(dict(name=case['name'],level=level,passed=False,error=str(exc)))
    result=dict(schema=1,passed=all(x['passed']for x in rows),results=rows,values=values,
                scope='Independent integer lane reference versus PTX and normal/poisoned SASS; hardware not run, full-warp participation is an explicit precondition')
    (HERE/'candidate-offline.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['passed'],[r for r in rows if not r['passed']][:4])
