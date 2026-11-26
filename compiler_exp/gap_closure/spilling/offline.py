"""First independent high-pressure scalar spill reference and resource gate."""
from pathlib import Path
import json
import random
import struct
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from probes import source,expected
from openassembler.build import generate
from openassembler.ptx.parse import parse_module
from openassembler.emu.machine import Launch,Memory,Machine
from openassembler.emu.program import load_listing
from tests.backend_harness import listing


if __name__=='__main__':
    records=[]
    for count,limit in((40,32),(96,32),(260,64)):
        for level in('safe','mined','sched','opt'):
            now=time.monotonic();k=parse_module(source(count,limit)).kernels['pressure'];resources={}
            try:
                c,l,w,e,n=generate(k,'sm_80',level,resource_info=resources)
                assert n<=limit and resources['spill_slots']>0
                rng=random.Random(17052);data=[rng.getrandbits(32)for _ in range(32*count)]
                for poison in(None,0xdeadbeef):
                    m=Memory();m.write_bytes(0x100010000,struct.pack('<'+str(len(data))+'I',*data))
                    Machine(load_listing(listing(c,l)),Launch(block=(32,1,1),params=struct.pack('<QQI',0x100000000,0x100010000,32),poison=poison),m).run()
                    actual=[m.read(0x100000000+4*i,4)for i in range(32)]
                    assert actual==expected(data,count,32,32),(count,actual[:4],expected(data,count,32,32)[:4])
                records.append(dict(count=count,limit=limit,level=level,registers=n,resources=resources,passed=True,seconds=time.monotonic()-now))
            except Exception as exc:records.append(dict(count=count,limit=limit,level=level,passed=False,error=type(exc).__name__+': '+str(exc)))
            print(records[-1],flush=True)
    (HERE/'candidate-offline.json').write_text(json.dumps(dict(schema=1,passed=all(r['passed']for r in records),results=records,
        scope='Isolated scalar spill experiment, independent integer loop reference and normal/poisoned SASS; no native packaging or hardware gate'),indent=2)+'\n')
