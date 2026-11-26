"""Complete shared-byte snapshots and scalar-width values with private guards."""
import hashlib
import json
from pathlib import Path
import random
import struct
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from probes import cases,expected,values,SENTINEL
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing


if __name__=='__main__':
    records=[];guard=b'\xa5'*64;lanes=128;inp,out=0x100010000,0x100000000
    for case in cases():
        for level in('safe','mined','sched','opt'):
            try:
                module=parse_module(case['source']);code,labels,_,_,registers=build.generate(module.kernels[case['name']],'sm_80',level)
                for seed in(13052,13053):
                    data=values(seed,lanes)
                    original=guard+struct.pack('<'+str(len(data))+'I',*data)+guard
                    for n in(0,91,128):
                        reference=guard+expected(data,case,n,lanes)+guard
                        for engine in('ptx','normal','poisoned'):
                            memory=Memory();memory.write_bytes(inp-64,original)
                            memory.write_bytes(out-64,guard+struct.pack('<I',SENTINEL)*(lanes*16)+guard)
                            if engine=='ptx':Interp(module,case['name'],PLaunch(block=(lanes,1,1),params={'output':out,'input':inp,'n':n}),memory).run()
                            else:Machine(load_listing(listing(code,labels)),Launch(block=(lanes,1,1),params=struct.pack('<QQI',out,inp,n),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
                            actual=bytes(memory.read(out-64+i,1)for i in range(len(reference)))
                            assert actual==reference,(case['name'],level,n,seed,engine,next((i,a,b)for i,(a,b)in enumerate(zip(actual,reference))if a!=b))
                            assert bytes(memory.read(inp-64+i,1)for i in range(len(original)))==original
                records.append(dict(name=case['name'],level=level,passed=True,registers=registers,executions=18))
            except Exception as error:records.append(dict(name=case['name'],level=level,passed=False,error=type(error).__name__+': '+str(error)))
            print(records[-1],flush=True)
    result=dict(schema=1,passed=all(row['passed']for row in records),results=records,
                source_hashes={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()for name in('candidate-ra.py','candidate-contracts.py','candidate-encode.py','candidate-compile.py','candidate-templates.json','probes.py')},
                scope='Complete independent shared snapshots, sign extension/upper words, guarded stores/loads, cross-warp ownership, PTX and normal/poisoned SASS, immutable inputs/guards/tails; native/hardware gates remain separate')
    (HERE/'candidate-offline.json').write_text(json.dumps(result,indent=2)+'\n')
