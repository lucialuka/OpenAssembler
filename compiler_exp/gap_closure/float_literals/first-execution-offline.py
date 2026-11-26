"""Complete independent output/input/guard checks through PTX and SASS."""
import hashlib
import json
from pathlib import Path
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
from openassembler.emu.machine import Launch,Machine,Memory
from openassembler.emu.program import load_listing
from tests.backend_harness import listing


if __name__=='__main__':
    records=[];guard=b'\xa5'*64;inp,out=0x100010000,0x100000000
    for case in cases():
        module=parse_module(case['source'])
        for level in('safe','mined','sched','opt'):
            try:
                code,labels,words,_,registers=build.generate(module.kernels[case['name']],'sm_80',level)
                for n in(0,19,32):
                    for seed in(13052,13053):
                        data=values(seed,32);original=guard+struct.pack('<32I',*data)+guard
                        reference=expected(data,case,n,32);wanted=guard+struct.pack('<'+str(len(reference))+'I',*reference)+guard
                        for engine in('ptx','normal','poisoned'):
                            memory=Memory();memory.write_bytes(inp-64,original)
                            memory.write_bytes(out-64,guard+struct.pack('<'+str(len(reference))+'I',*([SENTINEL]*len(reference)))+guard)
                            if engine=='ptx':Interp(module,case['name'],PLaunch(block=(32,1,1),params={'output':out,'input':inp,'n':n}),memory).run()
                            else:Machine(load_listing(listing(code,labels)),Launch(block=(32,1,1),params=struct.pack('<QQI',out,inp,n),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
                            actual=bytes(memory.read(out-64+i,1)for i in range(len(wanted)))
                            assert actual==wanted,(case['name'],level,n,seed,engine,next((i,a,b)for i,(a,b)in enumerate(zip(actual,wanted))if a!=b))
                            assert bytes(memory.read(inp-64+i,1)for i in range(len(original)))==original
                records.append(dict(name=case['name'],level=level,passed=True,registers=registers,executions=54))
            except Exception as error:records.append(dict(name=case['name'],level=level,passed=False,error=type(error).__name__+': '+str(error)))
            print(records[-1],flush=True)
    result=dict(schema=1,passed=all(row['passed']for row in records),results=records,source_sha256=hashlib.sha256((HERE/'candidate-parse.py').read_bytes()).hexdigest(),
                scope='Independent literal/arithmetic/guarded loop bits, complete PTX/normal+poisoned SASS outputs and immutable inputs/guards; separate native packaging/A100 gate required')
    (HERE/'execution-offline.json').write_text(json.dumps(result,indent=2)+'\n')
