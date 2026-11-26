"""Separate algebraic full-output reference, poisoned execution and boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import struct
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(HERE)]
from probes import cases,expected,SENTINEL
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.emu.machine import Launch,Memory,Machine
from openassembler.emu.program import load_listing
from tests.backend_harness import listing


def run(candidate):
    if candidate:
        from candidate import install
        install()
    records=[]
    for case in cases():
        for level in('safe','mined','sched','opt'):
            now=time.monotonic();resources={}
            try:
                code,labels,words,_,registers=build.generate(parse_module(case['source']).kernels[case['name']],'sm_80',level,resource_info=resources)
                assert registers<=case['limit']
                rng=random.Random(17052);data=[rng.getrandbits(32)for _ in range(32*case['count'])]
                before=struct.pack('<'+str(len(data))+'I',*data);guard=b'\xa5'*64
                inp,out=0x100010000,0x100000000;output_bytes=32*case['output_words']*4
                for poison in(None,0xdeadbeef):
                    memory=Memory();memory.write_bytes(inp-64,guard+before+guard)
                    memory.write_bytes(out-64,guard+struct.pack('<'+str(output_bytes//4)+'I',*([SENTINEL]*(output_bytes//4)))+guard)
                    Machine(load_listing(listing(code,labels)),Launch(block=(32,1,1),params=struct.pack('<QQI',out,inp,19),poison=poison),memory).run()
                    actual=[memory.read(out+4*i,4)for i in range(output_bytes//4)]
                    assert actual==expected(data,case,19,32),(case['name'],poison,next((i,a,b)for i,(a,b)in enumerate(zip(actual,expected(data,case,19,32)))if a!=b))
                    assert bytes(memory.read(inp+i,1)for i in range(len(before)))==before
                    for address,size in((inp,len(before)),(out,output_bytes)):
                        for offset in(-64,size):assert bytes(memory.read(address+offset+i,1)for i in range(64))==guard
                records.append(dict(name=case['name'],level=level,registers=registers,resources=resources,passed=True,seconds=time.monotonic()-now))
            except Exception as exc:records.append(dict(name=case['name'],level=level,passed=False,error=type(exc).__name__+': '+str(exc)))
            print(records[-1],flush=True)
    result=dict(schema=1,passed=all(r['passed']for r in records),candidate=candidate,results=records,
                allocator_sha256=hashlib.sha256((HERE/('spill.py'if candidate else'baseline-spill.py')).read_bytes()).hexdigest(),
                scope='Complete normal/poisoned integer outputs with guards; baseline/candidate resource effects are recorded independently of hardware/timing')
    (HERE/('candidate-offline.json'if candidate else'baseline.json')).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',action='store_true');run(parser.parse_args().candidate)
