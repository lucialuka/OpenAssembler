"""Independent per-thread integer reference versus local PTX/SASS state."""
from pathlib import Path
import json
import struct
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
from cases import cases,values,expected,N,GRID,THREADS,LANES,SENTINEL
from openassembler import build
from openassembler.emu.machine import Launch,Machine,Memory
from openassembler.emu.program import load_listing
from openassembler.ptx.interp import Launch as PTXLaunch,run_kernel
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing


if __name__=='__main__':
    install();rows=[];guard=b'\xa5'*64;original=values(13052)
    for case in cases():
        module=parse_module(case['source'])
        for level in('safe','mined','sched','opt'):
            try:
                code,labels,words,_,regs=build.generate(module.kernels[case['name']],'sm_80',level)
                for engine in('sass','poisoned','ptx'):
                    memory=Memory();out,inp=0x100000000,0x100010000
                    memory.write_bytes(inp-64,guard+struct.pack('<384I',*original)+guard)
                    memory.write_bytes(out-64,guard+struct.pack('<384I',*([SENTINEL]*LANES))+guard)
                    for cta in range(GRID[0]):
                        if engine=='ptx':run_kernel(module,case['name'],PTXLaunch(grid=GRID,block=(THREADS,1,1),ctaid=(cta,0,0),params={'output':out,'input':inp,'n':N}),memory)
                        else:Machine(load_listing(listing(code,labels)),Launch(grid=GRID,block=(THREADS,1,1),ctaid=(cta,0,0),params=struct.pack('<QQI',out,inp,N),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
                    actual=[memory.read(out+4*i,4)for i in range(LANES)]
                    assert actual==expected(original),(case['name'],level,engine,actual[:10],expected(original)[:10])
                    assert [memory.read(inp+4*i,4)for i in range(LANES)]==original
                    for address in(out,inp):
                        for offset in(-64,LANES*4):assert bytes(memory.read(address+offset+i,1)for i in range(64))==guard
                rows.append(dict(name=case['name'],level=level,passed=True,registers=regs))
            except Exception as exc:rows.append(dict(name=case['name'],level=level,passed=False,error=type(exc).__name__+': '+str(exc)))
    result=dict(schema=1,passed=all(r['passed']for r in rows),results=rows,scope='Independent local-state reference, PTX and normal/poisoned SASS; no GPU/memory-safety claim')
    (HERE/'candidate-offline.json').write_text(json.dumps(result,indent=2)+'\n');print(result['passed'],[r for r in rows if not r['passed']][:4])
