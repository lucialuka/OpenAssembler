"""Compile every ordering probe; simulate shared atomics without claiming inter-CTA ordering."""
import json
from pathlib import Path
import struct
import sys
from collections import Counter

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT),str(ROOT/'compiler_exp/gap_closure/atomics')]
from candidate import install
from probes import publication_source,shared_source
from cases import exchange_chain
from openassembler.build import generate
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing


if __name__=='__main__':
    install();rows=[]
    fixtures=[('publication',publication_source()),('both_acqrel',publication_source('both_acqrel',release='acq_rel',acquire='acq_rel'))]
    fixtures += [(name,shared_source(name,operation=operation,explicit=explicit))
                 for name,operation,explicit in [('shared_add','add',False),('shared_add_explicit','add',True),
                                                 ('shared_exch','exch',False),('shared_exch_explicit','exch',True)]]
    for name,source in fixtures:
        for level in ('safe','mined','sched','opt'):
            row={'name':name,'level':level,'source':source}
            try:
                code,labels,words,exits,regs=generate(parse_module(source).kernels[name],'sm_80',level)
                row.update(encoded_words=len(words),registers=regs)
                if not name.startswith('shared'):
                    row.update(passed=True,simulation='not-run: sequential CTA execution cannot model an inter-CTA handshake')
                else:
                    failures=[]
                    for poison in (None,0xdeadbeef):
                        memory=Memory();address=0x100000000;memory.write_bytes(address-64,b'\xa5'*64+b'\x00'*2320+b'\xa5'*64)
                        for cta in range(4):
                            Machine(load_listing(listing(code,labels)),Launch(grid=(4,1,1),block=(128,1,1),ctaid=(cta,0,0),
                                    params=struct.pack('<Q',address),smem_static=68,poison=poison),memory).run()
                        for cta in range(4):
                            for tid in range(128):
                                if tid%2 and memory.read(address+4*(cta*128+tid),4)!=0x13579bdf:failures.append('false shared predicate')
                            for bin_index in range(17):
                                lanes=[tid for tid in range(bin_index,128,17) if tid%2==0]
                                returns=[memory.read(address+4*(cta*128+tid),4) for tid in lanes]
                                final=memory.read(address+2048+4*(cta*17+bin_index),4)
                                if 'add' in name:
                                    if final!=len(lanes) or Counter(returns)!=Counter(range(len(lanes))):failures.append('shared add ticket/state')
                                elif not exchange_chain(0,final,[cta*128+tid+1 for tid in lanes],returns):failures.append('shared exchange chain')
                        if memory.read(address-64,64)!=int.from_bytes(b'\xa5'*64,'little') or memory.read(address+2320,64)!=int.from_bytes(b'\xa5'*64,'little'):failures.append('guard')
                    row.update(passed=not failures,failures=failures,simulation='normal and poisoned SASS, multi-warp CTA barriers')
            except Exception as exc:row.update(passed=False,error=type(exc).__name__+': '+str(exc))
            rows.append(row)
    result={'schema':1,'scope':'isolated candidate encoding and shared-memory emulation; publication visibility is hardware-only',
            'passed':all(r['passed'] for r in rows),'results':rows}
    path=Path(__file__).with_name('offline-candidate-v2-20261121.json');path.write_text(json.dumps(result,indent=2)+'\n')
    print(path,'passed',result['passed'],[{k:v for k,v in r.items() if k!='source'} for r in rows if not r['passed']][:2])
