"""Diagnostic instruction interleaving with checked global ranges; no ordering model."""
import json
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT)]
from candidate import install
from probes import EPOCHS,WORDS,publication_source
from openassembler.build import generate
from openassembler.emu import ops
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing


class CheckedMemory(Memory):
    def __init__(self,ranges):super().__init__();self.ranges=ranges
    def check(self,address,size):
        if not any(start<=address and address+size<=end for start,end in self.ranges):
            raise ValueError(f'global memory outside ranges: {address:#x}+{size}')
    def read(self,address,size):self.check(address,size);return super().read(address,size)
    def write(self,address,size,value):self.check(address,size);return super().write(address,size,value)


if __name__=='__main__':
    install();rows=[]
    source=publication_source();kernel=parse_module(source).kernels['publication']
    pointers=[0x100000000+i*0x100000 for i in range(4)];sizes=[12,WORDS*4,4,4]
    for level in ('safe','mined','sched','opt'):
        code,labels,words,_,_=generate(kernel,'sm_80',level);program=load_listing(listing(code,labels))
        memory=CheckedMemory([(address-256,address+size+256) for address,size in zip(pointers,sizes)])
        for address,size in zip(pointers,sizes):memory.write_bytes(address,b'\x00'*size)
        machines=[Machine(program,Launch(grid=(2,1,1),block=(32,1,1),ctaid=(cta,0,0),params=struct.pack('<4Q',*pointers)),memory) for cta in range(2)]
        status={'level':level,'source':source,'passed':False};steps=0
        try:
            while any(not t.done for machine in machines for warp in machine.warps for t in warp.lanes):
                for machine in machines:
                    for warp in machine.warps:
                        live=[t for t in warp.lanes if not t.done]
                        if not live:continue
                        pc=min(t.pc for t in live);group=[t for t in live if t.pc==pc];insn=program.insns[pc]
                        ops.execute(machine,warp,group,insn);machine.steps+=1;steps+=1
                        if steps>1000000:raise RuntimeError('diagnostic interleaving step limit')
            status.update(passed=memory.read(pointers[0],4)==memory.read(pointers[0]+4,4)==0,
                          output=[memory.read(pointers[0]+4*i,4) for i in range(3)],flag=memory.read(pointers[2],4),ack=memory.read(pointers[3],4))
        except Exception as exc:status.update(error=type(exc).__name__+': '+str(exc),steps=steps)
        rows.append(status)
    report={'schema':1,'scope':'single prescribed instruction interleaving with bounded global addresses, no hardware/cache/ordering semantics',
            'passed':all(r['passed'] for r in rows),'results':rows}
    output=Path(__file__).with_name('interleaved-after-20261123.json');output.write_text(json.dumps(report,indent=2)+'\n')
    print(output,report['passed'],[{k:v for k,v in r.items() if k!='source'} for r in rows])
