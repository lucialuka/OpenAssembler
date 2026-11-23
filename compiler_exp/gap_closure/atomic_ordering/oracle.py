"""Inspect frozen toolkit output for semantic differences without changing code."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from openassembler.emu.program import load_listing
from openassembler.isa.word import control


if __name__=='__main__':
    rows=[]
    for name in ('atom_acquire_gpu','atom_relaxed_gpu','atom_acq_rel_sys','fence_acq_rel_gpu','st_release','ld_acquire'):
        path=ROOT/'corpus/sm_80/sass'/('p_'+name+'.sass')
        source=ROOT/'corpus/sm_80/ptx'/('p_'+name+'.ptx')
        listing=path.read_bytes()
        selected=[{'address':hex(i.addr),'text':i.text,'word':f'{i.word:032x}',
                   'control':str(control(i.word))} for i in load_listing(listing.decode()).insns.values()
                  if i.op in ('ATOMG','MEMBAR','ERRBAR','CCTL','LDG','STG')]
        rows.append({'probe':name,'source':source.read_text(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                     'sass_sha256':hashlib.sha256(listing).hexdigest(),'observed_instructions':selected})
    output=Path(__file__).with_name('frozen-oracle-20261121.json')
    output.write_text(json.dumps({'schema':1,'scope':'existing retained CUDA12.8 ptxas observations; not a fresh toolkit/hardware gate',
                                  'results':rows},indent=2)+'\n')
    print(output)
