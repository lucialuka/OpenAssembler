"""Independent negative proof for packed/unpacked mutable-home snapshots.

Always replay the preserved pre-repair backend, even after production is fixed.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import struct
import sys
import types

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(HERE)]
from cases import source
from openassembler import build
import openassembler.backend as backend
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import run_kernel,Launch as PTXLaunch
from tests.backend_harness import listing


def run():
    raw=(HERE/'reviewer-before-ra.py').read_bytes()
    module=types.ModuleType('openassembler.backend.ra')
    module.__file__=str(HERE/'reviewer-before-ra.py');module.__package__='openassembler.backend'
    sys.modules[module.__name__]=module;backend.ra=module
    exec(compile(raw,module.__file__,'exec'),module.__dict__)
    bodies={
       'packed_snapshot':'''mov.b32 %a,0x11111111; mov.b32 %b,0x22222222;
mov.b64 %wide,{%a,%b};
@%take mov.b32 %a,0x33333333;
st.global.u64 [%dst],%wide;''',
       'unpacked_snapshot':'''ld.global.b64 %wide,[%src]; mov.b64 {%a,%b},%wide;
@%take ld.global.b64 %wide,[%src+8];
st.global.v2.u32 [%dst],{%a,%b};'''}
    report={'scope':'Independent retained wrong-result repro before snapshot repair; no GPU or performance claim',
            'python':platform.python_version(),'before_ra_sha256':hashlib.sha256(raw).hexdigest(),'cases':[]}
    payload=struct.pack('<4I',1,2,3,4)*4
    for name,body in bodies.items():
        ptx=source(body);parsed=parse_module(ptx)
        def fresh():
            mem=Memory();mem.write_bytes(0x10000,payload);return mem
        def words(mem):return [f'{mem.read(0x20000+16*t,8):016x}' for t in range(4)]
        ref=fresh();run_kernel(parsed,'probe',PTXLaunch(block=(4,1,1),params={'input':0x10000,'output':0x20000,'n':4}),ref)
        expected=words(ref)
        independent=['2222222211111111' if name=='packed_snapshot' else '0000000200000001']*4
        assert expected==independent
        row={'name':name,'ptx':ptx,'input_hex':payload.hex(),'expected_u64_hex':expected,
             'expected_reference':'Independent integer bit packing and PTX interpreter agree', 'threads':4,'levels':[]}
        for level in ('safe','opt'):
            code,labels,native,_,regs=build.generate(parsed.kernels['probe'],'sm_80',level)
            mem=fresh();Machine(load_listing(listing(code,labels)),Launch(block=(4,1,1),params=struct.pack('<QQI',0x20000,0x10000,4),param_base=0x160),mem).run()
            actual=words(mem);assert actual!=expected
            row['levels'].append({'level':level,'actual_u64_hex':actual,'mismatch_lanes':[i for i,(a,b) in enumerate(zip(actual,expected)) if a!=b],
                                  'native_text_sha256':hashlib.sha256(b''.join(w.to_bytes(16,'little') for w in native)).hexdigest(),'registers':regs,'status':'failed'})
        report['cases'].append(row)
    target=HERE/'reviewer-snapshot-negative.json'
    if target.exists():raise FileExistsError('Preserve the original negative evidence')
    target.write_text(json.dumps(report,indent=2)+'\n');print(target)

if __name__=='__main__':run()
