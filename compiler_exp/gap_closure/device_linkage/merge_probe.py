"""Isolated source-level two-module resolution hypothesis, not production linking."""
import hashlib,json,re,struct,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from openassembler import build,oracle
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import Interp,Launch
from openassembler.emu.machine import Memory

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
    entry=(HERE/'entry.ptx').read_text();library=(HERE/'helper.ptx').read_text()
    # This exact observation has one external function and one visible library
    # definition. Match that source shape explicitly; it is not a general parser
    # or a guessed object relocation/linkage implementation.
    assert len(re.findall(r'\.extern\s+\.func\b',entry))==1
    assert len(re.findall(r'\.visible\s+\.func\b',library))==1 and ' helper('in library
    header='\n'.join(line for line in library.splitlines()if not line.startswith(('.version','.target','.address_size')))
    merged=entry.replace('.extern .func','.visible .func')+'\n'+header
    path=HERE/'candidate-merged.ptx';path.write_text(merged);module=parse_module(merged);kernel=module.kernels['linked_probe']
    assert module.inlined_calls=={'linked_probe':{'helper':1}}
    rows=[];inp,out=0x100010000,0x100000000;data=[0,1,0xffffffff,0x80000000]+list(range(124));guard=b'\xa5'*64
    for n in(0,1,91,128):
        memory=Memory();original=guard+struct.pack('<128I',*data)+guard;memory.write_bytes(inp-64,original);memory.write_bytes(out-64,guard+struct.pack('<128I',*([0x13579bdf]*128))+guard)
        params={kernel.params[0][1]:out,kernel.params[1][1]:inp,kernel.params[2][1]:n}
        Interp(module,'linked_probe',Launch(block=(128,1,1),params=params),memory).run()
        expected=[(v*17+9)&0xffffffff if i<n else 0x13579bdf for i,v in enumerate(data)]
        assert bytes(memory.read(out-64+i,1)for i in range(640))==guard+struct.pack('<128I',*expected)+guard
        assert bytes(memory.read(inp-64+i,1)for i in range(640))==original
        rows.append(dict(n=n,passed=True))
    for level in('safe','mined','sched','opt'):
        code,labels,words,exits,registers=build.generate(kernel,'sm_80',level);rows.append(dict(level=level,passed=True,registers=registers,words_sha256=sha(b''.join(w.to_bytes(16,'little')for w in words))))
    binary=HERE/'candidate-merged.cubin';tool=oracle.run_tool('ptxas','-arch=sm_80','-o',oracle.tool_path(binary),oracle.tool_path(path),check=False)
    (HERE/'candidate-merged.tool.log').write_text(tool.stdout+tool.stderr);assert tool.returncode==0
    (HERE/'candidate-merged.sass').write_text(oracle.disasm_text(binary))
    result=dict(schema=1,passed=True,results=rows,entry_sha256=sha(entry.encode()),library_sha256=sha(library.encode()),merged_sha256=sha(merged.encode()),control_cubin_sha256=sha(binary.read_bytes()),script_sha256=sha(Path(__file__).read_bytes()),scope='Exact two-module source-resolution hypothesis and independent guarded CPU outputs plus encoding/toolkit acceptance; not production linking, original linked-binary comparison or hardware execution')
    (HERE/'merge-probe.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed isolated merge/CPU/encoding/toolkit observations; original-linked hardware gate not run')
