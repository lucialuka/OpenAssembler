"""Observe donor shared reservation/layout for the exact typed frontend candidate."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from fixtures import cases
from openassembler.pyfront import compile_file
from openassembler import oracle
from openassembler.container.elf import Elf
from openassembler.backend.ra import Selector
from openassembler.ptx.parse import parse_module

def sha(data):return hashlib.sha256(data).hexdigest()

if __name__=='__main__':
    records=[];base=HERE/'oracle';base.mkdir(exist_ok=True)
    for case in cases():
        source=base/(case['name']+'.py');source.write_text(case['source']);ptx,_=compile_file(source);path=source.with_suffix('.ptx');path.write_text(ptx)
        native=source.with_suffix('.cubin')
        result=oracle.run_tool('ptxas','-arch=sm_80','-o',oracle.tool_path(native),oracle.tool_path(path),check=False)
        source.with_suffix('.tool.log').write_text(result.stdout+result.stderr)
        assert result.returncode==0,(case['name'],result.stdout,result.stderr)
        binary=native.read_bytes();elf=Elf.parse(binary)
        sections=[dict(name=s.name,type=s.type,flags=s.flags,size=s.size,align=s.align,info=s.info,link=s.link)for s in elf.sections if s.name.startswith('.nv.shared.')]
        syms=[dict(name=s.name,value=s.value,size=s.size,section=elf.sections[s.shndx].name)for s in elf.symbols if s.shndx<len(elf.sections)and elf.sections[s.shndx].name.startswith('.nv.shared.')]
        listing=oracle.disasm_text(native);source.with_suffix('.sass').write_text(listing)
        kernel=parse_module(ptx).kernels[case['name']];selector=Selector(kernel)
        row=dict(name=case['name'],source_sha256=sha(source.read_bytes()),ptx_sha256=sha(ptx.encode()),cubin_sha256=sha(binary),oa_shared_bytes=selector.smem_bytes,oa_offsets=selector.sym_shared,donor_sections=sections,donor_symbols=syms)
        records.append(row);print(row,flush=True)
    (HERE/'oracle-probe.json').write_text(json.dumps(dict(schema=1,results=records,frontend_sha256=sha((HERE/'candidate-compile.py').read_bytes()),script_sha256=sha(Path(__file__).read_bytes()),scope='Exact source/toolkit/cubin shared section observations; execution and scheduling remain separate'),indent=2)+'\n')
