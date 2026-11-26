"""Define G13 local stack placement from actual donor metadata and SASS."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from openassembler import oracle
from openassembler.build import generate
from openassembler.container.elf import Elf
from openassembler.container import nvinfo
from openassembler.ptx.parse import parse_module


if __name__=='__main__':
    source=HERE/'local.ptx';records=[]
    try:generate(parse_module(source.read_text()).kernels['local_probe'],'sm_80','safe')
    except Exception as exc:baseline=dict(status='rejected',reason=type(exc).__name__+': '+str(exc))
    else:baseline=dict(status='accepted')
    for level in ('-O0','-O3'):
        binary=HERE/('local'+level+'.cubin');oracle.ptxas(source,binary,'sm_80',level)
        listing=oracle.disasm_text(binary);(HERE/('local'+level+'.sass.txt')).write_text(listing)
        elf=Elf.parse(binary.read_bytes());metadata={}
        for section in elf.sections:
            if section.name.startswith('.nv.info'):
                metadata[section.name]=[dict(name=r.name,format=r.fmt,raw=r.data.hex(),u32=nvinfo.u32s(r)if r.fmt==4 and len(r.data)%4==0 else None)for r in nvinfo.parse(section.data)]
        records.append(dict(flags=[level],binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),metadata=metadata,
                            local_sections=[dict(name=s.name,size=len(s.data),info=s.info,flags=s.flags)for s in elf.sections if 'local'in s.name],
                            instructions=[dict(text=i.text,word=f'{i.word:032x}')for i in oracle.parse_listing(listing)if i.text.startswith(('LDL','STL','MOV R1','IADD3 R1'))]))
    result=dict(schema=1,gap='G13',baseline=baseline,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                scope='Toolkit-only local stack/data layout observation; no native candidate, GPU execution or promotion',results=records)
    (HERE/'oracle-20261123.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
