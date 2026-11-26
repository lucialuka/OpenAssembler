"""Observe per-function local frame metadata across sizes and alignments."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from openassembler import oracle
from openassembler.container.elf import Elf
from openassembler.container import nvinfo


if __name__=='__main__':
    original=(HERE/'local.ptx').read_text();rows=[]
    for size,alignment in((32,4),(32,16),(48,16),(64,16),(128,16),(64,32)):
        name=f'local-{size}-a{alignment}';source=HERE/(name+'.ptx')
        source.write_text(original.replace('.align 16 .b8 slots[32]',f'.align {alignment} .b8 slots[{size}]'))
        cubin=HERE/(name+'.cubin');oracle.ptxas(source,cubin,'sm_80','-O0')
        disassembly=oracle.disasm_text(cubin);(HERE/(name+'.sass.txt')).write_text(disassembly)
        elf=Elf.parse(cubin.read_bytes())
        frames=[dict(name=r.name,values=nvinfo.u32s(r))for r in nvinfo.parse(elf.section('.nv.info').data)if r.name in('FRAME_SIZE','MIN_STACK_SIZE','MAX_STACK_SIZE')]
        rows.append(dict(declared_size=size,declared_alignment=alignment,frames=frames,
                         stack_updates=[i.text for i in oracle.parse_listing(disassembly)if i.text.startswith('IADD3 R1,')],
                         source=source.name,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cubin=cubin.name,cubin_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest()))
    result=dict(schema=1,gap='G13',scope='Toolkit-only independent stack frame layout observations; no native/hardware admission',results=rows)
    (HERE/'layout-sweep-20261123.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
