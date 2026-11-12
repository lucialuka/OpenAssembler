"""Small ELF/ownership fixtures for launch-planning tests without CUDA tools.

These containers are structural unit-test data, not executable GPU test kernels.
Actual builder/ownership integration uses ptxas and real OA-generated words.
"""
from __future__ import annotations
from openassembler.container.elf import Elf,Section,EM_CUDA,SHT_NULL,SHT_PROGBITS,SHT_STRTAB
from openassembler.native_ownership import record


def attach_native_fixture(out_dir,entry):
    names=['','.shstrtab']+['.text.'+k['name'] for k in entry['kernels']]
    strings=b'';offsets=[]
    for name in names:
        offsets.append(len(strings));strings+=name.encode()+b'\0'
    sections=[Section('',SHT_NULL,0,0,0,0,0,0,1,0,0,b''),
              Section('.shstrtab',SHT_STRTAB,0,0,64,len(strings),0,0,1,0,offsets[1],strings)]
    for i,kernel in enumerate(entry['kernels']):
        data=(b'\x18\x79'+b'\0'*14)
        sections.append(Section('.text.'+kernel['name'],SHT_PROGBITS,6,0,128+i*16,len(data),0,16<<24,16,0,offsets[i+2],data))
    ident=b'\x7fELF\x02\x01\x01'+b'\0'*9
    for section in sections[1:]:section.orig_size=0
    elf=Elf(ident,2,EM_CUDA,1,0,80,64,56,64,1,0,0,sections,[])
    binary=elf.serialize()
    path=out_dir/entry['cubin'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(binary)
    # Only native fixture entries receive a ledger; mixed/vendor tests must fail.
    if all(k.get('compiled_by')=='openassembler' for k in entry['kernels']):
        entry['native_ownership']=record(binary,entry['kernels'],ptx_sha256='0'*64)
    return entry


def save_native_manifest(out_dir,root,arch,opt,entries):
    from openassembler import project
    return project.save_manifest(out_dir,root,arch,opt,
                                 [attach_native_fixture(out_dir,e) for e in entries])
