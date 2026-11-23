"""Original-settings / relocated-settings compilation and exact A100 outputs."""
import base64
import ctypes as C
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import struct
import sys
import zipfile


def run(source_payload,destination):
    from openassembler import build,project
    from openassembler.remote_build import process
    from openassembler.torchfront.archive import unpack,pack
    from openassembler.native_ownership import validate
    from openassembler.gpu.cuda_driver import Driver
    root=Path(destination);root.mkdir(parents=True,exist_ok=True)
    request={'source':base64.b64encode(pack(source_payload)).decode(),'path':'source','options':asdict(build.Options(no_sass=False,verify=True)),'name':None}
    result=process(request)
    if result.get('error') or result.get('rc')!=0:raise RuntimeError(f'relocated database build failed: {result}')
    output=root/'artifacts';unpack(base64.b64decode(result['artifacts']),output)
    manifest=project.load_manifest(output);assert len(manifest['entries'])==2
    driver=Driver();driver.open();records=[]
    count,threads=67,96;guard=bytes([0xa5])*256
    values=[(i%29-14)*.25 for i in range(threads)]
    input_bytes=struct.pack('<'+'f'*threads,*values)
    try:
        device=asdict(driver.info());assert device['cc']=='8.0' and 'A100' in device['name']
        for entry in manifest['entries']:
            contract=entry['build_contract'];assert contract['submitted_origin']['kind']=='nvcc-compilation-database'
            assert contract['dependencies'] and contract['toolchain']
            kernel=entry['kernels'][0];name=kernel['name'];assert kernel['verify']=='ok'
            scale,bias=(3,5) if name=='configured_three' else (-2,7)
            header_size=len('../include/qualifier.h')+1
            source_size=len('generated_'+('three' if name=='configured_three' else 'negative')+'.cu')+1
            bias += header_size+source_size
            wanted=struct.pack('<'+'f'*count,*[x*scale+bias for x in values[:count]])+bytes([0x3c])*(4*(threads-count))
            binary=(output/entry['cubin']).read_bytes();validate(entry,binary)
            for compiler,key in [('oa','cubin'),('ptxas','reference')]:
                module=driver.load((output/entry[key]).read_bytes())
                pointers=[]
                try:
                    fn=driver.function(module,name);pointers=[driver.alloc(len(input_bytes)+512) for _ in range(2)]
                    driver.htod(pointers[0],guard+input_bytes+guard)
                    driver.htod(pointers[1],guard+bytes([0x3c])*len(input_bytes)+guard)
                    driver.launch(fn,(1,1,1),(threads,1,1),[C.c_uint64(pointers[1]+256),C.c_uint64(pointers[0]+256),C.c_int32(count)])
                    actual=driver.dtoh(pointers[1],len(input_bytes)+512)
                    immutable=driver.dtoh(pointers[0],len(input_bytes)+512)==guard+input_bytes+guard
                    assert actual==guard+wanted+guard and immutable
                    (root/(name+'-'+compiler+'.bin')).write_bytes(actual)
                    records.append({'kernel':name,'compiler':compiler,'elements':threads,'guards':True,'input_unchanged':immutable,
                        'output_sha256':hashlib.sha256(actual).hexdigest(),'cubin_sha256':hashlib.sha256((output/entry[key]).read_bytes()).hexdigest()})
                finally:
                    for pointer in pointers:driver.free(pointer)
                    driver.unload(module)
        report={'schema':1,'passed':True,'python':sys.version,'device':device,'runs':records,
                'scope':'per-unit definitions/includes/forced headers and generated units preserved through ordinary remote worker; OA vs same-PTX ptxas exact outputs; no performance claim',
                'contracts':[e['build_contract'] for e in manifest['entries']], 'source_payload_sha256':{n:hashlib.sha256(v).hexdigest() for n,v in source_payload.items()}}
        (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report
    finally:driver.close()
