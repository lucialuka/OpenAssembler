"""A100 complete typed Python shared-wide values, padding, resources and ownership."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import modal
ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
app=modal.App('oa-python-shared-wide')
image=modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04',add_python='3.12')

@app.function(image=image,gpu='A100-40GB',timeout=1800,retries=0,max_containers=1)
def execute(payload):
    import base64,ctypes as C,io,os,struct,subprocess,sys,tempfile,zipfile
    from dataclasses import asdict
    with tempfile.TemporaryDirectory(prefix='oa-python-shared-')as temporary:
        root=Path(temporary)
        for name,data in payload.items():
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        (root/'env/cuda.sh').chmod(0o700);os.chdir(root)
        sys.path[:0]=[str(root),str(root/'compiler_exp/gap_closure/python_shared_wide')]
        from fixtures import HEADER,cases,values,expected,delta,SENTINEL
        from openassembler import build
        from openassembler.pyfront import compile_file
        from openassembler.gpu.cuda_driver import Driver
        from openassembler.container.elf import Elf
        from openassembler.native_ownership import validate
        from openassembler.backend.storage import local_frame_inventory
        fixtures=list(cases());combined=HEADER+'\n'.join(c['source'][len(HEADER):]for c in fixtures)
        source=root/'sources.py';source.write_text(combined)
        generated,names=compile_file(source);assert names==[c['name']for c in fixtures]
        records=[];blob=io.BytesIO();guard=b'\xa5'*256;lanes=384;error=None;device=None
        driver=Driver();driver.open()
        try:
            device=asdict(driver.info());assert device['cc']=='8.0'and'A100'in device['name']
            with zipfile.ZipFile(blob,'w',zipfile.ZIP_DEFLATED)as archive:
                archive.writestr('sources.py',combined);archive.writestr('generated.ptx',generated)
                for level in('safe','mined','sched','opt'):
                    target=root/level;result=build.compile_paths(source,target,build.Options(opt_level=level));assert result.rc==0,result
                    manifest=json.loads((target/'manifest.json').read_text());entry=manifest['entries'][0]
                    assert(target/entry['ptx']).read_text()==generated and entry['language']=='python'
                    archive.writestr(level+'/manifest.json',json.dumps(manifest,indent=2))
                    for compiler,key in(('oa','cubin'),('ptxas','reference')):
                        binary=(target/entry[key]).read_bytes();elf=Elf.parse(binary)
                        if compiler=='oa':validate(entry,binary)
                        archive.writestr(level+'/'+compiler+'.cubin',binary)
                        archive.writestr(level+'/'+compiler+'.sass.txt',subprocess.check_output(['nvdisasm','-hex','-c',str(target/entry[key])]))
                        module=driver.load(binary)
                        try:
                            for case in fixtures:
                                fn=driver.function(module,case['name']);resources=driver.func_attrs(fn)
                                assert resources['shared_bytes']==case['shared_bytes'],(case['name'],resources)
                                assert elf.section('.nv.shared.'+case['name']).size==case['shared_bytes']
                                assert resources['local_bytes']==0
                                if compiler=='oa':
                                    contract=next(k for k in entry['kernels']if k['name']==case['name'])
                                    assert resources['num_regs']==contract['registers']
                                inp,out=driver.alloc(lanes*8+512),driver.alloc(lanes*32+512)
                                try:
                                    for n in(0,1,259,384):
                                        for seed in(14301,14302):
                                            data=values(seed,lanes);d=delta(case,seed);before=guard+struct.pack('<'+str(lanes)+'Q',*data)+guard
                                            initial=guard+struct.pack('<Q',SENTINEL)*(lanes*4)+guard
                                            driver.htod(inp,before);driver.htod(out,initial)
                                            driver.launch(fn,(3,1,1),(128,1,1),[C.c_uint64(inp+256),C.c_uint64(out+256),C.c_uint64(d&((1<<64)-1)),C.c_uint32(n)])
                                            actual=driver.dtoh(out,len(initial));unchanged=driver.dtoh(inp,len(before));wanted=guard+expected(data,case,n,d)+guard
                                            prefix=f'{level}/{compiler}/{case["name"]}/{n}/{seed}/'
                                            for name,raw in(('before-input',before),('input',unchanged),('output',actual)):archive.writestr(prefix+name+'.bin',raw)
                                            records.append(dict(name=case['name'],level=level,compiler=compiler,n=n,seed=seed,passed=actual==wanted and unchanged==before,
                                                input_sha256=hashlib.sha256(unchanged).hexdigest(),output_sha256=hashlib.sha256(actual).hexdigest(),
                                                cubin_sha256=hashlib.sha256(binary).hexdigest(),resources=resources,local_frame_bytes=local_frame_inventory(binary)[case['name']]))
                                finally:driver.free(inp);driver.free(out)
                        finally:driver.unload(module)
        except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc))
        finally:driver.close()
        report=dict(schema=1,passed=error is None and len(records)==768 and all(r['passed']for r in records),error=error,results=records,device=device,python=sys.version,
                    ptxas=subprocess.check_output(['ptxas','--version'],text=True),source_hashes={name:hashlib.sha256(data).hexdigest()for name,data in payload.items()},
                    scope='Ordinary Python source build, typed signed/unsigned64 shared transfers, mixed alignment/padding, 48KiB boundaries, cross-warp/CTA loops, input/output/tail guards and actual driver resources; no timing')
        return report,base64.b64encode(blob.getvalue()).decode()

@app.local_entrypoint()
def main():
    import base64,sys
    sys.path.insert(0,str(ROOT))
    from openassembler.transport import runtime_files
    payload=runtime_files(native=True)
    payload['openassembler/pyfront/compile.py']=(HERE/'candidate-compile.py').read_bytes()
    payload['openassembler/backend/compile.py']=(HERE/'candidate-backend-compile.py').read_bytes()
    payload['compiler_exp/gap_closure/python_shared_wide/fixtures.py']=(HERE/'fixtures.py').read_bytes()
    harness=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report,encoded=execute.remote(payload)
    archive=HERE/('candidate-a100-'+stamp+'.zip');archive.write_bytes(base64.b64decode(encoded))
    report.update(archive=dict(path=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),harness_sha256=harness)
    path=HERE/('candidate-a100-'+stamp+'.json');path.write_text(json.dumps(report,indent=2)+'\n');print(path,'passed',report['passed'],'launches',len(report['results']))
    if not report['passed']:raise RuntimeError('Python shared-wide hardware gate failed')
