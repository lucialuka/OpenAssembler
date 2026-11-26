"""A100 complete-byte gate for full-warp shuffle variants and reconvergence."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import modal

ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
app=modal.App('oa-gap-warp-shuffles')
image=modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04',add_python='3.12')


@app.function(image=image,gpu='A100-40GB',timeout=1800,retries=0,max_containers=1)
def execute(payload):
    import base64,ctypes as C,io,os,random,struct,subprocess,sys,tempfile,zipfile
    from dataclasses import asdict
    with tempfile.TemporaryDirectory(prefix='oa-g15-')as temporary:
        root=Path(temporary)
        for name,data in payload.items():
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        (root/'env/cuda.sh').chmod(0o700);os.chdir(root)
        sys.path[:0]=[str(root),str(root/'compiler_exp/gap_closure/warp_collectives')]
        from cases import HEADER,cases,expected
        from openassembler import build
        from openassembler.gpu.cuda_driver import Driver
        from openassembler.native_ownership import validate
        fixtures=list(cases());combined=HEADER+'\n'.join(case['source'][len(HEADER):]for case in fixtures)
        source=root/'shuffle.ptx';source.write_text(combined)
        records=[];blob=io.BytesIO();guard=b'\xa5'*256;sentinel=0x13579bdf
        driver=Driver();driver.open()
        try:
            device=asdict(driver.info());assert device['cc']=='8.0'and'A100'in device['name']
            with zipfile.ZipFile(blob,'w',zipfile.ZIP_DEFLATED)as archive:
                archive.writestr('shuffle.ptx',combined)
                for level in('safe','mined','sched','opt'):
                    target=root/level;result=build.compile_paths(source,target,build.Options(opt_level=level))
                    assert result.rc==0,result
                    manifest=json.loads((target/'manifest.json').read_text());entry=manifest['entries'][0]
                    archive.writestr(level+'/manifest.json',json.dumps(manifest,indent=2))
                    for compiler,key in(('oa','cubin'),('ptxas','reference')):
                        binary=(target/entry[key]).read_bytes()
                        if compiler=='oa':validate(entry,binary)
                        archive.writestr(level+'/'+compiler+'.cubin',binary)
                        archive.writestr(level+'/'+compiler+'.sass.txt',subprocess.check_output(['nvdisasm','-hex','-c',str(target/entry[key])]))
                        module=driver.load(binary)
                        try:
                            for case in fixtures:
                                fn=driver.function(module,case['name'])
                                inp,out=driver.alloc(1024),driver.alloc(1024)
                                try:
                                    for threads in (32,128):
                                        for delta in(0,1,17,31,32,33,0xffffffff):
                                            rng=random.Random(15052+delta+threads)
                                            values=[rng.getrandbits(32)for _ in range(128)]
                                            before=struct.pack('<128I',*values);initial=struct.pack('<128I',*([sentinel]*128))
                                            driver.htod(inp,guard+before+guard);driver.htod(out,guard+initial+guard)
                                            driver.launch(fn,(1,1,1),(threads,1,1),[C.c_uint64(out+256),C.c_uint64(inp+256),C.c_uint32(delta)])
                                            actual=driver.dtoh(out,1024);unchanged=driver.dtoh(inp,1024)
                                            wanted=expected(case,values[:threads],delta)+[sentinel]*(128-threads)
                                            passed=(actual[:256]==actual[-256:]==guard and unchanged==guard+before+guard
                                                    and actual[256:-256]==struct.pack('<128I',*wanted))
                                            prefix=f'{level}/{compiler}/{case["name"]}/{threads}/{delta}/'
                                            for name,data in(('before-input',guard+before+guard),('output',actual),('input',unchanged)):
                                                archive.writestr(prefix+name+'.bin',data)
                                            records.append(dict(name=case['name'],level=level,compiler=compiler,threads=threads,delta=delta,passed=passed,
                                                cubin_sha256=hashlib.sha256(binary).hexdigest(),output_sha256=hashlib.sha256(actual).hexdigest(),
                                                input_sha256=hashlib.sha256(unchanged).hexdigest(),resources=driver.func_attrs(fn)))
                                finally:driver.free(inp);driver.free(out)
                        finally:driver.unload(module)
        finally:driver.close()
        report=dict(schema=1,passed=all(x['passed']for x in records),results=records,device=device,python=sys.version,
                    ptxas=subprocess.check_output(['ptxas','--version'],text=True),
                    source_hashes={name:hashlib.sha256(data).hexdigest()for name,data in payload.items()},
                    scope='Full-mask full-warp literal/parameter/varying shuffles after divergent paths reconverge; exact outputs/guards/inputs/tails, no partial participation or timing claim')
        return report,base64.b64encode(blob.getvalue()).decode()


@app.local_entrypoint()
def main():
    import base64,sys
    sys.path.insert(0,str(ROOT))
    from openassembler.transport import runtime_files
    payload=runtime_files(native=True)
    for name in('ra','contracts'):payload[f'openassembler/backend/{name}.py']=(HERE/f'candidate-{name}.py').read_bytes()
    payload['compiler_exp/gap_closure/warp_collectives/cases.py']=(HERE/'cases.py').read_bytes()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');report,encoded=execute.remote(payload)
    archive=HERE/('candidate-a100-'+stamp+'.zip');archive.write_bytes(base64.b64decode(encoded))
    report.update(archive=dict(path=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=HERE/('candidate-a100-'+stamp+'.json');path.write_text(json.dumps(report,indent=2)+'\n')
    print(path,'passed',report['passed'],'launches',len(report['results']))
    if not report['passed']:raise RuntimeError('shuffle hardware candidate failed')
