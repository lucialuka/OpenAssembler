"""A100 complete-byte gate for private local arrays, exact frames and donor optimization."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import modal

ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
app=modal.App('oa-gap-local-storage')
image=modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04',add_python='3.12')


@app.function(image=image,gpu='A100-40GB',timeout=1800,retries=0,max_containers=1)
def execute(payload):
    import base64,ctypes as C,io,os,random,struct,subprocess,sys,tempfile,zipfile
    from dataclasses import asdict
    with tempfile.TemporaryDirectory(prefix='oa-g13-')as temporary:
        root=Path(temporary)
        for name,data in payload.items():
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        (root/'env/cuda.sh').chmod(0o700);os.chdir(root)
        sys.path[:0]=[str(root),str(root/'compiler_exp/gap_closure/local_storage')]
        from cases import HEADER,cases,expected,values,LANES,THREADS,GRID,N
        from openassembler import build
        from openassembler.gpu.cuda_driver import Driver
        from openassembler.native_ownership import validate
        from openassembler.backend.storage import local_frame_inventory,local_frame
        from openassembler.ptx.parse import parse_module
        fixtures=list(cases());combined=HEADER+'\n'.join(case['source'][len(HEADER):]for case in fixtures)
        source=root/'local.ptx';source.write_text(combined)
        records=[];blob=io.BytesIO();guard=b'\xa5'*256;sentinel=0x13579bdf
        driver=Driver();driver.open();error=None
        try:
            device=asdict(driver.info());assert device['cc']=='8.0'and'A100'in device['name']
            with zipfile.ZipFile(blob,'w',zipfile.ZIP_DEFLATED)as archive:
                archive.writestr('local.ptx',combined)
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
                                total=LANES*4+512
                                inp,out=driver.alloc(total),driver.alloc(total)
                                try:
                                    for n in (0,1,N,LANES):
                                        for seed in(13052,13053):
                                            data=values(seed)
                                            before=struct.pack('<'+str(LANES)+'I',*data)
                                            initial=struct.pack('<'+str(LANES)+'I',*([sentinel]*LANES))
                                            driver.htod(inp,guard+before+guard);driver.htod(out,guard+initial+guard)
                                            driver.launch(fn,GRID,(THREADS,1,1),[C.c_uint64(out+256),C.c_uint64(inp+256),C.c_uint32(n)])
                                            actual=driver.dtoh(out,total);unchanged=driver.dtoh(inp,total)
                                            wanted=expected(data,n)
                                            passed=(actual[:256]==actual[-256:]==guard and unchanged==guard+before+guard
                                                    and actual[256:-256]==struct.pack('<'+str(LANES)+'I',*wanted))
                                            prefix=f'{level}/{compiler}/{case["name"]}/{n}/{seed}/'
                                            for name,raw in(('before-input',guard+before+guard),('output',actual),('input',unchanged)):
                                                archive.writestr(prefix+name+'.bin',raw)
                                            frames=local_frame_inventory(binary)
                                            if compiler=='oa':assert frames[case['name']]==local_frame(parse_module(case['source']).kernels[case['name']]).bytes
                                            # Donor optimization may retain local storage
                                            # for a different layout. Preserve its actual
                                            # reservation, independently of OA's contract.
                                            records.append(dict(name=case['name'],level=level,compiler=compiler,n=n,seed=seed,passed=passed,
                                                local_frame_bytes=frames[case['name']],cubin_sha256=hashlib.sha256(binary).hexdigest(),output_sha256=hashlib.sha256(actual).hexdigest(),
                                                input_sha256=hashlib.sha256(unchanged).hexdigest(),resources=driver.func_attrs(fn)))
                                finally:driver.free(inp);driver.free(out)
                        finally:driver.unload(module)
        except Exception as exc:error={'type':type(exc).__name__,'message':str(exc)}
        finally:
            try:driver.close()
            except Exception as exc:
                if error is None:error={'type':type(exc).__name__,'message':str(exc)}
        report=dict(schema=1,passed=error is None and bool(records) and all(x['passed']for x in records),error=error,results=records,device=device,python=sys.version,
                    ptxas=subprocess.check_output(['ptxas','--version'],text=True),
                    source_hashes={name:hashlib.sha256(data).hexdigest()for name,data in payload.items()},
                    scope='Per-thread private 64-byte local frames across CTAs, symbols/pointers, predicated accesses and loops; optimized donor frame0 versus OA frame64, no timing or general memory-safety claim')
        return report,base64.b64encode(blob.getvalue()).decode()


@app.local_entrypoint()
def main():
    import base64,sys
    sys.path.insert(0,str(ROOT))
    from openassembler.transport import runtime_files
    payload=runtime_files(native=True)
    import json
    identities=json.loads((HERE/'candidate-identities.json').read_text())
    for name in identities:payload[name]=(HERE/('candidate-'+Path(name).name)).read_bytes()
    payload['compiler_exp/gap_closure/local_storage/cases.py']=(HERE/'cases.py').read_bytes()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');report,encoded=execute.remote(payload)
    archive=HERE/('candidate-a100-'+stamp+'.zip');archive.write_bytes(base64.b64decode(encoded))
    report.update(archive=dict(path=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=HERE/('candidate-a100-'+stamp+'.json');path.write_text(json.dumps(report,indent=2)+'\n')
    print(path,'passed',report['passed'],'launches',len(report['results']))
    if not report['passed']:raise RuntimeError('local hardware candidate failed')
