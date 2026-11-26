"""A100 direct scalar calls from original PTX through ordinary native builds."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import modal
ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
app=modal.App('oa-direct-device-functions')
image=modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04',add_python='3.12')
@app.function(image=image,gpu='A100-40GB',timeout=1800,retries=0,max_containers=1)
def execute(payload):
    import base64,ctypes as C,io,os,struct,subprocess,sys,tempfile,zipfile
    from dataclasses import asdict
    with tempfile.TemporaryDirectory(prefix='oa-direct-functions-')as temp:
        root=Path(temp)
        for name,data in payload.items():
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        (root/'env/cuda.sh').chmod(0o700);os.chdir(root);sys.path[:0]=[str(root),str(root/'compiler_exp/gap_closure/device_functions')]
        from fixtures import expanded_cases as cases,values,expected,SENTINEL
        from openassembler import build
        from openassembler.gpu.cuda_driver import Driver
        from openassembler.native_ownership import validate
        from openassembler.backend.storage import local_frame_inventory
        from openassembler.ptx.parse import parse_module
        fixtures=list(cases());driver=Driver();driver.open();device=asdict(driver.info());assert device['cc']=='8.0'and'A100'in device['name']
        records=[];error=None;blob=io.BytesIO();guard=b'\xa5'*256;lanes=384
        try:
            with zipfile.ZipFile(blob,'w',zipfile.ZIP_DEFLATED)as archive:
                for case in fixtures:
                    source=root/(case['name']+('.cu'if 'cuda_source'in case else'.ptx'));source.write_text(case.get('cuda_source',case['source']))
                    archive.writestr(case['name']+'/input'+source.suffix,source.read_bytes())
                    for level in('safe','mined','sched','opt'):
                        target=root/(case['name']+'-'+level);result=build.compile_paths(source,target,build.Options(opt_level=level));assert result.rc==0,result
                        manifest=json.loads((target/'manifest.json').read_text());entry=manifest['entries'][0];contract=entry['kernels'][0]
                        emitted=(target/entry['ptx']).read_text();parsed=parse_module(emitted)
                        if level=='safe':archive.writestr(case['name']+'/source.ptx',emitted)
                        archive.writestr(case['name']+'/'+level+'/emitted.ptx',emitted)
                        assert contract['inlined_device_calls']['strategy']=='direct-scalar-static-inlining'
                        assert{name:r['call_sites']for name,r in contract['inlined_device_calls']['functions'].items()}==parsed.inlined_calls[case['entry']]
                        prefix=case['name']+'/'+level+'/'
                        archive.writestr(prefix+'manifest.json',json.dumps(manifest,indent=2))
                        for compiler,key in(('oa','cubin'),('ptxas','reference')):
                            binary=(target/entry[key]).read_bytes();archive.writestr(prefix+compiler+'.cubin',binary)
                            archive.writestr(prefix+compiler+'.sass.txt',subprocess.check_output(['nvdisasm','-hex','-c',str(target/entry[key])]))
                            if compiler=='oa':validate(entry,binary)
                            module=driver.load(binary)
                            try:
                                fn=driver.function(module,case['entry']);resources=driver.func_attrs(fn);frames=local_frame_inventory(binary)
                                if compiler=='oa':assert frames[case['entry']]==contract.get('local_bytes',0)==resources['local_bytes']and resources['num_regs']==contract['registers']
                                inp,out=driver.alloc(lanes*8+512),driver.alloc(lanes*8+512)
                                try:
                                    for n in(0,1,259,384):
                                        for seed in(17341,17342):
                                            data=values(seed,lanes,case['mode']);before=guard+struct.pack('<384Q',*data)+guard;initial=guard+struct.pack('<Q',SENTINEL)*lanes+guard
                                            driver.htod(inp,before);driver.htod(out,initial)
                                            driver.launch(fn,(3,1,1),(128,1,1),[C.c_uint64(out+256),C.c_uint64(inp+256),C.c_uint32(n)])
                                            actual=driver.dtoh(out,len(initial));unchanged=driver.dtoh(inp,len(before));wanted=guard+expected(data,case,n)+guard
                                            run=prefix+compiler+'/'+str(n)+'/'+str(seed)+'/'
                                            for name,raw in(('before-input',before),('input',unchanged),('output',actual)):archive.writestr(run+name+'.bin',raw)
                                            records.append(dict(name=case['name'],level=level,compiler=compiler,n=n,seed=seed,passed=actual==wanted and unchanged==before,resources=resources,local_frame_bytes=frames[case['entry']],cubin_sha256=hashlib.sha256(binary).hexdigest(),input_sha256=hashlib.sha256(unchanged).hexdigest(),output_sha256=hashlib.sha256(actual).hexdigest()))
                                finally:driver.free(inp);driver.free(out)
                            finally:driver.unload(module)
        except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc))
        finally:driver.close()
        return dict(schema=1,passed=error is None and len(records)==len(fixtures)*64 and all(r['passed']for r in records),error=error,results=records,device=device,python=sys.version,ptxas=subprocess.check_output(['ptxas','--version'],text=True),source_hashes={name:hashlib.sha256(data).hexdigest()for name,data in payload.items()},scope='Direct scalar static function expansion, ordinary source builds/ownership, nested/guarded/repeated calls/local resources and independent full bytes against same-original-PTX controls; no native stack/linking or performance claim'),base64.b64encode(blob.getvalue()).decode()
@app.local_entrypoint()
def main():
    import base64,sys
    sys.path.insert(0,str(ROOT));from openassembler.transport import runtime_files
    payload=runtime_files(native=True)
    for name,source in{'openassembler/ptx/parse.py':'candidate-parse.py','openassembler/ptx/inlining.py':'inlining.py','openassembler/build.py':'candidate-build.py','openassembler/native_ownership.py':'candidate-native-ownership.py','compiler_exp/gap_closure/device_functions/fixtures.py':'fixtures.py','compiler_exp/gap_closure/device_functions/functions.ptx':'functions.ptx','compiler_exp/gap_closure/device_functions/nvcc_scalar.ptx':'nvcc_scalar.ptx','compiler_exp/gap_closure/device_functions/nvcc_scalar.cu':'nvcc_scalar.cu'}.items():payload[name]=(HERE/source).read_bytes()
    harness=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');report,encoded=execute.remote(payload)
    archive=HERE/('candidate-a100-'+stamp+'.zip');archive.write_bytes(base64.b64decode(encoded));report.update(archive=dict(path=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),harness_sha256=harness)
    path=HERE/('candidate-a100-'+stamp+'.json');path.write_text(json.dumps(report,indent=2)+'\n');print(path,'passed',report['passed'],'launches',len(report['results']))
    if not report['passed']:raise RuntimeError('direct function hardware gate failed')
