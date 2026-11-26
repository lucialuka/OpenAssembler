"""Paired resident kernel timing/resource controls for one pressure workload."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import modal

ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
app=modal.App('oa-spill-resources')
image=modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04',add_python='3.12')


@app.function(image=image,gpu='A100-40GB',timeout=1800,retries=0)
def execute(files):
    import base64,ctypes as C,io,os,random,statistics,struct,subprocess,sys,tempfile,zipfile
    from dataclasses import asdict
    with tempfile.TemporaryDirectory(prefix='oa-spill-resources-')as temporary:
        root=Path(temporary)
        for name,data in files.items():
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        os.chdir(root);(root/'env/cuda.sh').chmod(0o700)
        sys.path[:0]=[str(root),str(root/'compiler_exp/gap_closure/spilling')]
        from probes import HEADER,cases,expected
        from openassembler import build
        from openassembler.gpu.cuda_driver import Driver
        from openassembler.native_ownership import validate
        fixtures=[case for case in cases()if case['name']in('pressure_260_64','pressure_auto')]
        source=HEADER+'\n'.join(case['source'][len(HEADER):]for case in fixtures)
        (root/'pressure.ptx').write_text(source)
        result=build.compile_paths(root/'pressure.ptx',root/'build',build.Options(opt_level='opt'))
        assert result.rc==0
        manifest=json.loads((root/'build/manifest.json').read_text());entry=manifest['entries'][0]
        blob=io.BytesIO();samples={};records=[];modules=[];resources={};guard=b'\xa5'*256
        rng=random.Random(17052);data=[rng.getrandbits(32)for _ in range(384*260)]
        before=struct.pack('<'+str(len(data))+'I',*data);wanted=struct.pack('<384I',*expected(data,260,259,384))
        driver=Driver();driver.open();inp=out=None
        try:
            device=asdict(driver.info());assert device['cc']=='8.0'and'A100'in device['name']
            inp=driver.alloc(len(before)+512);out=driver.alloc(len(wanted)+512)
            driver.htod(inp,guard+before+guard);driver.htod(out,guard+struct.pack('<384I',*([0x13579bdf]*384))+guard)
            args=[C.c_uint64(out+256),C.c_uint64(inp+256),C.c_uint32(259)]
            with zipfile.ZipFile(blob,'w',zipfile.ZIP_DEFLATED)as archive:
                archive.writestr('pressure.ptx',source);archive.writestr('manifest.json',json.dumps(manifest,indent=2))
                archive.writestr('input.bin',guard+before+guard)
                functions={}
                for compiler,key in(('oa','cubin'),('ptxas','reference')):
                    binary=(root/'build'/entry[key]).read_bytes()
                    if compiler=='oa':validate(entry,binary)
                    archive.writestr(compiler+'.cubin',binary)
                    module=driver.load(binary);modules.append(module)
                    for case in fixtures:
                        name=compiler+'/'+case['name'];functions[name]=driver.function(module,case['name'])
                        resources[name]=driver.func_attrs(functions[name]);samples[name]=[]
                        for _ in range(3):driver.launch(functions[name],(3,1,1),(128,1,1),args)
                        assert driver.dtoh(out,len(wanted)+512)==guard+wanted+guard
                order=random.Random(17053)
                for repetition in range(13):
                    names=list(functions);order.shuffle(names)
                    for name in names:
                        duration=driver.launch(functions[name],(3,1,1),(128,1,1),args,time_it=True)
                        actual=driver.dtoh(out,len(wanted)+512);unchanged=driver.dtoh(inp,len(before)+512)
                        assert actual==guard+wanted+guard and unchanged==guard+before+guard
                        samples[name].append(duration)
                        archive.writestr(f'outputs/{repetition}/{name.replace("/","-")}.bin',actual)
                        records.append(dict(repetition=repetition,name=name,cuda_event_ms=duration,passed=True,output_sha256=hashlib.sha256(actual).hexdigest()))
        finally:
            if inp is not None:driver.free(inp)
            if out is not None:driver.free(out)
            for module in modules:driver.unload(module)
            driver.close()
        report=dict(schema=1,passed=True,device=device,python=sys.version,
                    ptxas=subprocess.check_output(['ptxas','--version'],text=True),samples_ms=samples,records=records,resources=resources,
                    medians_ms={name:statistics.median(values)for name,values in samples.items()},
                    source_hashes={name:hashlib.sha256(data).hexdigest()for name,data in files.items()},
                    scope='One exact synthetic scalar-pressure workload, resident CUDA-event kernel intervals in randomized orders, 13 samples per native/reference/limit variant; no host/build/amortization/whole-application or universal optimum claim')
        return report,base64.b64encode(blob.getvalue()).decode()


@app.local_entrypoint()
def main():
    import base64,sys
    sys.path.insert(0,str(ROOT))
    from openassembler.transport import runtime_files
    files=runtime_files(native=True);files['compiler_exp/gap_closure/spilling/probes.py']=(HERE/'probes.py').read_bytes()
    harness_digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report,encoded=execute.remote(files)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');archive=HERE/('resources-'+stamp+'.zip')
    archive.write_bytes(base64.b64decode(encoded));report.update(archive=dict(path=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),harness_sha256=harness_digest)
    path=HERE/('resources-'+stamp+'.json');path.write_text(json.dumps(report,indent=2)+'\n');print(path,report['medians_ms'])
