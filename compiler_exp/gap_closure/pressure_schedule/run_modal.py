"""Matched A100 correctness, resources and paired resident kernel intervals."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import modal

ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
app=modal.App('oa-pressure-aware-virtual-scheduling')
image=modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04',add_python='3.12')


@app.function(image=image,gpu='A100-40GB',timeout=2400,retries=0,max_containers=1)
def execute(payload):
    import base64,ctypes as C,io,os,random,statistics,struct,subprocess,sys,tempfile,types,zipfile
    from dataclasses import asdict
    with tempfile.TemporaryDirectory(prefix='oa-pressure-')as temporary:
        root=Path(temporary)
        for name,data in payload.items():
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        os.chdir(root);(root/'env/cuda.sh').chmod(0o700)
        experiment=root/'compiler_exp/gap_closure/pressure_schedule'
        sys.path[:0]=[str(root),str(experiment)]
        from hardware_probes import HEADER,cases,expected
        from openassembler import build
        from openassembler.gpu.cuda_driver import Driver
        from openassembler.backend.storage import local_frame_inventory
        from openassembler.native_ownership import validate
        fixtures=list(cases());source=HEADER+'\n'.join(case['source'][len(HEADER):]for case in fixtures)
        (root/'pressure.ptx').write_text(source)
        allocators={'baseline':(experiment/'baseline-spill.py').read_bytes(),
                    'candidate':(root/'openassembler/backend/spill.py').read_bytes()}
        def install(mode):
            path=root/'openassembler/backend/spill.py';path.write_bytes(allocators[mode])
            name='openassembler.backend.spill';module=types.ModuleType(name);module.__file__=str(path)
            sys.modules[name]=module;exec(compile(allocators[mode],str(path),'exec'),module.__dict__)
        driver=Driver();driver.open();device=asdict(driver.info())
        assert device['cc']=='8.0'and'A100'in device['name']
        blob=io.BytesIO();records=[];timings=[];warmups=[];binaries={};resources={};contracts={};error=None
        lanes=384;guard=b'\xa5'*256;stored=set()
        try:
            with zipfile.ZipFile(blob,'w',zipfile.ZIP_DEFLATED)as archive:
                archive.writestr('pressure.ptx',source)
                def keep(raw):
                    digest=hashlib.sha256(raw).hexdigest()
                    if digest not in stored:archive.writestr('blobs/'+digest+'.bin',raw);stored.add(digest)
                    return digest
                def launch_data(case,n,seed):
                    rng=random.Random(seed);values=[rng.getrandbits(32)for _ in range(lanes*case['count'])]
                    original=guard+struct.pack('<'+str(len(values))+'I',*values)+guard
                    reference=expected(values,case,n,lanes)
                    wanted=guard+struct.pack('<'+str(len(reference))+'I',*reference)+guard
                    initial=guard+struct.pack('<'+str(lanes*case['output_words'])+'I',*([0x13579bdf]*(lanes*case['output_words'])))+guard
                    return original,initial,wanted
                for level in('safe','mined','sched','opt'):
                    for compiler in('baseline','candidate'):
                        install(compiler);target=root/(level+'-'+compiler)
                        result=build.compile_paths(root/'pressure.ptx',target,build.Options(opt_level=level))
                        assert result.rc==0,(compiler,level,result)
                        manifest=json.loads((target/'manifest.json').read_text());entry=manifest['entries'][0]
                        binary=(target/entry['cubin']).read_bytes();validate(entry,binary)
                        binaries[level,compiler]=binary
                        contracts[level,compiler]={k['name']:k for k in entry['kernels']}
                        archive.writestr(f'{level}/{compiler}/manifest.json',json.dumps(manifest,indent=2))
                        archive.writestr(f'{level}/{compiler}/native.cubin',binary)
                        archive.writestr(f'{level}/{compiler}/native.sass.txt',subprocess.check_output(['nvdisasm','-hex','-c',str(target/entry['cubin'])]))
                        if compiler=='candidate':
                            binaries[level,'ptxas']=(target/entry['reference']).read_bytes()
                            archive.writestr(f'{level}/ptxas/native.cubin',binaries[level,'ptxas'])
                    for compiler in('baseline','candidate','ptxas'):
                        binary=binaries[level,compiler];module=driver.load(binary);frames=local_frame_inventory(binary)
                        try:
                            for case in fixtures:
                                fn=driver.function(module,case['name']);attrs=driver.func_attrs(fn)
                                assert attrs['local_bytes']==frames[case['name']]
                                if compiler!='ptxas':
                                    contract=contracts[level,compiler][case['name']]
                                    assert frames[case['name']]==contract.get('local_bytes',0)
                                    assert attrs['num_regs']==contract['registers']<=case['limit']
                                inp=driver.alloc(lanes*case['count']*4+512);out=driver.alloc(lanes*case['output_words']*4+512)
                                try:
                                    for n in(0,1,259,lanes):
                                        for seed in(13052,13053):
                                            original,initial,wanted=launch_data(case,n,seed)
                                            driver.htod(inp,original);driver.htod(out,initial)
                                            driver.launch(fn,(3,1,1),(128,1,1),[C.c_uint64(out+256),C.c_uint64(inp+256),C.c_uint32(n)])
                                            actual=driver.dtoh(out,len(wanted));unchanged=driver.dtoh(inp,len(original))
                                            records.append(dict(name=case['name'],domain=case['domain'],level=level,compiler=compiler,n=n,seed=seed,
                                                passed=actual==wanted and unchanged==original,resources=attrs,frame=frames[case['name']],
                                                cubin_sha256=hashlib.sha256(binary).hexdigest(),
                                                artifacts={'before':keep(original),'input':keep(unchanged),'output':keep(actual)}))
                                finally:driver.free(inp);driver.free(out)
                        finally:driver.unload(module)
                assert all(row['passed']for row in records),'complete correctness gate failed; timing withheld'
                modules=[];buffers=[];functions={};inputs={};samples={}
                try:
                    for compiler in('baseline','candidate','ptxas'):
                        module=driver.load(binaries['opt',compiler]);modules.append(module)
                        for case in fixtures:
                            name=case['name']+'/'+compiler;fn=driver.function(module,case['name']);functions[name]=fn
                            resources[name]=driver.func_attrs(fn);samples[name]=[]
                    for case in fixtures:
                        original,initial,wanted=launch_data(case,259,17052)
                        inp=driver.alloc(len(original));out=driver.alloc(len(initial));buffers.extend([inp,out])
                        driver.htod(inp,original);driver.htod(out,initial)
                        inputs[case['name']]=(inp,out,original,wanted)
                    def check(name):
                        case_name=name.split('/')[0];inp,out,original,wanted=inputs[case_name]
                        actual=driver.dtoh(out,len(wanted));unchanged=driver.dtoh(inp,len(original))
                        assert actual==wanted and unchanged==original,(name,'timed full-state failure')
                        return {'before':keep(original),'input':keep(unchanged),'output':keep(actual)}
                    for name,fn in functions.items():
                        inp,out,_,_=inputs[name.split('/')[0]]
                        for repetition in range(3):
                            driver.launch(fn,(3,1,1),(128,1,1),[C.c_uint64(out+256),C.c_uint64(inp+256),C.c_uint32(259)])
                            warmups.append(dict(name=name,repetition=repetition,passed=True,artifacts=check(name)))
                    order=random.Random(17053)
                    for repetition in range(21):
                        names=list(functions);order.shuffle(names)
                        for name in names:
                            inp,out,_,_=inputs[name.split('/')[0]]
                            duration=driver.launch(functions[name],(3,1,1),(128,1,1),[C.c_uint64(out+256),C.c_uint64(inp+256),C.c_uint32(259)],time_it=True)
                            samples[name].append(duration)
                            timings.append(dict(repetition=repetition,name=name,cuda_event_ms=duration,passed=True,artifacts=check(name)))
                finally:
                    for address in buffers:driver.free(address)
                    for module in modules:driver.unload(module)
        except Exception as exc:error={'type':type(exc).__name__,'message':str(exc)}
        finally:driver.close();install('candidate')
        report=dict(schema=1,passed=error is None and bool(records)and all(row['passed']for row in records),error=error,
                    device=device,python=sys.version,ptxas=subprocess.check_output(['ptxas','--version'],text=True),
                    results=records,timings=timings,warmups=warmups,timing_resources=resources,samples_ms=locals().get('samples',{}),
                    medians_ms={name:statistics.median(values)for name,values in locals().get('samples',{}).items()if values},
                    allocator_sha256={mode:hashlib.sha256(data).hexdigest()for mode,data in allocators.items()},
                    source_hashes={name:hashlib.sha256(data).hexdigest()for name,data in payload.items()},
                    scope='Seventeen exact pressure workloads, four OA levels, frozen baseline/candidate/same-PTX controls; complete retained bytes and 21 paired resident CUDA-event samples per opt variant. No host/build/whole-application or universal occupancy optimum claim')
        return report,base64.b64encode(blob.getvalue()).decode()


@app.local_entrypoint()
def main():
    import base64,sys
    sys.path.insert(0,str(ROOT))
    from openassembler.transport import runtime_files
    payload=runtime_files(native=True)
    for leaf in('pressure','spill'):payload[f'openassembler/backend/{leaf}.py']=(HERE/(leaf+'.py')).read_bytes()
    for name in('baseline-spill.py','probes.py','hardware_probes.py','legacy_scalar_probes.py','legacy_group_probes.py','legacy_tight_probes.py'):
        payload['compiler_exp/gap_closure/pressure_schedule/'+name]=(HERE/name).read_bytes()
    harness_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');report,encoded=execute.remote(payload)
    archive=HERE/('candidate-a100-'+stamp+'.zip');archive.write_bytes(base64.b64decode(encoded))
    report.update(archive=dict(path=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),harness_sha256=harness_hash)
    path=HERE/('candidate-a100-'+stamp+'.json');path.write_text(json.dumps(report,indent=2)+'\n')
    print(path,'passed',report['passed'],'correctness',len(report['results']),'timed',len(report['timings']))
    if not report['passed']:raise RuntimeError('A100 pressure scheduling candidate failed')
