"""Ordinary Python source compilation + typed explicit specs, independent full-width gate."""
from __future__ import annotations
import ctypes as C
from dataclasses import asdict
import hashlib,json,platform,struct,subprocess
from pathlib import Path
from openassembler import build
from openassembler.gpu.cuda_driver import Driver,run_plan_job
from openassembler.pyfront import compile_file
from openassembler.runner import Plan,spec_case
from .probe import cases,FORMAT
HERE=Path(__file__).resolve().parent


def run(root):
    root=Path(root);root.mkdir(exist_ok=True)
    source=HERE/'fixture.py';ptx=compile_file(source)[0]
    (root/'source.py').write_bytes(source.read_bytes());(root/'source.ptx').write_text(ptx)
    variants={};manifests={};outputs={}
    for level in ('safe','mined','sched','opt'):
        output=root/level;result=build.compile_paths(source,output,build.Options(opt_level=level,no_sass=False))
        assert result.rc==0,result
        manifest=json.loads((output/'manifest.json').read_text());entry=manifest['entries'][0]
        assert entry['native_ownership'] and len(entry['kernels'])==5
        variants['oa_'+level]=(output/entry['cubin']).read_bytes();manifests[level]=manifest;outputs[level]=output
        if level=='safe':variants['same_ptx_ptxas']=(output/entry['reference']).read_bytes()
    verified=build.build_source(HERE/'verified_fixture.py','python','verified',build.Options(verify=True,no_sass=True),single=True)
    assert verified.rc==0 and all(k.verify=='ok' for k in verified.kept),[(k.name,k.verify,k.verify_msg) for k in verified.kernels]
    verified_record=verified.entry(True)
    records=[];spec_records=[];driver=Driver();driver.open()
    try:
        device=asdict(driver.info());assert device['cc']=='8.0' and 'A100' in device['name'],device
        for variant,binary in variants.items():
            module=driver.load(binary)
            try:
                for iteration,(n,delta) in enumerate(((14,0),(0,1),(16,2))):
                    for case in cases(n=n,delta=delta):
                        fn=driver.function(module,case['name']);allocations=[];addresses={};payloads={};guard=b'\xa5'*256
                        try:
                            for name,(kind,values) in case['arrays'].items():
                                payload=struct.pack('<'+str(len(values))+FORMAT[kind],*values)
                                allocation=driver.alloc(512+len(payload));allocations.append(allocation);addresses[name]=allocation+256
                                payloads[name]=guard+payload+guard;driver.htod(allocation,payloads[name])
                            params=next(k['params'] for k in manifests['safe']['entries'][0]['kernels'] if k['name']==case['name'])
                            args=[]
                            for kind,name in params:
                                value=addresses.get(name,case['scalars'].get(name))
                                args.append({'s64':C.c_int64,'u64':C.c_uint64,'s32':C.c_int32,'u32':C.c_uint32}[kind](value))
                            driver.launch(fn,(1,1,1),(32,1,1),args)
                            actual={};guards={}
                            for name,(kind,values) in case['arrays'].items():
                                raw=driver.dtoh(addresses[name]-256,len(payloads[name]));guards[name]=raw[:256]==raw[-256:]==guard
                                actual[name]=list(struct.unpack('<'+str(len(values))+FORMAT[kind],raw[256:-256]))
                            passed=actual==case['expected'] and all(guards.values())
                            records.append({'variant':variant,'case':case['name'],'iteration':iteration,'active_lanes':n,
                                            'arrays':case['arrays'],'scalars':case['scalars'],'actual':actual,'expected':case['expected'],
                                            'guards':guards,'passed':passed,'elements':sum(len(v) for v in actual.values()),
                                            'cubin_sha256':hashlib.sha256(binary).hexdigest(),'resources':driver.func_attrs(fn)})
                            (root/'partial-results.json').write_text(json.dumps(records,indent=2)+'\n')
                            assert passed,records[-1]
                        finally:
                            for address in allocations:driver.free(address)
            finally:driver.unload(module)
        # Exercise the saved native bundle's ordinary explicit spec path, including
        # signed64 scalar packing and typed64 buffer initialization/readback.
        output=outputs['opt'];manifest=json.loads((output/'manifest.json').read_text())
        for case in cases():
            dtype=lambda kind:{'s32':'i32','s64':'i64'}.get(kind,kind)
            spec={'buffers':{name:{'dtype':dtype(kind),'count':len(values),'init':values} for name,(kind,values) in case['arrays'].items()},
                  'steps':[{'kernel':case['name'],'block':32,'grid':1,'args':[{'buf':name} if name in case['arrays'] else {dtype(kind):case['scalars'][name]}
                            for kind,name in next(k['params'] for k in manifest['entries'][0]['kernels'] if k['name']==case['name'])]}],'reps':0}
            plan=Plan(reference_comparison=False);spec_case(spec,manifest,output,0,plan,'wide-'+case['name'])
            result=run_plan_job(driver,plan.jobs[0]);actual={}
            for name,(kind,values) in case['arrays'].items():
                actual[name]=list(struct.unpack('<'+str(len(values))+FORMAT[kind],b''.join(result['out'][name])))
            assert actual==case['expected'],(case['name'],actual)
            spec_records.append({'case':case['name'],'spec':spec,'actual':actual,'expected':case['expected'],'passed':True})
    finally:driver.close()
    report={'schema':1,'gap':'G03','passed':all(r['passed'] for r in records+spec_records),
            'device':device,'python':platform.python_version(),'ptxas':subprocess.check_output(['ptxas','--version'],text=True),
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest(),
            'custom_launches':len(records),'compared_elements':sum(r['elements'] for r in records),'explicit_spec_cases':len(spec_records),
            'results':records,'explicit_spec_results':spec_records,'manifests':manifests,'typed_auto_verification':verified_record,'performance':'not_measured'}
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report
