"""A100 checked Buffer bounds/strides/storage/alias gate via normal compiler and runner."""
import ctypes as C
from dataclasses import asdict
import hashlib,json,platform,struct,subprocess
from pathlib import Path
from openassembler import build
from openassembler.gpu.cuda_driver import Driver,run_plan_job
from openassembler.runner import Plan,spec_case,RunError
from .probe import cases,FORMAT
HERE=Path(__file__).resolve().parent


def run(root):
    root=Path(root);root.mkdir(exist_ok=True);variants={};manifests={};outputs={}
    for level in('safe','mined','sched','opt'):
        output=root/level;result=build.compile_paths(HERE/'fixture.py',output,build.Options(opt_level=level,no_sass=False))
        assert result.rc==0,result
        manifest=json.loads((output/'manifest.json').read_text());entry=manifest['entries'][0]
        assert entry['native_ownership'] and all(k.get('buffer_contract')for k in entry['kernels'])
        assert all(k['verify']=='not-run'for k in entry['kernels'])
        variants['oa_'+level]=(output/entry['cubin']).read_bytes();manifests[level]=manifest;outputs[level]=output
        if level=='safe':variants['same_ptx_ptxas']=(output/entry['reference']).read_bytes()
    checked=build.build_source(HERE/'fixture.py','python','verify',build.Options(no_sass=True,verify=True),single=True)
    assert checked.rc==2 and all(k.verify=='not-verified'for k in checked.kept)
    verification={'returncode':checked.rc,'results':[{'name':k.name,'status':k.verify,'reason':k.verify_msg}for k in checked.kept]}
    records=[];spec_records=[];driver=Driver();driver.open()
    try:
        device=asdict(driver.info());assert device['cc']=='8.0'and'A100'in device['name'],device
        for variant,binary in variants.items():
            module=driver.load(binary)
            try:
                for case in cases():
                    fn=driver.function(module,case['name']);allocations=[];addresses={};payloads={};guard=b'\xa5'*256
                    try:
                        for name,(kind,values)in case['arrays'].items():
                            payload=struct.pack('<'+str(len(values))+FORMAT[kind],*values);allocation=driver.alloc(512+len(payload));allocations.append(allocation);addresses[name]=allocation+256;payloads[name]=guard+payload+guard;driver.htod(allocation,payloads[name])
                        params=next(k['params']for k in manifests['safe']['entries'][0]['kernels']if k['name']==case['name'])
                        driver.launch(fn,(1,1,1),(case['threads'],1,1),[C.c_uint64(addresses[name])for _,name in params])
                        actual={};guards={}
                        for name,(kind,values)in case['arrays'].items():
                            raw=driver.dtoh(addresses[name]-256,len(payloads[name]));guards[name]=raw[:256]==raw[-256:]==guard;actual[name]=list(struct.unpack('<'+str(len(values))+FORMAT[kind],raw[256:-256]))
                        passed=actual==case['expected']and all(guards.values());records.append({'variant':variant,'case':case['name'],'inputs':case['arrays'],'actual':actual,'expected':case['expected'],'guards':guards,'passed':passed,'elements':sum(len(v)for v in actual.values()),'resources':driver.func_attrs(fn),'cubin_sha256':hashlib.sha256(binary).hexdigest()})
                        (root/'partial-results.json').write_text(json.dumps(records,indent=2)+'\n');assert passed,records[-1]
                    finally:
                        for allocation in allocations:driver.free(allocation)
            finally:driver.unload(module)
        for case in cases():
            output=outputs['opt'];manifest=json.loads((output/'manifest.json').read_text())
            spec={'buffers':{name:{'dtype':kind,'count':len(values),'init':values}for name,(kind,values)in case['arrays'].items()},'steps':[{'kernel':case['name'],'block':case['threads'],'args':[{'buf':name}for _,name in next(k['params']for k in manifest['entries'][0]['kernels']if k['name']==case['name'])]}],'reps':0}
            plan=Plan(reference_comparison=False);spec_case(spec,manifest,output,0,plan,'checked-'+case['name']);got=run_plan_job(driver,plan.jobs[0]);actual={name:list(struct.unpack('<'+str(len(values))+FORMAT[kind],b''.join(got['out'][name])))for name,(kind,values)in case['arrays'].items()};assert actual==case['expected']
            spec_records.append({'case':case['name'],'spec':spec,'actual':actual,'expected':case['expected'],'passed':True})
        rejected=[]
        for error in('short','alignment','alias','dtype'):
            case=next(c for c in cases()if c['name']=='checked_wide');output=outputs['opt'];manifest=json.loads((output/'manifest.json').read_text())
            spec={'buffers':{name:{'dtype':kind,'count':len(values),'init':values}for name,(kind,values)in case['arrays'].items()},'steps':[{'kernel':case['name'],'block':16,'args':[{'buf':name}for name in case['arrays']]}]}
            if error=='short':spec['buffers']['a']={'dtype':'i64','count':7,'init':[0]*7}
            elif error=='alignment':spec['buffers']['a']={'dtype':'i64','count':10,'init':[0]*10};spec['steps'][0]['args'][0]['offset']=8
            elif error=='alias':spec['steps'][0]['args'][1]={'buf':'a'}
            else:spec['buffers']['a']={'dtype':'u64','count':8,'init':[0]*8}
            try:spec_case(spec,manifest,output,0,Plan(reference_comparison=False),'invalid')
            except RunError as exc:rejected.append({'case':error,'reason':str(exc),'passed':True})
            else:raise AssertionError('malformed Buffer binding accepted: '+error)
    finally:driver.close()
    report={'schema':1,'gap':'G07/G09','passed':True,'device':device,'python':platform.python_version(),'ptxas':subprocess.check_output(['ptxas','--version'],text=True),'custom_launches':len(records),'compared_elements':sum(r['elements']for r in records),'results':records,'explicit_spec_results':spec_records,'rejected_bindings':rejected,'manifests':manifests,'generic_verification':verification,'performance':'not_measured'}
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report
