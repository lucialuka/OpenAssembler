"""A100 Python atomic primitive gate with independent contention invariants."""
from collections import Counter
import ctypes as C
from dataclasses import asdict
import hashlib,json,platform,struct,subprocess
from pathlib import Path
from openassembler import build
from openassembler.gpu.cuda_driver import Driver
from .probe import cases
HERE=Path(__file__).resolve().parent
MASK=0xffffffff


def chain(initial,final,returned,offered):
    edges={}
    for old,new in zip(returned,offered):
        if old in edges:return False
        edges[old]=new
    current=initial;seen=0
    while current in edges:
        current=edges.pop(current);seen+=1
    return not edges and current==final and seen==len(offered)


def run(root):
    root=Path(root);root.mkdir(exist_ok=True);variants={};manifests={}
    for level in('safe','mined','sched','opt'):
        output=root/level;result=build.compile_paths(HERE/'fixture.py',output,build.Options(opt_level=level,no_sass=False))
        assert result.rc==0,result
        manifest=json.loads((output/'manifest.json').read_text());entry=manifest['entries'][0];assert entry['native_ownership']
        manifests[level]=manifest;variants['oa_'+level]=(output/entry['cubin']).read_bytes()
        if level=='safe':variants['same_ptx_ptxas']=(output/entry['reference']).read_bytes()
    records=[];driver=Driver();driver.open()
    try:
        device=asdict(driver.info());assert device['cc']=='8.0'and'A100'in device['name'],device
        for variant,binary in variants.items():
            module=driver.load(binary)
            try:
                batch=[]
                for case in cases():
                    if case['name'].startswith('checked')or case['name']=='evaluation_order':batch.append((case['name'],case['arrays'],case['scalar'],case['threads'],1,case['expected'],'exact',0))
                for repetition,initial in enumerate((0x7ffffff8,0xfffffffa,13)):
                    for name in('add','exchange','compare_exchange'):
                        state=initial if name=='add'else 0xdeadbeef if name=='exchange'else 0
                        arrays={'state':[state],'returned':[0xabcdef01]*384}
                        if name=='add':arrays={'values':[1]*384,**arrays}
                        batch.append((name,arrays,{'n':259},128,3,None,'contention',repetition))
                for name,arrays,scalars,threads,grid,expected,mode,repetition in batch:
                    fn=driver.function(module,name);allocations=[];addresses={};payloads={};guard=b'\xa5'*256
                    try:
                        for key,values in arrays.items():
                            payload=struct.pack('<'+str(len(values))+'I',*(x&MASK for x in values));allocation=driver.alloc(512+len(payload));allocations.append(allocation);addresses[key]=allocation+256;payloads[key]=guard+payload+guard;driver.htod(allocation,payloads[key])
                        params=next(k['params']for k in manifests['safe']['entries'][0]['kernels']if k['name']==name)
                        args=[C.c_uint64(addresses[key])if kind=='u64'else C.c_int32(scalars[key])for kind,key in params]
                        driver.launch(fn,(grid,1,1),(threads,1,1),args);actual={};guards={}
                        for key,values in arrays.items():
                            raw=driver.dtoh(addresses[key]-256,len(payloads[key]));guards[key]=raw[:256]==raw[-256:]==guard;actual[key]=list(struct.unpack('<'+str(len(values))+'I',raw[256:-256]))
                        if mode=='exact':
                            passed=actual=={key:[x&MASK for x in values]for key,values in expected.items()}
                        else:
                            n=scalars['n'];old=arrays['state'][0];final=actual['state'][0];returned=actual['returned'][:n]
                            passed=actual['returned'][n:]==arrays['returned'][n:]
                            if name=='add':passed&=final==(old+n)&MASK and Counter(returned)==Counter((old+i)&MASK for i in range(n))and actual['values']==arrays['values']
                            elif name=='exchange':passed&=chain(old,final,returned,list(range(1,n+1)))
                            else:passed&=1<=final<=n and Counter(returned)==Counter([0]+[final]*(n-1))
                        passed&=all(guards.values());records.append({'variant':variant,'case':name,'mode':mode,'repetition':repetition,'inputs':arrays,'actual':actual,'expected':expected,'scalar':scalars,'guards':guards,'passed':bool(passed),'resources':driver.func_attrs(fn),'cubin_sha256':hashlib.sha256(binary).hexdigest()});(root/'partial-results.json').write_text(json.dumps(records,indent=2)+'\n');assert passed,records[-1]
                    finally:
                        for allocation in allocations:driver.free(allocation)
            finally:driver.unload(module)
    finally:driver.close()
    report={'schema':1,'gap':'G08/G14','passed':all(r['passed']for r in records),'device':device,'python':platform.python_version(),'ptxas':subprocess.check_output(['ptxas','--version'],text=True),'launches':len(records),'results':records,'manifests':manifests,'performance':'not_measured'}
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report
