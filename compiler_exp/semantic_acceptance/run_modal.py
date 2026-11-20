"""Full-output A100 confirmation of scalar modifier semantics; no timing claim."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import modal

HERE=Path(__file__).resolve().parent
ROOT=Path.cwd()
app=modal.App('openassembler-semantic-acceptance')
image=modal.Image.from_registry('pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel').apt_install('git').pip_install('pytest==8.4.2')


@app.function(image=image,timeout=300,max_containers=1,retries=0)
def regress(runtime,tests):
    import os,subprocess,sys,tempfile
    with tempfile.TemporaryDirectory(prefix='oa-semantic-tests-') as temporary:
        root=Path(temporary)
        for name,data in {**runtime,**tests}.items():
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        (root/'env/cuda.sh').chmod(0o700)
        files=[name for name in tests if name.startswith('tests/test_')]
        completed=subprocess.run([sys.executable,'-m','pytest','-q','-rs',*files],cwd=root,
                                 env=dict(os.environ,PYTHONPATH=str(root)),capture_output=True,text=True,timeout=240)
        return {'returncode':completed.returncode,'stdout':completed.stdout,'stderr':completed.stderr,
                'torch':__import__('torch').__version__,'python':sys.version,'tests':files,
                'compiler_hashes':{n:hashlib.sha256(v).hexdigest() for n,v in runtime.items()},
                'test_hashes':{n:hashlib.sha256(v).hexdigest() for n,v in tests.items()}}


@app.function(image=image,gpu='A100-40GB',timeout=600,max_containers=1,retries=0)
def execute(runtime,experiment):
    import base64
    import ctypes as C
    from dataclasses import asdict
    import io
    import os
    import struct
    import sys
    import tempfile
    import zipfile
    with tempfile.TemporaryDirectory(prefix='oa-semantic-gate-') as temporary:
        root=Path(temporary)
        for name,data in runtime.items():
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        for name,data in experiment.items():(root/name).write_bytes(data)
        (root/'env/cuda.sh').chmod(0o700)
        sys.path.insert(0,str(root))
        from openassembler import build,oracle
        from openassembler.gpu.cuda_driver import Driver
        from openassembler.native_ownership import validate
        from probe import cases,expected,HEADER
        driver=Driver();driver.open()
        records=[];artifacts=io.BytesIO()
        try:
            device=asdict(driver.info())
            if device['cc']!='8.0' or 'A100' not in device['name']:raise ValueError('A100 required')
            with zipfile.ZipFile(artifacts,'w',zipfile.ZIP_DEFLATED) as archive:
                for op,pairs in cases():
                    compare=op.startswith('setp')
                    instruction=(f'{op} %p,%a,%b; selp.u32 %v,1,0,%p;' if compare else f'{op} %v,%a,%b;')
                    text=HEADER+'''.visible .entry probe(.param .u64 out,.param .u64 aa,.param .u64 bb,.param .u32 n) {
.reg .u64 %out,%aa,%bb,%offset,%ptr; .reg .b32 %a,%b,%v,%t,%ct,%nt,%i,%n; .reg .pred %p,%end;
ld.param.u64 %out,[out]; ld.param.u64 %aa,[aa]; ld.param.u64 %bb,[bb]; ld.param.u32 %n,[n];
mov.u32 %t,%tid.x; mov.u32 %ct,%ctaid.x; mov.u32 %nt,%ntid.x; mad.lo.u32 %i,%ct,%nt,%t;
setp.ge.u32 %end,%i,%n; @%end bra DONE;
mul.wide.u32 %offset,%i,4; add.u64 %ptr,%aa,%offset; ld.global.u32 %a,[%ptr];
add.u64 %ptr,%bb,%offset; ld.global.u32 %b,[%ptr];
'''+instruction+'''
add.u64 %ptr,%out,%offset; st.global.u32 [%ptr],%v;
DONE: ret; }
'''
                    name=op.replace('.','_');source=root/(name+'.ptx');source.write_text(text)
                    aa=struct.pack(f'<{len(pairs)}I',*[a for a,b in pairs]);bb=struct.pack(f'<{len(pairs)}I',*[b for a,b in pairs])
                    truth=struct.pack(f'<{len(pairs)}I',*[expected(op,a,b) for a,b in pairs]);guard=bytes([0xa5])*256
                    archive.writestr(name+'/source.ptx',text)
                    for level in ('safe','mined','sched','opt'):
                        directory=root/(name+'-'+level)
                        result=build.compile_paths(source,directory,build.Options(opt_level=level))
                        if result.rc:raise ValueError(f'native compile failed {op}/{level}')
                        entry=json.loads((directory/'manifest.json').read_text())['entries'][0]
                        binary=(directory/entry['cubin']).read_bytes();validate(entry,binary)
                        for kind,key in [('oa','cubin'),('ptxas','reference')]:
                            blob=(directory/entry[key]).read_bytes();module=driver.load(blob)
                            try:
                                fn=driver.function(module,'probe');pointers=[driver.alloc(len(truth)+512) for _ in range(3)]
                                try:
                                    driver.htod(pointers[0],guard+aa+guard);driver.htod(pointers[1],guard+bb+guard)
                                    for poison in (0x00,0x3c,0xff):
                                        driver.htod(pointers[2],guard+bytes([poison])*len(truth)+guard)
                                        driver.launch(fn,((len(pairs)+63)//64,1,1),(64,1,1),
                                                      [C.c_uint64(pointers[i]+256) for i in (2,0,1)]+[C.c_uint32(len(pairs))])
                                        actual=driver.dtoh(pointers[2],len(truth)+512)
                                        unchanged=(driver.dtoh(pointers[0],len(aa)+512)==guard+aa+guard and
                                                   driver.dtoh(pointers[1],len(bb)+512)==guard+bb+guard)
                                        rows=[(i,a,b) for i,(a,b) in enumerate(zip(struct.unpack(f'<{len(pairs)}I',actual[256:-256]),
                                                                                  struct.unpack(f'<{len(pairs)}I',truth))) if a!=b]
                                        records.append({'op':op,'level':level,'compiler':kind,'poison':poison,'pairs':len(pairs),
                                                        'mismatches':len(rows),'first_mismatches':rows[:5],
                                                        'guards':actual[:256]==actual[-256:]==guard,'inputs_unchanged':unchanged,
                                                        'resources':driver.func_attrs(fn),'cubin_sha256':hashlib.sha256(blob).hexdigest()})
                                finally:
                                    for p in pointers:driver.free(p)
                            finally:driver.unload(module)
                            archive.writestr(f'{name}/{level}/{kind}.cubin',blob)
                        archive.writestr(f'{name}/{level}/manifest.json',(directory/'manifest.json').read_bytes())
                        archive.writestr(f'{name}/{level}/oa.sass',(directory/entry['sass']).read_bytes())
            report={'schema':1,'completed_utc':datetime.now(timezone.utc).isoformat(),'device':device,
                    'torch':__import__('torch').__version__,'ptxas':oracle.run_tool('ptxas','--version').stdout,
                    'scope':'Independent integer/float-bit expectations; all output words, input immutability and guards; no timing',
                    'results':records,'passed':all(r['mismatches']==0 and r['guards'] and r['inputs_unchanged'] for r in records),
                    'compiler_hashes':{n:hashlib.sha256(v).hexdigest() for n,v in runtime.items()},
                    'experiment_hashes':{n:hashlib.sha256(v).hexdigest() for n,v in experiment.items()}}
            return report,base64.b64encode(artifacts.getvalue()).decode()
        finally:driver.close()


@app.local_entrypoint()
def main(action: str = 'hardware'):
    import base64,sys
    sys.path.insert(0,str(ROOT))
    from openassembler.transport import runtime_files
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if action=='tests':
        names=['tests/__init__.py','tests/backend_harness.py','tests/test_backend_contracts.py',
               'tests/test_source_diagnostics.py','tests/test_model_graph.py','tests/test_model_discovery.py',
               'tests/test_welcome.py','tests/test_pyfront_static.py']
        names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'examples/native_model').glob('*') if p.is_file()]
        tests={n:(ROOT/n).read_bytes() for n in names}
        result=regress.remote(runtime_files(native=True),tests)
        (HERE/('pinned-tests-'+stamp+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(result['stdout']);print(result['stderr'])
        if result['returncode']:raise ValueError('pinned regression failed')
        return
    if action!='hardware':raise ValueError('action must be hardware or tests')
    report,archive=execute.remote(runtime_files(native=True),{'probe.py':(HERE/'probe.py').read_bytes()})
    payload=base64.b64decode(archive);path=HERE/('a100-'+stamp+'.artifacts.zip');path.write_bytes(payload)
    report['artifact_archive']={'path':path.name,'sha256':hashlib.sha256(payload).hexdigest()}
    report['harness_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result=HERE/('a100-'+stamp+'.json');result.write_text(json.dumps(report,indent=2)+'\n')
    print(result,'passed',report['passed'])
    if not report['passed']:raise ValueError('hardware semantics gate failed')
