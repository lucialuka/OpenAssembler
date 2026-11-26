"""Isolated A100 ordering/shared probes; never edits production or claims a model proof."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import modal

ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
app=modal.App('openassembler-atomic-ordering-research')
image=modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04',add_python='3.12')


@app.function(image=image,gpu='A100-40GB',timeout=1800,retries=0,max_containers=1)
def execute(files):
    import base64,ctypes as C,io,os,struct,subprocess,sys,tempfile,zipfile
    from collections import Counter
    from dataclasses import asdict
    with tempfile.TemporaryDirectory(prefix='oa-ordering-') as temporary:
        root=Path(temporary)
        for name,data in files.items():
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        (root/'env/cuda.sh').chmod(0o700);os.chdir(root)
        sys.path[:0]=[str(root),str(root/'compiler_exp/gap_closure/atomic_ordering'),str(root/'compiler_exp/gap_closure/atomics')]
        from probes import HEADER,EPOCHS,WORDS,publication_source,shared_source
        from cases import exchange_chain
        from openassembler import build
        from openassembler.gpu.cuda_driver import Driver
        from openassembler.native_ownership import validate
        fixtures=[('publication',publication_source()),('both_acqrel',publication_source('both_acqrel',release='acq_rel',acquire='acq_rel'))]
        fixtures += [(name,shared_source(name,operation=operation,explicit=explicit))
                     for name,operation,explicit in [('shared_add','add',False),('shared_add_explicit','add',True),
                                                     ('shared_exch','exch',False),('shared_exch_explicit','exch',True)]]
        combined=HEADER+'\n'.join(text[len(HEADER):] for _,text in fixtures)
        source=root/'probes.ptx';source.write_text(combined)
        records=[];blob=io.BytesIO();guard=b'\xa5'*256
        driver=Driver();driver.open();failed_launch=False
        class ProbeStopped(Exception):pass
        try:
            device=asdict(driver.info())
            if device['cc']!='8.0' or 'A100' not in device['name']:raise ValueError('A100 required')
            with zipfile.ZipFile(blob,'w',zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('probes.ptx',combined)
                for level in ('safe','mined','sched','opt'):
                    directory=root/level
                    result=build.compile_paths(source,directory,build.Options(opt_level=level))
                    if result.rc:raise RuntimeError(f'candidate compilation failed {level}: {result}')
                    manifest=json.loads((directory/'manifest.json').read_text());entry=manifest['entries'][0]
                    archive.writestr(level+'/manifest.json',json.dumps(manifest,indent=2))
                    for compiler,key in [('oa','cubin'),('ptxas','reference')]:
                        binary=(directory/entry[key]).read_bytes()
                        if compiler=='oa':validate(entry,binary)
                        archive.writestr(level+'/'+compiler+'.cubin',binary)
                        dis=subprocess.check_output(['nvdisasm','-hex','-c',str(directory/entry[key])],text=True)
                        archive.writestr(level+'/'+compiler+'.sass.txt',dis)
                        module=driver.load(binary)
                        try:
                            for name,_ in fixtures:
                                fn=driver.function(module,name)
                                shared=name.startswith('shared')
                                shapes={'output':(2048+4*17*4)} if shared else {'output':12,'payload':WORDS*4,'flag':4,'ack':4}
                                pointers={key:driver.alloc(size+512) for key,size in shapes.items()}
                                try:
                                    for repeat in range(3 if shared else 16):
                                        for key,size in shapes.items():driver.htod(pointers[key],guard+b'\x00'*size+guard)
                                        args=[C.c_uint64(pointers[key]+256) for key in shapes]
                                        try:
                                            driver.launch(fn,(4,1,1) if shared else (2,1,1),(128,1,1) if shared else (32,1,1),args)
                                        except Exception as exc:
                                            failed_launch=True
                                            failure={'name':name,'level':level,'compiler':compiler,'repeat':repeat,
                                                     'passed':False,'error':type(exc).__name__+': '+str(exc)}
                                            records.append(failure);archive.writestr('failed-launch.json',json.dumps(failure,indent=2))
                                            raise ProbeStopped() from None
                                        raw={key:driver.dtoh(pointers[key],size+512) for key,size in shapes.items()}
                                        actual={key:data[256:-256] for key,data in raw.items()}
                                        failures=[];guards=all(data[:256]==data[-256:]==guard for data in raw.values())
                                        if shared:
                                            output=struct.unpack('<'+'I'*(len(actual['output'])//4),actual['output'])
                                            for cta in range(4):
                                                for tid in range(128):
                                                    if tid%2 and output[cta*128+tid]!=0x13579bdf:failures.append('false shared predicate modified result')
                                                for bin_index in range(17):
                                                    lanes=[tid for tid in range(bin_index,128,17) if tid%2==0]
                                                    returns=[output[cta*128+tid] for tid in lanes]
                                                    final=output[512+cta*17+bin_index]
                                                    if 'add' in name:
                                                        if final!=len(lanes) or Counter(returns)!=Counter(range(len(lanes))):failures.append('shared old-value add tickets')
                                                    elif not exchange_chain(0,final,[cta*128+tid+1 for tid in lanes],returns):failures.append('shared exchange linearization')
                                        else:
                                            status=struct.unpack('<3I',actual['output'])
                                            if status[0]!=0 or status[1]!=0:failures.append('publication timeout or stale epoch payload')
                                            if struct.unpack('<I',actual['flag'])[0]!=EPOCHS or struct.unpack('<I',actual['ack'])[0]!=EPOCHS:failures.append('incomplete handshake epochs')
                                            expected=[(EPOCHS*0x1f123bb5+i)&0xffffffff for i in range(WORDS)]
                                            if list(struct.unpack('<'+'I'*WORDS,actual['payload']))!=expected:failures.append('final payload mismatch')
                                        for key,data in raw.items():archive.writestr(f'{level}/{compiler}/{name}/{repeat}/{key}.bin',data)
                                        records.append({'name':name,'level':level,'compiler':compiler,'repeat':repeat,'guards':guards,
                                                        'passed':guards and not failures,'failures':failures,
                                                        'output_sha256':{key:hashlib.sha256(data).hexdigest() for key,data in raw.items()},
                                                        'cubin_sha256':hashlib.sha256(binary).hexdigest(),'resources':driver.func_attrs(fn)})
                                finally:
                                    if not failed_launch:
                                        for pointer in pointers.values():driver.free(pointer)
                        finally:
                            if not failed_launch:driver.unload(module)
        except ProbeStopped:pass
        finally:
            if not failed_launch:driver.close()
        return {'schema':1,'python':sys.version,'ptxas':subprocess.check_output(['ptxas','--version'],text=True),
                'device':device,'scope':'bounded two-CTA epoch visibility and multi-warp CTA shared atomic invariants; not a general memory-model proof or timing benchmark',
                'passed':all(r['passed'] for r in records),'results':records,
                'source_hashes':{name:hashlib.sha256(data).hexdigest() for name,data in files.items()}},base64.b64encode(blob.getvalue()).decode()


@app.local_entrypoint()
def main():
    import base64,sys
    sys.path.insert(0,str(ROOT))
    from openassembler.transport import runtime_files
    files=runtime_files(native=True)
    for name in ('ra.py','contracts.py','compile.py','sched.py'):
        files['openassembler/backend/'+name]=(HERE/('candidate-'+name)).read_bytes()
    files['compiler_exp/gap_closure/atomic_ordering/probes.py']=(HERE/'probes.py').read_bytes()
    files['compiler_exp/gap_closure/atomics/cases.py']=(HERE.parent/'atomics/cases.py').read_bytes()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report,payload=execute.remote(files)
    archive=HERE/('a100-'+stamp+'.zip');archive.write_bytes(base64.b64decode(payload))
    report['archive']={'path':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
    report['harness_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output=HERE/('a100-'+stamp+'.json');output.write_text(json.dumps(report,indent=2)+'\n')
    print(output,'passed',report['passed'],'launches',len(report['results']))
    if not report['passed']:raise RuntimeError('ordering/shared candidate failed retained gates')
