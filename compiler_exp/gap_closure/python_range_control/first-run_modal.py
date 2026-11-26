"""Bounded original-source baseline/candidate range overflow comparison on A100."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import modal
ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
app=modal.App('oa-range-counter-boundaries')
image=modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04',add_python='3.12')
@app.function(image=image,gpu='A100-40GB',timeout=1200,retries=0,max_containers=1)
def execute(payload,sources,compilers):
 import base64,ctypes as C,io,os,struct,subprocess,sys,tempfile,zipfile
 from dataclasses import asdict
 with tempfile.TemporaryDirectory(prefix='oa-range-counter-')as temporary:
  root=Path(temporary)
  for name,data in payload.items():
   path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
  (root/'env/cuda.sh').chmod(0o700);os.chdir(root);sys.path.insert(0,str(root))
  from openassembler.gpu.cuda_driver import Driver
  from openassembler.container.cubin import regcount
  from openassembler.native_ownership import validate
  driver=Driver();driver.open();device=asdict(driver.info());assert device['cc']=='8.0'and'A100'in device['name']
  rows=[];blob=io.BytesIO();error=None;guard=b'\xa5'*64
  try:
   with zipfile.ZipFile(blob,'w',zipfile.ZIP_DEFLATED)as archive:
    original='\n'.join(row['source'].replace('def k(', 'def '+row['name']+'(')for row in sources)
    source=root/'original.py';source.write_text(original);archive.writestr('original.py',original)
    for strategy,compiler in compilers.items():
     (root/'openassembler/pyfront/compile.py').write_bytes(compiler);archive.writestr(strategy+'/compiler.py',compiler);binaries=[]
     for level in('safe','mined','sched','opt'):
      output=root/(strategy+'-'+level)
      result=subprocess.run([sys.executable,'-m','openassembler','compile',str(source),'--out-dir',str(output),'--opt-level',level,'--no-sass'],capture_output=True,text=True)
      archive.writestr(strategy+'/'+level+'/build.log',result.stdout+result.stderr);assert result.returncode==0,result.stdout+result.stderr
      manifest=json.loads((output/'manifest.json').read_text());entry,=manifest['entries'];binary=(output/entry['cubin']).read_bytes();validate(entry,binary)
      prefix=strategy+'/'+level+'/';archive.writestr(prefix+'source.ptx',(output/entry['ptx']).read_bytes());archive.writestr(prefix+'manifest.json',json.dumps(manifest,indent=2));archive.writestr(prefix+'oa.cubin',binary);binaries.append(('oa-'+level,binary))
      if level=='safe':
       binary=(output/entry['reference']).read_bytes();archive.writestr(strategy+'/ptxas.cubin',binary);binaries.append(('ptxas',binary))
     for backend,binary in binaries:
      file=root/(strategy+'-'+backend+'.cubin');file.write_bytes(binary);archive.writestr(strategy+'/'+backend+'.sass',subprocess.check_output(['nvdisasm','-hex','-c',str(file)]));module=driver.load(binary)
      try:
       for row in sources:
        name=row['name'];fn=driver.function(module,name);address=driver.alloc(152)
        try:
         driver.htod(address,guard+bytes(24)+guard);driver.launch(fn,(1,1,1),(1,1,1),[C.c_uint64(address+64)])
         actual=driver.dtoh(address,152);wanted=bytes.fromhex(row['expected_hex']);assert actual[:64]==actual[-64:]==guard
         correct=actual==wanted;expected_correct=strategy=='candidate'or name in('ordinary','empty');assert correct==expected_correct,(strategy,backend,name,actual.hex(),wanted.hex())
         archive.writestr(f'{strategy}/{backend}/{name}/output.bin',actual)
         rows.append(dict(strategy=strategy,backend=backend,entry=name,matches_reference=correct,expected_matches_reference=expected_correct,cubin_sha256=hashlib.sha256(binary).hexdigest(),output_sha256=hashlib.sha256(actual).hexdigest(),registers=regcount(binary,name),resources=driver.func_attrs(fn),passed=True))
        finally:driver.free(address)
      finally:driver.unload(module)
  except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc))
  finally:driver.close()
  return dict(schema=1,passed=error is None and len(rows)==60 and all(r['passed']for r in rows),error=error,results=rows,device=device,python=sys.version,nvcc=subprocess.check_output(['nvcc','--version'],text=True),payload_hashes={n:hashlib.sha256(d).hexdigest()for n,d in payload.items()},compiler_hashes={n:hashlib.sha256(d).hexdigest()for n,d in compilers.items()},source_sha256=hashlib.sha256(original.encode()).hexdigest(),scope='60 bounded A100 original Python launches across frozen baseline/candidate, six finite ranges, emitted PTX and four OA levels. Four baseline wrap counterexamples are required mismatches; all candidate and ordinary/empty controls require complete bytes. Eight-trip watchdog, no production promotion or timing claim'),base64.b64encode(blob.getvalue()).decode()
@app.local_entrypoint()
def main():
 import base64,sys
 sys.path.insert(0,str(ROOT));from openassembler.transport import runtime_files
 baseline=json.loads((HERE/'baseline.json').read_text());assert baseline['passed']and baseline['counterexamples']==4
 compilers={name:(HERE/(name+'-pyfront.py')).read_bytes()for name in('baseline','candidate')}
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');report,encoded=execute.remote(runtime_files(native=True),baseline['results'],compilers)
 archive=HERE/('candidate-a100-'+stamp+'.zip');archive.write_bytes(base64.b64decode(encoded));report.update(archive=dict(path=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reference_sha256=hashlib.sha256((HERE/'baseline.json').read_bytes()).hexdigest())
 path=HERE/('candidate-a100-'+stamp+'.json');path.write_text(json.dumps(report,indent=2)+'\n');print(path,'passed',report['passed'],'launches',len(report['results']))
 if not report['passed']:raise RuntimeError('bounded range counter gate failed')
