"""Independent A100 truth and atomic effect bytes for original condition chains."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import modal
ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
app=modal.App('oa-original-condition-chains');image=modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04',add_python='3.12')
@app.function(image=image,gpu='A100-40GB',timeout=1200,retries=0,max_containers=1)
def execute(payload,cases):
 import base64,ctypes as C,io,os,subprocess,sys,tempfile,zipfile
 from dataclasses import asdict
 with tempfile.TemporaryDirectory(prefix='oa-chain-effects-')as temporary:
  root=Path(temporary)
  for name,data in payload.items():
   path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
  os.chdir(root);sys.path.insert(0,str(root));(root/'env/cuda.sh').chmod(0o700)
  from openassembler.cli import main as cli
  from openassembler.gpu.cuda_driver import Driver
  from openassembler.native_ownership import validate
  driver=Driver();driver.open();device=asdict(driver.info());assert device['cc']=='8.0'and'A100'in device['name'];rows=[];blob=io.BytesIO();error=None
  try:
   with zipfile.ZipFile(blob,'w',zipfile.ZIP_DEFLATED)as archive:
    for case in cases:
     name=case['dtype'];folder=root/name;folder.mkdir();source=folder/'source.py';source.write_text(case['source']);archive.writestr(name+'/source.py',case['source']);binaries=[]
     for level in('safe','mined','sched','opt'):
      out=folder/level;assert cli(['compile',str(source),'--out-dir',str(out),'--opt-level',level,'--no-sass'])==0
      manifest=json.loads((out/'manifest.json').read_text());entry,=manifest['entries'];binary=(out/entry['cubin']).read_bytes();validate(entry,binary);archive.writestr(name+'/'+level+'/source.ptx',(out/entry['ptx']).read_bytes());archive.writestr(name+'/'+level+'/manifest.json',json.dumps(manifest,indent=2));archive.writestr(name+'/'+level+'/oa.cubin',binary);binaries.append(('oa-'+level,binary))
      if level=='safe':
       binary=(out/entry['reference']).read_bytes();archive.writestr(name+'/ptxas.cubin',binary);binaries.append(('ptxas',binary))
     for backend,binary in binaries:
      path=folder/(backend+'.cubin');path.write_bytes(binary);archive.writestr(name+'/'+backend+'.sass',subprocess.check_output(['nvdisasm','-hex','-c',str(path)]));module=driver.load(binary)
      try:
       fn=driver.function(module,'k');before=bytes.fromhex(case['input_hex']);wanted=bytes.fromhex(case['expected_hex']);initial=b'\xa5'*64+bytes(len(wanted)-128)+b'\xa5'*64;ap,out=driver.alloc(len(before)),driver.alloc(len(wanted));cp=None
       try:
        driver.htod(ap,before);driver.htod(out,initial);args=[C.c_uint64(ap+64),C.c_uint64(out+64)]
        if 'counter_hex'in case:cp=driver.alloc(len(initial));driver.htod(cp,initial);args.append(C.c_uint64(cp+64))
        driver.launch(fn,(1,1,1),(case['lanes'],1,1),args);actual,after=driver.dtoh(out,len(wanted)),driver.dtoh(ap,len(before));counter=driver.dtoh(cp,len(initial))if cp else None
        for key,data in(('input',after),('output',actual)):
         archive.writestr(f'{name}/{backend}/{key}.bin',data)
        if counter is not None:archive.writestr(f'{name}/{backend}/counter.bin',counter)
        assert actual==wanted and after==before and(counter is None or counter==bytes.fromhex(case['counter_hex'])),(name,backend)
        rows.append(dict(dtype=name,backend=backend,output_sha256=hashlib.sha256(actual).hexdigest(),input_sha256=hashlib.sha256(after).hexdigest(),counter_sha256=hashlib.sha256(counter).hexdigest()if counter is not None else None,cubin_sha256=hashlib.sha256(binary).hexdigest(),resources=driver.func_attrs(fn),passed=True))
       finally:
        driver.free(ap);driver.free(out)
        if cp:driver.free(cp)
      finally:driver.unload(module)
  except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc))
  finally:driver.close()
  return dict(schema=1,passed=error is None and len(rows)==30,error=error,results=rows,device=device,python=sys.version,ptxas=subprocess.check_output(['ptxas','--version'],text=True),source_hashes={n:hashlib.sha256(d).hexdigest()for n,d in payload.items()},scope='30 original condition-chain A100 executions across five scalar/special-value domains plus explicit atomic single-evaluation/short-circuit effects, emitted PTX controls and four OA levels, exact whole input/output/counter/guard bytes. Finite research gate only, no production/source-graph/remote integration or performance claim'),base64.b64encode(blob.getvalue()).decode()
@app.local_entrypoint()
def main():
 import base64,struct,sys
 sys.path.insert(0,str(ROOT));from openassembler.transport import runtime_files
 domains=json.loads((HERE/'domains.json').read_text());atomic=json.loads((HERE/'candidate-offline.json').read_text());assert domains['passed']and atomic['passed'];cases=domains['cases'];guard=b'\xa5'*64;cases.append(dict(dtype='atomic',source=atomic['source'],lanes=8,input_hex=(guard+struct.pack('<8i',*atomic['inputs'])+guard).hex(),expected_hex=(guard+struct.pack('<8i',*atomic['expected_outputs'])+guard).hex(),counter_hex=(guard+struct.pack('<8i',*atomic['expected_atomic_counters'])+guard).hex()))
 payload=runtime_files(native=True);payload['openassembler/pyfront/compile.py']=(HERE/'candidate-pyfront.py').read_bytes();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');report,encoded=execute.remote(payload,cases);archive=HERE/('candidate-a100-'+stamp+'.zip');archive.write_bytes(base64.b64decode(encoded));report.update(archive=dict(path=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reference_hashes={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest()for n in('domains.json','candidate-offline.json')});path=HERE/('candidate-a100-'+stamp+'.json');path.write_text(json.dumps(report,indent=2)+'\n');print(path,'passed',report['passed'],'launches',len(report['results']))
 if not report['passed']:raise RuntimeError('condition-chain hardware gate failed')
