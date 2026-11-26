"""Research gate: original separate-unit link versus isolated source merge on A100."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import modal
ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
app=modal.App('oa-device-linkage-baseline')
image=modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04',add_python='3.12')
@app.function(image=image,gpu='A100-40GB',timeout=1200,retries=0,max_containers=1)
def execute(payload):
 import base64,ctypes as C,io,os,random,struct,subprocess,sys,tempfile,zipfile
 from dataclasses import asdict
 with tempfile.TemporaryDirectory(prefix='oa-device-linkage-')as temporary:
  root=Path(temporary)
  for name,data in payload.items():
   p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
  (root/'env/cuda.sh').chmod(0o700);os.chdir(root);sys.path.insert(0,str(root))
  from openassembler import build
  from openassembler.gpu.cuda_driver import Driver
  from openassembler.native_ownership import validate
  driver=Driver();driver.open();device=asdict(driver.info());assert device['cc']=='8.0'and'A100'in device['name']
  records=[];error=None;blob=io.BytesIO();guard=b'\xa5'*256;lanes=384
  try:
   with zipfile.ZipFile(blob,'w',zipfile.ZIP_DEFLATED)as archive:
    for name in('entry','helper'):
     source=root/(name+'.cu');source.write_bytes(payload['compiler_exp/gap_closure/device_linkage/'+name+'.cu'])
     archive.writestr(name+'.cu',source.read_bytes())
     for mode,suffix in(('-ptx','.ptx'),('-dc','.o')):
      cmd=['nvcc','-arch=sm_80','-O3','-rdc=true',mode,'-o',str(source.with_suffix(suffix)),str(source)]
      p=subprocess.run(cmd,capture_output=True,text=True);archive.writestr(name+suffix+'.tool.log',p.stdout+p.stderr);assert p.returncode==0,p.stderr
      archive.writestr(name+suffix,source.with_suffix(suffix).read_bytes())
    cmd=['nvcc','-arch=sm_80','-dlink','-cubin','-o',str(root/'linked.cubin'),str(root/'entry.o'),str(root/'helper.o')]
    p=subprocess.run(cmd,capture_output=True,text=True);archive.writestr('link.tool.log',p.stdout+p.stderr);assert p.returncode==0,p.stderr
    original=(root/'linked.cubin').read_bytes();archive.writestr('linked.cubin',original)
    entry=(root/'entry.ptx').read_text();helper=(root/'helper.ptx').read_text();assert entry.count('.extern .func')==1 and helper.count('.visible .func')==1
    merged=entry.replace('.extern .func','.visible .func')+'\n'+'\n'.join(line for line in helper.splitlines()if not line.startswith(('.version','.target','.address_size')))
    source=root/'merged.ptx';source.write_text(merged);archive.writestr('merged.ptx',merged)
    binaries=[('original-linked',original)]
    for level in('safe','mined','sched','opt'):
     target=root/level;result=build.compile_paths(source,target,build.Options(opt_level=level));assert result.rc==0,result
     manifest=json.loads((target/'manifest.json').read_text());e=manifest['entries'][0];binary=(target/e['cubin']).read_bytes();validate(e,binary)
     archive.writestr(level+'/manifest.json',json.dumps(manifest,indent=2));archive.writestr(level+'/oa.cubin',binary);binaries.append(('oa-'+level,binary))
     if level=='safe':
      control=(target/e['reference']).read_bytes();archive.writestr('merged-ptxas.cubin',control);binaries.append(('merged-ptxas',control))
    for compiler,binary in binaries:
     module=driver.load(binary)
     try:
      fn=driver.function(module,'linked_probe');resources=driver.func_attrs(fn);inp,out=driver.alloc(lanes*4+512),driver.alloc(lanes*4+512)
      try:
       for n in(0,1,259,384):
        for seed in(19271,19272):
         rng=random.Random(seed);data=[0,1,0xffffffff,0x80000000]+[rng.getrandbits(32)for _ in range(lanes-4)];before=guard+struct.pack('<384I',*data)+guard;initial=guard+struct.pack('<I',0x13579bdf)*lanes+guard
         driver.htod(inp,before);driver.htod(out,initial);driver.launch(fn,(3,1,1),(128,1,1),[C.c_uint64(out+256),C.c_uint64(inp+256),C.c_uint32(n)])
         actual=driver.dtoh(out,len(initial));unchanged=driver.dtoh(inp,len(before));reference=[(v*17+9)&0xffffffff if i<n else 0x13579bdf for i,v in enumerate(data)];wanted=guard+struct.pack('<384I',*reference)+guard
         prefix=compiler+'/'+str(n)+'/'+str(seed)+'/'
         for name,raw in(('input-before',before),('input',unchanged),('output',actual)):archive.writestr(prefix+name+'.bin',raw)
         records.append(dict(compiler=compiler,n=n,seed=seed,passed=actual==wanted and unchanged==before,resources=resources,cubin_sha256=hashlib.sha256(binary).hexdigest(),input_sha256=hashlib.sha256(unchanged).hexdigest(),output_sha256=hashlib.sha256(actual).hexdigest()))
      finally:driver.free(inp);driver.free(out)
     finally:driver.unload(module)
  except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc))
  finally:driver.close()
  return dict(schema=1,passed=error is None and len(records)==48 and all(r['passed']for r in records),error=error,results=records,device=device,python=sys.version,nvcc=subprocess.check_output(['nvcc','--version'],text=True),source_hashes={name:hashlib.sha256(data).hexdigest()for name,data in payload.items()},scope='Exact two-source research hypothesis compared with actual original nvcc objects/device link and merged same-PTX controls; no production linker, arbitrary library or performance claim'),base64.b64encode(blob.getvalue()).decode()
@app.local_entrypoint()
def main():
 import base64,sys
 sys.path.insert(0,str(ROOT));from openassembler.transport import runtime_files
 payload=runtime_files(native=True)
 for name in('entry.cu','helper.cu'):payload['compiler_exp/gap_closure/device_linkage/'+name]=(HERE/name).read_bytes()
 harness=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');report,encoded=execute.remote(payload)
 archive=HERE/('research-a100-'+stamp+'.zip');archive.write_bytes(base64.b64decode(encoded));report.update(archive=dict(path=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),harness_sha256=harness)
 path=HERE/('research-a100-'+stamp+'.json');path.write_text(json.dumps(report,indent=2)+'\n');print(path,'passed',report['passed'],'launches',len(report['results']))
 if not report['passed']:raise RuntimeError('device-linkage research gate failed')
