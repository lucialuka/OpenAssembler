"""Original logical Boolean helpers through CLI and serialized worker on A100."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import modal
ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
app=modal.App('oa-logical-boolean-helpers')
image=modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04',add_python='3.12')
@app.function(image=image,gpu='A100-40GB',timeout=1800,retries=0,max_containers=1)
def execute(payload):
 import base64,ctypes as C,io,os,struct,subprocess,sys,tempfile,zipfile
 from dataclasses import asdict
 with tempfile.TemporaryDirectory(prefix='oa-python-source-graph-')as temp:
  root=Path(temp)
  for name,data in payload.items():
   path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
  (root/'env/cuda.sh').chmod(0o700);os.chdir(root);sys.path[:0]=[str(root),str(root/'compiler_exp/gap_closure/python_boolean_abi/helpers')]
  from fixtures import configurations,write_tree,NAMES,raw_input,initial,expected,expected_counter,BODIES
  from openassembler import build
  from openassembler.cli import main as cli
  from openassembler.remote_build import _sources,process
  from openassembler.torchfront.archive import pack,unpack
  from openassembler.native_ownership import validate
  from openassembler.gpu.cuda_driver import Driver
  from openassembler.backend.storage import local_frame_inventory
  driver=Driver();driver.open();device=asdict(driver.info());assert device['cc']=='8.0'and'A100'in device['name']
  blob=io.BytesIO();rows=[];error=None;guard=b'\xa5'*256;lanes=384
  try:
   with zipfile.ZipFile(blob,'w',zipfile.ZIP_DEFLATED)as archive:
    for case in configurations():
     folder=root/case['name'];source=write_tree(folder,case)
     for name,text in case['files'].items():archive.writestr(case['name']+'/original/'+name,text)
     binaries=[]
     for level in('safe','mined','sched','opt'):
      output=folder/('output-'+level);options=build.Options(opt_level=level,python_source_root=folder if case['explicit']else None)
      if case['remote']:
       origin,files,includes=_sources(source,output,options);archive.writestr(case['name']+'/'+level+'/source-bundle.zip',pack(files))
       request=dict(source=base64.b64encode(pack(files)).decode(),path='source/'+source.relative_to(origin).as_posix(),options={**asdict(options),'include_dirs':includes,'python_source_root':'source'if case['explicit']else None},name=None)
       result=process(request);assert not result.get('error')and result['rc']==0,result
       output.mkdir();unpack(base64.b64decode(result['artifacts']),output)
      else:
       args=['compile',str(source.parent if case.get('directory')else source),'--out-dir',str(output),'--opt-level',level]
       if case['explicit']:args+=['--python-source-root',str(folder)]
       assert cli(args)==0,(case['name'],level)
      manifest=json.loads((output/'manifest.json').read_text());entry,=manifest['entries'];assert[k['name']for k in entry['kernels']]==list(NAMES)
      prefix=case['name']+'/'+level+'/'
      emitted=(output/entry['ptx']).read_bytes();archive.writestr(prefix+'source.ptx',emitted);archive.writestr(prefix+'manifest.json',json.dumps(manifest,indent=2))
      binary=(output/entry['cubin']).read_bytes();validate(entry,binary);archive.writestr(prefix+'oa.cubin',binary);binaries.append(('oa-'+level,binary,entry))
      if level=='safe':
       reference=(output/entry['reference']).read_bytes();archive.writestr(case['name']+'/ptxas.cubin',reference);binaries.append(('ptxas',reference,entry))
     for compiler,binary,entry in binaries:
      file=folder/(compiler+'.cubin');file.write_bytes(binary);archive.writestr(case['name']+'/'+compiler+'.sass',subprocess.check_output(['nvdisasm','-hex','-c',str(file)]))
      module=driver.load(binary)
      try:
       for kernel in entry['kernels']:
        name=kernel['name'];fn=driver.function(module,name);resources=driver.func_attrs(fn);frame=local_frame_inventory(binary)[name]if compiler.startswith('oa-')else None
        if compiler.startswith('oa-'):assert frame==resources['local_bytes']==0 and kernel['python_source_graph']['strategy']=='static-local-python-helper-source-graph'
        width=8 if BODIES[name][0].endswith('64')else 4;ap,out,cp=driver.alloc(lanes*width+512),driver.alloc(lanes*width+512),driver.alloc(lanes*4+512)
        try:
         for seed in(31881,31882):
          a=raw_input(name,seed,lanes);before_a=guard+a+guard
          for n in(0,1,259,384):
           driver.htod(ap,before_a);driver.htod(out,guard+initial(name,lanes)+guard);driver.htod(cp,guard+bytes(lanes*4)+guard)
           driver.launch(fn,(3,1,1),(128,1,1),[C.c_uint64(ap+256),C.c_uint64(out+256),C.c_uint64(cp+256),C.c_uint32(n)])
           actual,after_a,after_counter=driver.dtoh(out,lanes*width+512),driver.dtoh(ap,lanes*width+512),driver.dtoh(cp,lanes*4+512)
           wanted=guard+expected(name,seed,n,lanes)+guard;run=f'{case["name"]}/{compiler}/{name}/{seed}/{n}/'
           for key,data in(('before-a',before_a),('a',after_a),('output',actual),('counter',after_counter)):archive.writestr(run+key+'.bin',data)
           rows.append(dict(configuration=case['name'],compiler=compiler,entry=name,seed=seed,n=n,passed=actual==wanted and after_a==before_a and after_counter==guard+expected_counter(name,seed,n,lanes)+guard,resources=resources,local_frame_bytes=frame,cubin_sha256=hashlib.sha256(binary).hexdigest(),output_sha256=hashlib.sha256(actual).hexdigest()))
        finally:driver.free(ap);driver.free(out);driver.free(cp)
      finally:driver.unload(module)
  except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc))
  finally:driver.close()
  return dict(schema=1,passed=error is None and len(rows)==2400 and all(r['passed']for r in rows),error=error,results=rows,device=device,python=sys.version,nvcc=subprocess.check_output(['nvcc','--version'],text=True),source_hashes={name:hashlib.sha256(data).hexdigest()for name,data in payload.items()},scope='Original logical Boolean helper signatures and complete predicate return paths in same-file/imported/package source graphs through ordinary CLI and worker; independent truth/single-evaluation/short-circuit/atomic/memory snapshots modulo64/FP32 bytes, ownership and unchanged inputs; A100 correctness only, no timing'),base64.b64encode(blob.getvalue()).decode()
@app.local_entrypoint()
def main():
 import base64,sys
 sys.path.insert(0,str(ROOT));from openassembler.transport import runtime_files
 payload=runtime_files(native=True)
 for target,file in{'openassembler/pyfront/helpers.py':'candidate-helpers.py','openassembler/pyfront/compile.py':'candidate-pyfront.py','openassembler/pyfront/modules.py':'candidate-modules.py','compiler_exp/gap_closure/python_boolean_abi/helpers/fixtures.py':'fixtures.py'}.items():payload[target]=(HERE/file).read_bytes()

 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');report,encoded=execute.remote(payload)
 archive=HERE/('candidate-a100-'+stamp+'.zip');archive.write_bytes(base64.b64decode(encoded));report.update(archive=dict(path=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 path=HERE/('candidate-a100-'+stamp+'.json');path.write_text(json.dumps(report,indent=2)+'\n');print(path,'passed',report['passed'],'launches',len(report['results']))
 if not report['passed']:raise RuntimeError('Python helper graph gate failed')
