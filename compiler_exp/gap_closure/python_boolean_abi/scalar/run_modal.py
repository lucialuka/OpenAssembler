"""Original logical Boolean arguments through CLI/worker and public run specs on A100."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import modal
ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
app=modal.App('oa-logical-boolean-parameters')
image=modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04',add_python='3.12')

@app.function(image=image,gpu='A100-40GB',timeout=1800,retries=0,max_containers=1)
def execute(payload):
 import base64,ctypes as C,io,os,struct,subprocess,sys,tempfile,zipfile
 from dataclasses import asdict
 with tempfile.TemporaryDirectory(prefix='oa-boolean-abi-')as temp:
  root=Path(temp)
  for name,data in payload.items():
   path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
  (root/'env/cuda.sh').chmod(0o700);os.chdir(root);sys.path[:0]=[str(root),str(root/'compiler_exp/gap_closure/python_boolean_abi/scalar')]
  from fixtures import configurations,write_tree,NAMES,BODIES,parameters,values,raw_input,initial,expected,expected_counter,width
  from openassembler import build
  from openassembler.cli import main as cli
  from openassembler.remote_build import _sources,process
  from openassembler.torchfront.archive import pack,unpack
  from openassembler.native_ownership import validate
  from openassembler.gpu.cuda_driver import Driver,run_plan_job
  from openassembler.runner import RunOptions,build_plan
  driver=Driver();driver.open();device=asdict(driver.info());assert device['cc']=='8.0'and'A100'in device['name']
  blob=io.BytesIO();rows=[];spec_rows=[];error=None;guard=b'\xa5'*256;lanes=384
  try:
   with zipfile.ZipFile(blob,'w',zipfile.ZIP_DEFLATED)as archive:
    for case in configurations():
     folder=root/case['name'];source=write_tree(folder,case)
     for name,text in case['files'].items():archive.writestr(case['name']+'/original/'+name,text)
     binaries=[]
     for level in('safe','mined','sched','opt'):
      output=folder/('output-'+level);options=build.Options(opt_level=level)
      if case['remote']:
       origin,files,includes=_sources(source,output,options);archive.writestr(case['name']+'/'+level+'/source-bundle.zip',pack(files))
       result=process(dict(source=base64.b64encode(pack(files)).decode(),path='source/'+source.relative_to(origin).as_posix(),options={**asdict(options),'include_dirs':includes},name=None))
       assert not result.get('error')and result['rc']==0,result
       output.mkdir();unpack(base64.b64decode(result['artifacts']),output)
      else:assert cli(['compile',str(source),'--out-dir',str(output),'--opt-level',level])==0
      manifest=json.loads((output/'manifest.json').read_text());entry,=manifest['entries'];assert[k['name']for k in entry['kernels']]==list(NAMES)
      prefix=case['name']+'/'+level+'/';archive.writestr(prefix+'source.ptx',(output/entry['ptx']).read_bytes());archive.writestr(prefix+'manifest.json',json.dumps(manifest,indent=2))
      binary=(output/entry['cubin']).read_bytes();validate(entry,binary);archive.writestr(prefix+'oa.cubin',binary);binaries.append(('oa-'+level,binary,entry))
      if level=='safe':reference=(output/entry['reference']).read_bytes();archive.writestr(case['name']+'/ptxas.cubin',reference);binaries.append(('ptxas',reference,entry))
      # Execute actual public spec and batch-driver binding for all flag pairs.
      for name in NAMES:
       for first,second in((False,False),(False,True),(True,False),(True,True)):
        vals=values(name,first,second,259);arrays={};initials={}
        for key,data in [('a',raw_input(name,73117,lanes)),('out',initial(name,lanes)),('counter',bytes(lanes*4))]:
         data=guard+data+guard;size=4 if key=='counter'else width(name);fmt='I'if size==4 else'Q';arrays[key]=dict(dtype='u32'if size==4 else'u64',count=len(data)//size,init=list(struct.unpack('<'+str(len(data)//size)+fmt,data)));initials[key]=data
        args=[]
        for key,dtype in parameters(name):
         if dtype.startswith('Array'):args.append(dict(buf=key,offset=256))
         elif dtype=='bool':args.append(dict(bool=vals[key]))
         else:args.append({dtype:vals[key]})
        spec=dict(buffers=arrays,steps=[dict(kernel=name,grid=3,block=128,args=args)],compare=list(arrays),reps=0);path=folder/'launch.json';path.write_text(json.dumps(spec));plan,_=build_plan(RunOptions(out_dir=output,spec=path,reps=0))
        result=run_plan_job(driver,plan.jobs[0]);actual={key:b''.join(spans)for key,spans in result['out'].items()}
        wanted=dict(a=initials['a'],out=guard+expected(name,73117,first,second,259,lanes)+guard,counter=guard+expected_counter(name,73117,first,second,259,lanes)+guard)
        prefix=f'{case["name"]}/explicit/{level}/{name}/{int(first)}{int(second)}/';archive.writestr(prefix+'spec.json',json.dumps(spec))
        for key,data in actual.items():archive.writestr(prefix+key+'.bin',data)
        spec_rows.append(dict(configuration=case['name'],entry=name,level=level,first=first,second=second,passed=actual==wanted))
     for compiler,binary,entry in binaries:
      file=folder/(compiler+'.cubin');file.write_bytes(binary);archive.writestr(case['name']+'/'+compiler+'.sass',subprocess.check_output(['nvdisasm','-hex','-c',str(file)]));loaded=driver.load(binary)
      try:
       for kernel in entry['kernels']:
        name=kernel['name'];assert any(row['dtype']=='bool'for row in kernel['logical_parameter_contract']['parameters']);fn=driver.function(loaded,name);resources=driver.func_attrs(fn)
        ap,out,cp=driver.alloc(lanes*width(name)+512),driver.alloc(lanes*width(name)+512),driver.alloc(lanes*4+512)
        try:
         for seed in(73117,73118):
          before=guard+raw_input(name,seed,lanes)+guard
          for first,second in((False,False),(False,True),(True,False),(True,True)):
           for n in(0,1,259,384):
            args=values(name,first,second,n,ap+256,out+256,cp+256);driver.htod(ap,before);driver.htod(out,guard+initial(name,lanes)+guard);driver.htod(cp,guard+bytes(lanes*4)+guard)
            cargs=[(C.c_uint8 if dtype=='bool'else C.c_uint64 if dtype.startswith('Array')or dtype.endswith('64')else C.c_uint32)(int(args[key]))for key,dtype in parameters(name)]
            driver.launch(fn,(3,1,1),(128,1,1),cargs)
            actual,after,counter=driver.dtoh(out,lanes*width(name)+512),driver.dtoh(ap,lanes*width(name)+512),driver.dtoh(cp,lanes*4+512);prefix=f'{case["name"]}/{compiler}/{name}/{seed}/{int(first)}{int(second)}/{n}/'
            for key,data in(('before-a',before),('a',after),('output',actual),('counter',counter)):archive.writestr(prefix+key+'.bin',data)
            rows.append(dict(configuration=case['name'],compiler=compiler,entry=name,seed=seed,first=first,second=second,n=n,passed=after==before and actual==guard+expected(name,seed,first,second,n,lanes)+guard and counter==guard+expected_counter(name,seed,first,second,n,lanes)+guard,resources=resources,cubin_sha256=hashlib.sha256(binary).hexdigest(),output_sha256=hashlib.sha256(actual).hexdigest()))
        finally:driver.free(ap);driver.free(out);driver.free(cp)
      finally:driver.unload(loaded)
  except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc))
  finally:driver.close()
  return dict(schema=1,passed=error is None and len(rows)==9600 and len(spec_rows)==960 and all(row['passed']for row in rows+spec_rows),error=error,results=rows,explicit_launch_results=spec_rows,device=device,python=sys.version,nvcc=subprocess.check_output(['nvcc','--version'],text=True),source_hashes={name:hashlib.sha256(data).hexdigest()for name,data in payload.items()},scope='Three unexecuted original Boolean parameter trees through normal CLI and worker, every flag pair and mixed numeric/wide slot, independent complete return/effect/raw FP32/guard/tail bytes, source/logical/native ownership; 960 actual public explicit-run launches; A100 correctness without timing'),base64.b64encode(blob.getvalue()).decode()

@app.local_entrypoint()
def main():
 import base64,sys
 sys.path.insert(0,str(ROOT));from openassembler.transport import runtime_files
 payload=runtime_files(native=True)
 for suffix,name in json.loads((HERE/'inputs.json').read_text()).items():payload[name]=(HERE/('candidate-'+suffix+'.py')).read_bytes()
 payload['compiler_exp/gap_closure/python_boolean_abi/scalar/fixtures.py']=(HERE/'fixtures.py').read_bytes()
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');report,encoded=execute.remote(payload);archive=HERE/('candidate-a100-'+stamp+'.zip');archive.write_bytes(base64.b64decode(encoded));report.update(archive=dict(path=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());path=HERE/('candidate-a100-'+stamp+'.json');path.write_text(json.dumps(report,indent=2)+'\n');print(path,'passed',report['passed'],'launches',len(report['results']),'explicit',len(report['explicit_launch_results']))
 if not report['passed']:raise RuntimeError('Boolean scalar parameter gate failed')
