"""Original separate-module controls versus ordinary staged library builds on A100."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import modal
ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
app=modal.App('oa-device-library-promotion')
image=modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04',add_python='3.12')
@app.function(image=image,gpu='A100-40GB',timeout=1800,retries=0,max_containers=1)
def execute(payload):
 import base64,ctypes as C,io,os,struct,subprocess,sys,tempfile,zipfile
 from dataclasses import asdict
 with tempfile.TemporaryDirectory(prefix='oa-device-library-')as temporary:
  root=Path(temporary)
  for name,data in payload.items():
   path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
  (root/'env/cuda.sh').chmod(0o700);os.chdir(root);sys.path[:0]=[str(root),str(root/'compiler_exp/gap_closure/device_linkage/promotion')]
  from fixtures import cases,values,expected,SENTINEL
  from openassembler import build
  from openassembler.cli import main as cli
  from openassembler.remote_build import _sources,process
  from openassembler.torchfront.archive import pack,unpack
  from openassembler.gpu.cuda_driver import Driver
  from openassembler.native_ownership import validate
  from openassembler.backend.storage import local_frame_inventory
  fixtures=list(cases())
  cuda_root='''extern "C" __device__ unsigned helper(unsigned);
extern "C" __global__ void probe(unsigned long long*out,const unsigned long long*input,unsigned n){unsigned i=blockIdx.x*128+threadIdx.x;if(i<n){unsigned long long x=input[i];out[i]=(x&0xffffffff00000000ULL)|helper((unsigned)x);}}'''
  cuda_lib='#include "settings.cuh"\nextern "C" __device__ __noinline__ unsigned helper(unsigned x){return x*MUL+OFFSET;}\n'
  for name in('cuda_database','cuda_remote'):fixtures.append(dict(name=name,root=cuda_root,libraries=[cuda_lib],cuda=True))
  driver=Driver();driver.open();device=asdict(driver.info());assert device['cc']=='8.0'and'A100'in device['name']
  rows=[];error=None;blob=io.BytesIO();guard=b'\xa5'*256;lanes=384
  def tool(args,archive,prefix):
   p=subprocess.run(args,capture_output=True,text=True);archive.writestr(prefix+'.log',p.stdout+p.stderr);assert p.returncode==0,(args,p.stderr)
  try:
   with zipfile.ZipFile(blob,'w',zipfile.ZIP_DEFLATED)as archive:
    for case in fixtures:
     folder=root/case['name'];folder.mkdir();sources=[];objects=[];is_cuda=case.get('cuda',False)
     for i,text in enumerate([case['root'],*case['libraries']]):
      path=folder/(f'unit{i}'+('.cu'if is_cuda else'.ptx'));path.write_text(text);sources.append(path);archive.writestr(case['name']+'/'+path.name,text)
     database=None
     if is_cuda:
      (folder/'settings.cuh').write_text('#define OFFSET 9u\n');archive.writestr(case['name']+'/settings.cuh',(folder/'settings.cuh').read_bytes())
      database=folder/'compile_commands.json';database.write_text(json.dumps([dict(directory=str(folder),file=str(p),arguments=['nvcc','-c','-O3','-DMUL=17',str(p)])for p in sources]));archive.writestr(case['name']+'/compile_commands.json',database.read_bytes())
     for i,source in enumerate(sources):
      obj=source.with_suffix('.o');objects.append(obj)
      if is_cuda:args=['nvcc','-arch=sm_80','-O3','-DMUL=17','-rdc=true','-dc','-o',str(obj),str(source)]
      else:args=['ptxas','-arch=sm_80','-c','-o',str(obj),str(source)]
      tool(args,archive,f'{case["name"]}/original-unit{i}');archive.writestr(case['name']+'/'+obj.name,obj.read_bytes())
     original_path=folder/'original.cubin'
     tool(['nvlink','-arch=sm_80','-o',str(original_path),*[str(p)for p in objects]],archive,case['name']+'/original-link')
     original=original_path.read_bytes();archive.writestr(case['name']+'/original-linked.cubin',original);binaries=[('original-linked',original)]
     for level in('safe','mined','sched','opt'):
      target=folder/level;options=build.Options(opt_level=level,device_libraries=sources[1:],compile_commands=database)
      if case['name']=='cuda_remote':
       origin,files,includes=_sources(sources[0],target,options);library_records=json.loads(files['.oa-device-libraries.json'])
       request=dict(source=base64.b64encode(pack(files)).decode(),path='source/'+sources[0].name,options={**asdict(options),'include_dirs':includes,'compile_commands':'source/compile_commands.json','compile_command':None,'device_libraries':[r['path']for r in library_records],'device_library_options':[{k:v for k,v in r.items()if k!='path'}for r in library_records]},name=None)
       result=process(request);assert not result.get('error')and result['rc']==0,result.get('error',result.get('errors'))
       target.mkdir();unpack(base64.b64decode(result['artifacts']),target);archive.writestr(case['name']+'/'+level+'/source-bundle.zip',pack(files))
      else:
       args=['compile',str(sources[0]),'--out-dir',str(target),'--opt-level',level]
       for library in sources[1:]:args+=['--device-library',str(library)]
       if database:args+=['--compile-commands',str(database)]
       assert cli(args)==0,(case['name'],level)
      manifest=json.loads((target/'manifest.json').read_text());entry=manifest['entries'][0];contract=entry['kernels'][0]
      assert contract['device_link']['strategy']=='scalar-ptx-source-link'and len(contract['device_link']['libraries'])==len(sources)-1
      binary=(target/entry['cubin']).read_bytes();validate(entry,binary);binaries.append(('oa-'+level,binary))
      prefix=case['name']+'/'+level+'/';archive.writestr(prefix+'manifest.json',json.dumps(manifest,indent=2));archive.writestr(prefix+'oa.cubin',binary);archive.writestr(prefix+'merged.ptx',(target/entry['ptx']).read_bytes())
      if level=='safe':
       control=(target/entry['reference']).read_bytes();archive.writestr(case['name']+'/merged-ptxas.cubin',control);binaries.append(('merged-ptxas',control))
     for compiler,binary in binaries:
      path=folder/(compiler+'.cubin');path.write_bytes(binary);archive.writestr(case['name']+'/'+compiler+'.sass',subprocess.check_output(['nvdisasm','-hex','-c',str(path)]))
      module=driver.load(binary)
      try:
       fn=driver.function(module,'probe');resources=driver.func_attrs(fn);frames=local_frame_inventory(binary);inp,out=driver.alloc(lanes*8+512),driver.alloc(lanes*8+512)
       if compiler.startswith('oa-'):assert resources['local_bytes']==(32 if case['name']=='local_frame'else 0)
       try:
        for n in(0,1,259,384):
         for seed in(21941,21942):
          data=values(seed,lanes);before=guard+struct.pack('<384Q',*data)+guard;initial=guard+struct.pack('<Q',SENTINEL)*lanes+guard
          driver.htod(inp,before);driver.htod(out,initial);driver.launch(fn,(3,1,1),(128,1,1),[C.c_uint64(out+256),C.c_uint64(inp+256),C.c_uint32(n)])
          actual=driver.dtoh(out,len(initial));unchanged=driver.dtoh(inp,len(before));reference=expected(data,case,n);wanted=guard+reference+guard
          prefix=f'{case["name"]}/{compiler}/{n}/{seed}/'
          for name,raw in(('before-input',before),('input',unchanged),('output',actual)):archive.writestr(prefix+name+'.bin',raw)
          rows.append(dict(name=case['name'],compiler=compiler,n=n,seed=seed,passed=actual==wanted and unchanged==before,resources=resources,local_frame_bytes=frames['probe'],cubin_sha256=hashlib.sha256(binary).hexdigest(),output_sha256=hashlib.sha256(actual).hexdigest(),input_sha256=hashlib.sha256(unchanged).hexdigest()))
       finally:driver.free(inp);driver.free(out)
      finally:driver.unload(module)
  except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc))
  finally:driver.close()
  return dict(schema=1,passed=error is None and len(rows)==len(fixtures)*48 and all(r['passed']for r in rows),error=error,results=rows,device=device,python=sys.version,nvcc=subprocess.check_output(['nvcc','--version'],text=True),source_hashes={name:hashlib.sha256(data).hexdigest()for name,data in payload.items()},scope='Normal CLI/source-library and remote-worker compilation versus original independent PTX/CUDA object linking and merged-PTX controls; full bytes/effects/resources/ownership, no performance claim'),base64.b64encode(blob.getvalue()).decode()
@app.local_entrypoint()
def main():
 import base64,sys
 sys.path.insert(0,str(ROOT));from openassembler.transport import runtime_files
 payload=runtime_files(native=True)
 for name,file in{'openassembler/ptx/link.py':'link.py','openassembler/ptx/inlining.py':'candidate-inlining.py','openassembler/build.py':'candidate-build.py','openassembler/cppfront.py':'candidate-cppfront.py','openassembler/cli.py':'candidate-cli.py','openassembler/native_ownership.py':'candidate-native_ownership.py','openassembler/remote_build.py':'candidate-remote_build.py','compiler_exp/gap_closure/device_linkage/promotion/fixtures.py':'fixtures.py'}.items():payload[name]=(HERE/file).read_bytes()
 harness=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');report,encoded=execute.remote(payload)
 archive=HERE/('candidate-a100-'+stamp+'.zip');archive.write_bytes(base64.b64decode(encoded));report.update(archive=dict(path=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),harness_sha256=harness)
 path=HERE/('candidate-a100-'+stamp+'.json');path.write_text(json.dumps(report,indent=2)+'\n');print(path,'passed',report['passed'],'launches',len(report['results']))
 if not report['passed']:raise RuntimeError('device library production gate failed')
