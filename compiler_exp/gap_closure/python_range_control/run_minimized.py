"""Actual ptxas O0/O3 A100 observations for isolated counter divergence."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import modal
ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
app=modal.App('oa-range-counter-minimized');image=modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04',add_python='3.12')
@app.function(image=image,gpu='A100-40GB',timeout=1200,retries=0,max_containers=1)
def execute(payload,source,controls):
 import base64,ctypes as C,io,os,struct,subprocess,sys,tempfile,zipfile
 from dataclasses import asdict
 with tempfile.TemporaryDirectory(prefix='oa-minimized-range-')as temporary:
  root=Path(temporary)
  for name,data in payload.items():
   path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
  os.chdir(root);sys.path.insert(0,str(root));(root/'probe.ptx').write_text(source)
  from openassembler.gpu.cuda_driver import Driver
  from openassembler.container.cubin import regcount
  driver=Driver();driver.open();device=asdict(driver.info());assert device['cc']=='8.0'and'A100'in device['name'];rows=[];blob=io.BytesIO();error=None;guard=b'\xa5'*64
  try:
   with zipfile.ZipFile(blob,'w',zipfile.ZIP_DEFLATED)as archive:
    archive.writestr('probe.ptx',source)
    for level in(0,3):
     binary=root/f'o{level}.cubin';result=subprocess.run(['ptxas','-arch=sm_80','-O'+str(level),'-o',str(binary),str(root/'probe.ptx')],capture_output=True,text=True);archive.writestr(f'o{level}/tool.log',result.stdout+result.stderr);assert result.returncode==0,result.stderr
     data=binary.read_bytes();archive.writestr(f'o{level}/reference.cubin',data);archive.writestr(f'o{level}/reference.sass',subprocess.check_output(['nvdisasm','-hex','-c',str(binary)]));module=driver.load(data)
     try:
      for row in controls:
       name=row['entry'];fn=driver.function(module,name);address=driver.alloc(152)
       try:
        driver.htod(address,guard+bytes(24)+guard);args=[C.c_uint64(address+64)]
        if row['mode']=='runtime_signed':args.extend([C.c_int32(row['start']),C.c_int32(row['stop'])])
        driver.launch(fn,(1,1,1),(1,1,1),args);actual=driver.dtoh(address,152);assert actual[:64]==actual[-64:]==guard
        values=list(struct.unpack('<3q',actual[64:-64]));assert row['mode']!='isolated_add'or values==row['wrapped_reference']
        archive.writestr(f'o{level}/{name}/output.bin',actual);rows.append(dict(entry=name,mode=row['mode'],optimization=level,observed_signed=values,matches_python=values==row['python_reference'],matches_wrapped=values==row['wrapped_reference'],output_sha256=hashlib.sha256(actual).hexdigest(),registers=regcount(data,name),passed=True))
       finally:driver.free(address)
     finally:driver.unload(module)
  except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc))
  finally:driver.close()
  return dict(schema=1,passed=error is None and len(rows)==40,error=error,results=rows,device=device,ptxas=subprocess.check_output(['ptxas','--version'],text=True),python=sys.version,source_sha256=hashlib.sha256(source.encode()).hexdigest(),scope='40 finite A100 ptxas O0/O3 instruction/loop observations. Only isolated additions enforce wrapped output; loop results classify mathematical Python versus explicit i32-wrap references without guessing the external optimizer contract. No production promotion or timing claim'),base64.b64encode(blob.getvalue()).decode()
@app.local_entrypoint()
def main():
 import base64,sys
 sys.path.insert(0,str(ROOT));from openassembler.transport import runtime_files
 controls=json.loads((HERE/'minimized.json').read_text());source=(HERE/'minimized.ptx').read_text();assert controls['source_sha256']==hashlib.sha256(source.encode()).hexdigest()
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');report,encoded=execute.remote(runtime_files(native=True),source,controls['results']);archive=HERE/('minimized-a100-'+stamp+'.zip');archive.write_bytes(base64.b64decode(encoded));report.update(archive=dict(path=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),controls_sha256=hashlib.sha256((HERE/'minimized.json').read_bytes()).hexdigest());path=HERE/('minimized-a100-'+stamp+'.json');path.write_text(json.dumps(report,indent=2)+'\n');print(path,'passed',report['passed'],'observations',len(report['results']))
 if not report['passed']:raise RuntimeError('minimized counter gate failed')
