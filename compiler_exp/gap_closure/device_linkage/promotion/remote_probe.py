"""Exercise actual native request packaging/worker for an external CUDA library."""
import base64,json,sys,tempfile
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from openassembler import build,remote_build
from openassembler.torchfront.archive import pack,unpack

if __name__=='__main__':
 with tempfile.TemporaryDirectory(prefix='oa-link-remote-',dir=HERE)as temporary:
  root=Path(temporary);main=root/'main';library=root/'external';main.mkdir();library.mkdir()
  src=main/'entry.cu';src.write_bytes((HERE.parent/'entry.cu').read_bytes());lib=library/'helper.cu';lib.write_bytes((HERE.parent/'helper.cu').read_bytes())
  db=main/'compile_commands.json';db.write_text(json.dumps([dict(directory=str(path.parent),file=str(path),arguments=['nvcc','-c','-DREMOTE_LIBRARY_TEST=1',str(path)])for path in(src,lib)]))
  options=build.Options(no_sass=True,device_libraries=[lib],compile_commands=db);out=root/'output'
  origin,payload,includes=remote_build._sources(src,out,options);records=json.loads(payload['.oa-device-libraries.json'])
  request={'source':base64.b64encode(pack(payload)).decode(),'path':'source/entry.cu','options':{**asdict(options),'device_libraries':[r['path']for r in records],'device_library_options':[{k:v for k,v in r.items()if k!='path'}for r in records],'include_dirs':includes,'compile_commands':'source/compile_commands.json','compile_command':None},'name':None}
  # Linux worker code uses system temporary storage; route this local replay
  # into the Colima-mounted workspace without altering production worker code.
  previous=tempfile.tempdir;tempfile.tempdir=str(root)
  try:result=remote_build.process(request)
  finally:tempfile.tempdir=previous
  assert not result.get('error'),result
  assert result['rc']==0,result.get('errors')
  target=root/'returned';target.mkdir();unpack(base64.b64decode(result['artifacts']),target)
  entry=json.loads((target/'manifest.json').read_text())['entries'][0]
  from openassembler.native_ownership import validate
  validate(entry,(target/entry['cubin']).read_bytes())
  contract=entry['kernels'][0]['device_link'];assert contract['libraries'][0]['source']==str(lib)
  assert contract['libraries'][0]['dependencies']
  (HERE/'remote-probe.json').write_text(json.dumps(dict(schema=1,passed=True,contract=contract,scope='Actual serialized source bundle and native process worker with an external CUDA library; build/ownership only, no GPU execution'),indent=2)+'\n')
  print('Passed remote request/worker/external-library ownership')
