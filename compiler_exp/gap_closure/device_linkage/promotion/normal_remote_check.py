"""Exercise ordinary CLI sender, real remote compiler worker and publication."""
import json,subprocess,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
if __name__=='__main__':
 root=HERE/'normal-remote-input';root.mkdir(exist_ok=True);main=root/'entry';lib=root/'library';main.mkdir(exist_ok=True);lib.mkdir(exist_ok=True)
 source=main/'entry.cu';library=lib/'helper.cu';source.write_bytes((HERE.parent/'entry.cu').read_bytes());library.write_bytes((HERE.parent/'helper.cu').read_bytes())
 database=main/'compile_commands.json';database.write_text(json.dumps([dict(directory=str(path.parent),file=str(path),arguments=['nvcc','-c','-O3',str(path)])for path in(source,library)]))
 out=HERE/'normal-remote-build';args=[sys.executable,'-m','openassembler','compile',str(source),'--device-library',str(library),'--compile-commands',str(database),'--out-dir',str(out),'--build-backend','modal','--no-sass']
 result=subprocess.run(args,cwd=ROOT);assert result.returncode==0,result.returncode
 sys.path.insert(0,str(ROOT));from openassembler.native_ownership import validate
 entry=json.loads((out/'manifest.json').read_text())['entries'][0];binary=(out/entry['cubin']).read_bytes();validate(entry,binary)
 contract=entry['kernels'][0]['device_link'];assert contract['libraries'][0]['source']==str(library.resolve())
 assert contract['libraries'][0]['source_sha256']==hashlib.sha256(library.read_bytes()).hexdigest()
 report=dict(schema=1,passed=True,args=args,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),cubin_sha256=hashlib.sha256(binary).hexdigest(),contract=contract,scope='Ordinary CLI with real Modal compiler sender/worker/publication, external CUDA library and database settings; ownership/hash checks, no GPU execution')
 (HERE/'normal-remote-audit.json').write_text(json.dumps(report,indent=2)+'\n');print('Passed normal CLI/Modal publication with external library')
