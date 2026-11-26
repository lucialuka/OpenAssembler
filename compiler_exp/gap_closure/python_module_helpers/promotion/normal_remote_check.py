"""Public CLI to actual Modal compiler and local publication for an imported tree."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from fixtures import configurations,write_tree
if __name__=='__main__':
 root=HERE/'normal-remote-input';case=next(c for c in configurations()if c['name']=='explicit_root');source=write_tree(root,case);out=HERE/'normal-remote-build'
 args=[sys.executable,'-m','openassembler','compile',str(source.parent),'--python-source-root',str(root),'--out-dir',str(out),'--build-backend','modal','--no-sass']
 result=subprocess.run(args,cwd=ROOT);assert result.returncode==0,result.returncode
 from openassembler.native_ownership import validate
 from openassembler.pyfront.modules import validate_sources
 manifest=json.loads((out/'manifest.json').read_text());entry,=manifest['entries'];binary=(out/entry['cubin']).read_bytes();validate(entry,binary)
 from openassembler.project import entry_source_abs
 assert entry_source_abs(manifest,entry).resolve()==source.resolve()
 graph=entry['kernels'][0]['python_source_graph'];validate_sources(source,graph,root)
 report=dict(schema=1,passed=True,args=args,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cubin_sha256=hashlib.sha256(binary).hexdigest(),manifest_sha256=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),contract=graph,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Ordinary production CLI sender, actual Modal source graph compiler worker and local artifact publication with explicit root; complete graph/native ownership/source hashes, no GPU execution')
 (HERE/'normal-remote-audit.json').write_text(json.dumps(report,indent=2)+'\n');print('Passed public CLI/Modal publication with explicit Python source root')
