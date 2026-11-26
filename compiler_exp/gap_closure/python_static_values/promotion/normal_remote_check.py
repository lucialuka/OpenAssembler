"""Actual public Modal CLI publication of original static declarations."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from fixtures import configurations,write_tree,NAMES
if __name__=='__main__':
 root=HERE/'normal-remote-input';case=next(c for c in configurations()if c['name']=='remote_package');source=write_tree(root,case);out=HERE/'normal-remote-build'
 args=[sys.executable,'-m','openassembler','compile',str(source),'--out-dir',str(out),'--build-backend','modal','--no-sass']
 result=subprocess.run(args,cwd=ROOT);assert result.returncode==0,result.returncode
 from openassembler.native_ownership import validate
 from openassembler.pyfront.modules import validate_sources
 manifest=json.loads((out/'manifest.json').read_text());entry,=manifest['entries'];binary=(out/entry['cubin']).read_bytes();validate(entry,binary)
 assert [k['name']for k in entry['kernels']]==list(NAMES)
 graph=entry['kernels'][0]['python_source_graph'];validate_sources(source,graph);assert graph['static_values']
 proof=dict(schema=1,passed=True,args=args,cubin_sha256=hashlib.sha256(binary).hexdigest(),manifest_sha256=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),contract=graph,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Original Final source declarations through actual public CLI/Modal worker/publication, exact original dependency/static-value graph and native ownership; no GPU execution')
 (HERE/'normal-remote-audit.json').write_text(json.dumps(proof,indent=2)+'\n');print('Passed public CLI/Modal publication with original static values')
