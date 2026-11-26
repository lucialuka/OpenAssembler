"""Actual serialized compiler worker and ignored reachable helper source custody."""
import base64,hashlib,json,sys,tempfile
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from fixtures import configurations,write_tree
from openassembler import build,oracle
from openassembler.remote_build import _sources,process
from openassembler.torchfront.archive import pack,unpack
from openassembler.native_ownership import validate
if __name__=='__main__':
 folder=HERE/'remote-input';case=next(c for c in configurations()if c['name']=='remote_package');source=write_tree(folder,case)
 (folder/'.gitignore').write_text('arithmetic/base.py\n')
 options=build.Options(no_sass=True);root,files,includes=_sources(source,HERE/'remote-output',options)
 assert root==folder and files['source/arithmetic/base.py']==(folder/'arithmetic/base.py').read_bytes()
 request=dict(source=base64.b64encode(pack(files)).decode(),path='source/'+source.relative_to(root).as_posix(),options={**asdict(options),'include_dirs':includes,'python_source_root':None},name=None)
 # Colima's Linux toolkit can only see the mounted workspace, not macOS /tmp.
 oracle.SCRATCH.mkdir(parents=True,exist_ok=True);tempfile.tempdir=str(oracle.SCRATCH)
 result=process(request);assert not result.get('error')and result['rc']==0,result
 output=HERE/'remote-output';output.mkdir(exist_ok=True);unpack(base64.b64decode(result['artifacts']),output)
 manifest=json.loads((output/'manifest.json').read_text());entry,=manifest['entries'];assert entry['source']=='application/entry.py'
 validate(entry,(output/entry['cubin']).read_bytes());graphs=[k['python_source_graph']for k in entry['kernels']];assert all(g==graphs[0]for g in graphs)
 assert len(graphs[0]['files'])==5
 proof=dict(schema=1,passed=True,kernels=[k['name']for k in entry['kernels']],graph=graphs[0],manifest_sha256=hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),payload_sha256=hashlib.sha256(pack(files)).hexdigest(),scope='Actual _sources serialized request/worker, package-root entry rebasing, ignored reachable module included, exact returned graph/native ownership; no GPU execution')
 (HERE/'remote-probe.json').write_text(json.dumps(proof,indent=2)+'\n');print('Passed serialized worker with ignored dependency and five source files')
