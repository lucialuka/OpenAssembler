"""Actual ptxas originals from five Python source graphs and full raw artifact custody."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from fixtures import configurations,write_tree,NAMES
from openassembler.pyfront import compile_file
from openassembler.pyfront.modules import contract_from_ptx
from openassembler import oracle

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
 rows=[];files={}
 for case in configurations():
  folder=HERE/'oracle'/case['name'];source=write_tree(folder/'original',case);options={'source_root':folder/'original'}if case['explicit']else{}
  text,names=compile_file(source,**options);assert names==list(NAMES);ptx=folder/'source.ptx';ptx.write_text(text);binary=folder/'reference.cubin'
  result=oracle.run_tool('ptxas','-arch=sm_80','-o',oracle.tool_path(binary),oracle.tool_path(ptx),check=False);(folder/'tool.log').write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr
  listing=oracle.disasm_text(binary);(folder/'reference.sass').write_text(listing)
  rows.append(dict(configuration=case['name'],accepted=True,contract=contract_from_ptx(text),ptx_sha256=sha(ptx.read_bytes()),cubin_sha256=sha(binary.read_bytes())))
  for path in folder.rglob('*'):
   if path.is_file():files[str(path.relative_to(HERE))]=sha(path.read_bytes())
 report=dict(schema=1,passed=True,results=rows,artifact_hashes=files,script_sha256=sha(Path(__file__).read_bytes()),source_hashes={name:sha((HERE/name).read_bytes())for name in('static_values.py','candidate-modules.py','candidate-pyfront.py','candidate-helpers.py','fixtures.py')},scope='Four original Final Python source graphs with 36 kernels, independent actual ptxas acceptance/cubins/listings, exact source/graph/raw artifact identities; no GPU execution')
 (HERE/'oracle-probe.json').write_text(json.dumps(report,indent=2)+'\n');print('Passed 4 source graphs;',len(files),'raw artifact hashes')
