"""Actual toolkit controls for original PTX, CUDA and typed Python-emitted PTX."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from python_fixtures import cases
from candidate import install
from openassembler import oracle

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
 install()
 from openassembler.pyfront import compile_file
 from openassembler.cppfront import compile_to_ptx_module
 base=HERE/'oracle';base.mkdir(exist_ok=True);rows=[]
 for case in cases():
  name=case['name'];source=base/(name+'.ptx')
  if case['mode']=='python':
   original=source.with_suffix('.py');original.write_text(case['python_source']);text,names=compile_file(original);assert names==[case['entry']]
  elif case['mode']=='cuda':
   original=source.with_suffix('.cu');original.write_text(case['cuda_source']);text,names=compile_to_ptx_module(original);assert names==[case['entry']]
  else:text=case['source'];original=source
  source.write_text(text);binary=source.with_suffix('.cubin')
  result=oracle.run_tool('ptxas','-arch=sm_80','-o',oracle.tool_path(binary),oracle.tool_path(source),check=False);source.with_suffix('.tool.log').write_text(result.stdout+result.stderr)
  row=dict(name=name,accepted=result.returncode==0,source_sha256=sha(original.read_bytes()),ptx_sha256=sha(source.read_bytes()))
  if result.returncode==0:
   listing=oracle.disasm_text(binary);source.with_suffix('.sass').write_text(listing);row['cubin_sha256']=sha(binary.read_bytes());row['multiply_instructions']=[i.text for i in oracle.parse_listing(listing)if i.text.startswith('IMAD')]
  rows.append(row)
 result=dict(schema=1,passed=all(r['accepted']for r in rows)and len(rows)==35,results=rows,script_sha256=sha(Path(__file__).read_bytes()),source_hashes={name:sha((HERE/name).read_bytes())for name in('candidate-pyfront.py','python_fixtures.py','fixtures.py','cuda.cu')},scope='Original typed PTX and actual CUDA/Python frontends, independent ptxas acceptance/cubin/listings, no execution')
 (HERE/'oracle-probe.json').write_text(json.dumps(result,indent=2)+'\n');print('Toolkit accepted',sum(r['accepted']for r in rows),'of',len(rows))
 if not result['passed']:raise SystemExit(1)
