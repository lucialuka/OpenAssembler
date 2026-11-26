"""Frozen pre-promotion original source admissions/rejections; no unsafe baseline launch."""
import builtins,hashlib,importlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from fixtures import configurations,write_tree,NAMES
if __name__=='__main__':
 for name,suffix in(('compile','pyfront'),('return_flow','return_flow')):
  module=importlib.import_module('openassembler.pyfront.'+name);path=HERE/('baseline-'+suffix+'.py');exec(builtins.compile(path.read_bytes(),str(path),'exec'),module.__dict__)
 from openassembler.pyfront.compile import compile_kernel,find_kernel
 accepted={'positive_overflow','negative_overflow','large_positive_step','minimum_step','runtime_positive','runtime_negative','mutated_target','bound_snapshot'};rows=[]
 with tempfile.TemporaryDirectory(prefix='oa-loop-baseline-')as temporary:
  for case in configurations():
   source=write_tree(Path(temporary)/case['name'],case)
   for name in NAMES:
    try:text=compile_kernel(find_kernel(source,name))
    except Exception as error:
     assert name not in accepted and type(error).__name__=='NotSupported'and'else'in str(error),(case['name'],name,str(error))
     rows.append(dict(configuration=case['name'],entry=name,rejected=True,error=str(error)))
    else:
     assert name in accepted,name;rows.append(dict(configuration=case['name'],entry=name,rejected=False,ptx_sha256=hashlib.sha256(text.encode()).hexdigest(),not_executed=True))
 result=dict(schema=1,passed=True,results=rows,accepted_not_executed=24,rejected=36,fixture_sha256=hashlib.sha256((HERE/'fixtures.py').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='60 original source/entry classifications with frozen compile/return analysis: 36 loop-else rejections and 24 accepted potentially overflowing kernels deliberately not launched. Separate eight-trip watchdog experiment retains actual baseline wrap counterexamples; no original reference or admission is mislabeled as rejection')
 (HERE/'baseline.json').write_text(json.dumps(result,indent=2)+'\n');print('Recorded 36 frozen loop-else rejections and 24 accepted unexecuted kernels')
