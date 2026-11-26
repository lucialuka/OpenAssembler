"""Frozen original typed/effectful condition-chain rejections without source execution."""
import builtins,hashlib,importlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from fixtures import configurations,write_tree,NAMES
if __name__=='__main__':
 for name,suffix in(('modules','modules'),('compile','pyfront')):
  module=importlib.import_module('openassembler.pyfront.'+name);frozen=HERE/('baseline-'+suffix+'.py');exec(builtins.compile(frozen.read_bytes(),str(frozen),'exec'),module.__dict__)
 from openassembler.pyfront.compile import compile_kernel,find_kernel
 rows=[]
 with tempfile.TemporaryDirectory(prefix='oa-chain-baseline-')as temporary:
  for case in configurations():
   source=write_tree(Path(temporary)/case['name'],case)
   for name in NAMES:
    try:compile_kernel(find_kernel(source,name))
    except Exception as error:
     assert type(error).__name__=='NotSupported'and'chained comparisons'in str(error),(case['name'],name,str(error));rows.append(dict(configuration=case['name'],entry=name,rejected=True,error=str(error)))
    else:raise AssertionError(name)
 report=dict(schema=1,passed=True,results=rows,fixture_sha256=hashlib.sha256((HERE/'fixtures.py').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),frozen_frontend_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),scope='60 original same-file/import/package condition-chain rejections from exact pre-promotion frontend; user modules never execute')
 (HERE/'baseline.json').write_text(json.dumps(report,indent=2)+'\n');print('Recorded 60 frozen original chain rejections')
