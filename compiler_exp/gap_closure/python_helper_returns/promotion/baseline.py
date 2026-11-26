"""Repeat original return-path source/entry rejections from frozen pre-promotion frontend."""
import builtins,hashlib,importlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from fixtures import configurations,write_tree,NAMES
if __name__=='__main__':
 for name,suffix in(('modules','modules'),('helpers','helpers'),('compile','pyfront')):
  module=importlib.import_module('openassembler.pyfront.'+name);path=HERE/('baseline-'+suffix+'.py');exec(builtins.compile(path.read_bytes(),str(path),'exec'),module.__dict__)
 from openassembler.pyfront.compile import compile_kernel,find_kernel
 rows=[]
 with tempfile.TemporaryDirectory(prefix='oa-tuple-baseline-')as temp:
  for case in configurations():
   path=write_tree(Path(temp)/case['name'],case)
   for name in NAMES:
    try:compile_kernel(find_kernel(path,name))
    except Exception as error:
     assert type(error).__name__=='NotSupported',(case['name'],name,type(error).__name__,str(error))
     rows.append(dict(configuration=case['name'],entry=name,rejected=True,error=type(error).__name__+': '+str(error)))
    else:raise AssertionError(name)
 result=dict(schema=1,passed=True,results=rows,fixture_sha256=hashlib.sha256((HERE/'fixtures.py').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='60 original return-path source/entry rejections replayed with exact frozen production frontend before early-return admission; source modules never execute')
 (HERE/'baseline.json').write_text(json.dumps(result,indent=2)+'\n');print('Recorded 60 frozen original return-path-loop rejections')
