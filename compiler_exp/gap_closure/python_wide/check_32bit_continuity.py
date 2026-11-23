"""Compare unchanged 32-bit kernel PTX with exact pre-wide frontend snapshots."""
from pathlib import Path
import hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
FILES=['examples/vector_add.py','tests/test_pyfront.py','tests/test_pyfront_semantics.py']
CODE='''import hashlib,importlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);snapshot=sys.argv[2];sys.path.insert(0,str(root))
if snapshot:
 for name in ('types','compile','helpers','__init__'):
  module=importlib.import_module('openassembler.pyfront'+('' if name=='__init__' else '.'+name))
  p=Path(snapshot)/(name+'.py');exec(compile(p.read_bytes(),str(p),'exec'),module.__dict__)
from openassembler.pyfront import compile_file
result={}
for name in json.loads(sys.argv[3]):
 ptx,kernels=compile_file(root/name);result[name]={'ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest(),'kernels':kernels}
print(json.dumps(result))
'''
if __name__=='__main__':
    def check(snapshot):return json.loads(subprocess.check_output([sys.executable,'-c',CODE,str(ROOT),snapshot,json.dumps(FILES)],text=True))
    before=check(str(HERE/'before'));current=check('');assert before==current
    report={'scope':'Exact PTX identity for unchanged preexisting32-bit source-definedkernels','files':current,'kernels':sum(len(r['kernels']) for r in current.values()),'passed':True}
    (HERE/'32bit-continuity.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
