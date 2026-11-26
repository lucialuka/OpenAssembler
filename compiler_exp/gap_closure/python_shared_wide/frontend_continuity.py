"""Isolated old/current frontend PTX and rejection comparison for existing sources."""
import ast
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
CODE='''import hashlib,importlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);snapshot=sys.argv[2];sys.path.insert(0,str(root))
if snapshot:
 module=importlib.import_module('openassembler.pyfront.compile');p=Path(snapshot)
 exec(compile(p.read_bytes(),str(p),'exec'),module.__dict__)
from openassembler.pyfront import compile_file
result={}
for name in json.loads(sys.argv[3]):
 try:
  ptx,names=compile_file(root/name)
  result[name]={'status':'accepted','ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest(),'kernels':names}
 except Exception as error:result[name]={'status':'rejected','exception':type(error).__name__,'message':str(error)}
print(json.dumps(result))
'''
def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
    files=[]
    for base in('tests','examples'):
        for path in sorted((ROOT/base).rglob('*.py')):
            try:tree=ast.parse(path.read_text())
            except SyntaxError:continue
            if any(isinstance(n,ast.FunctionDef)and any('kernel'in ast.unparse(d)for d in n.decorator_list)for n in tree.body):files.append(str(path.relative_to(ROOT)))
    def check(snapshot):return json.loads(subprocess.check_output([sys.executable,'-c',CODE,str(ROOT),snapshot,json.dumps(files)],text=True))
    before=check(str(HERE/'baseline-compile.py'));after=check('');assert before==after
    result=dict(schema=1,passed=True,results=after,source_hashes={name:sha((ROOT/name).read_bytes())for name in files},
                accepted_files=sum(r['status']=='accepted'for r in after.values()),kernels=sum(len(r.get('kernels',[]))for r in after.values()),
                baseline_frontend_sha256=sha((HERE/'baseline-compile.py').read_bytes()),current_frontend_sha256=sha((ROOT/'openassembler/pyfront/compile.py').read_bytes()),
                auditor_sha256=sha(Path(__file__).read_bytes()),scope='Isolated current source files under tests/examples with top-level kernel decorators; exact old/current PTX and rejection equality, not hardware execution or universal Python coverage')
    (HERE/'frontend-continuity.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',len(files),'files;',result['kernels'],'kernels')
