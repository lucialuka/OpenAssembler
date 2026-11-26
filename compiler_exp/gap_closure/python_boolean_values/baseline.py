"""Original operand/Boolean value rejection from the frozen pre-promotion frontend."""
import ast,builtins,hashlib,importlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
if __name__=='__main__':
 frozen=HERE/'baseline-pyfront.py';module=importlib.import_module('openassembler.pyfront.compile');exec(builtins.compile(frozen.read_bytes(),str(frozen),'exec'),module.__dict__)
 from openassembler.pyfront.compile import compile_file,NotSupported
 tree=ast.parse((HERE/'probe.py').read_text());source=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='SOURCE'for t in n.targets)))
 with tempfile.TemporaryDirectory(prefix='oa-boolean-baseline-')as temporary:
  path=Path(temporary)/'source.py';path.write_text(source)
  try:compile_file(path)
  except NotSupported as error:
   assert 'only supported directly'in str(error);result=dict(schema=1,passed=True,rejected=True,source=source,diagnostic=str(error),frozen_frontend_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Original operand-returning and/or plus Boolean local program rejection from frozen production frontend; no source module execution')
  else:raise AssertionError('Boolean value baseline unexpectedly accepted')
 (HERE/'baseline.json').write_text(json.dumps(result,indent=2)+'\n');print('Recorded original operand/Boolean-value rejection')
