"""Frozen production original chain rejection without installing the candidate."""
import ast,hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from openassembler.pyfront import compile_file,NotSupported
if __name__=='__main__':
 tree=ast.parse((HERE/'probe.py').read_text());source=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='SOURCE'for t in n.targets)))
 with tempfile.TemporaryDirectory(prefix='oa-chain-baseline-')as temporary:
  path=Path(temporary)/'source.py';path.write_text(source)
  try:compile_file(path)
  except NotSupported as error:
   assert 'chained comparisons'in str(error);result=dict(schema=1,passed=True,rejected=True,source=source,diagnostic=str(error),frontend_sha256=hashlib.sha256((ROOT/'openassembler/pyfront/compile.py').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Original condition/atomic chain rejection before candidate promotion; source module not executed')
  else:raise AssertionError('production comparison chain unexpectedly accepted')
 (HERE/'baseline.json').write_text(json.dumps(result,indent=2)+'\n');print('Recorded original comparison-chain rejection')
