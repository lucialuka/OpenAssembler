"""Replay the original frozen chain rejection without overwriting historical evidence."""
import builtins,hashlib,importlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
if __name__=='__main__':
 original=json.loads((HERE/'baseline.json').read_text());assert original['passed']and original['rejected'];frozen=HERE/'baseline-pyfront.py';assert hashlib.sha256(frozen.read_bytes()).hexdigest()==original['frontend_sha256']
 module=importlib.import_module('openassembler.pyfront.compile');exec(builtins.compile(frozen.read_bytes(),str(frozen),'exec'),module.__dict__)
 from openassembler.pyfront.compile import compile_file,NotSupported
 with tempfile.TemporaryDirectory(prefix='oa-frozen-chain-rejection-')as temporary:
  source=Path(temporary)/'source.py';source.write_text(original['source'])
  try:compile_file(source)
  except NotSupported as error:
   assert'chained comparisons'in str(error);result=dict(schema=1,passed=True,rejected=True,baseline_sha256=hashlib.sha256((HERE/'baseline.json').read_bytes()).hexdigest(),frozen_frontend_sha256=original['frontend_sha256'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),diagnostic=str(error),scope='Original frozen condition-chain rejection replay after production promotion; retained baseline report is unchanged and current production is not falsely asserted to reject')
  else:raise AssertionError('frozen rejection changed')
 (HERE/'baseline-replay.json').write_text(json.dumps(result,indent=2)+'\n');print('Replayed original frozen chain rejection without overwriting history')
