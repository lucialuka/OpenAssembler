"""Verify the actual parser admission boundary before inferring numeric behavior."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from openassembler.ptx.parse import parse_module
if __name__=='__main__':
 source=HERE/'narrow.ptx'
 try:parse_module(source.read_text())
 except SyntaxError as error:
  assert 'scalar 32/64-bit parameters required' in str(error)
  report=dict(schema=1,passed=True,production_fix=False,rejected=True,stage='PTX parameter parser',diagnostic=str(error),counterexamples=0,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parser_sha256=hashlib.sha256((ROOT/'openassembler/ptx/parse.py').read_bytes()).hexdigest(),scope='The actual .param.u8 entry is explicitly rejected before interpretation or native lowering; the neighboring-byte extraction hypothesis is not an admitted miscompile. Boolean ABI first requires a complete narrow-parameter parse/packing/loading contract. No numeric execution, GPU or production claim')
 else:raise AssertionError('narrow parameter parser unexpectedly admitted')
 (HERE/'narrow-baseline.json').write_text(json.dumps(report,indent=2)+'\n');print('Confirmed explicit narrow-parameter parser rejection; no admitted numerical counterexample')
