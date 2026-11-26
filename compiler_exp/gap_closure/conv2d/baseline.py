"""Record original Conv2d rejection without executing the model module."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import builtins,importlib
module=importlib.import_module("openassembler.modelfront.graph")
frozen=HERE/"baseline/graph.py"
exec(builtins.compile(frozen.read_bytes(),str(frozen),"exec"),module.__dict__)
import torch
from openassembler.modelfront.graph import capture_model,ModelGraphError
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
class Probe(torch.nn.Module):
 def __init__(self):super().__init__();self.conv=torch.nn.Conv2d(2,3,3,padding=1)
 def forward(self,x):return self.conv(x)
if __name__=='__main__':
 try:capture_model(Probe(),torch.zeros(1,2,5,5))
 except ModelGraphError as error:
  assert 'unsupported native model node' in str(error)and'conv' in str(error);diagnostic=str(error)
 else:raise AssertionError('Conv2d unexpectedly admitted before the candidate')
 paths=[HERE/'baseline/graph.py',HERE/'baseline/kernels.py',HERE/'baseline/reference.py',Path(__file__)]
 result=dict(schema=1,passed=True,production_fix=False,diagnostic=diagnostic,source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths},scope='Original FX Conv2d module rejection with exact source-side contract; no numerical execution or GPU claim')
 (HERE/'baseline.json').write_text(json.dumps(result,indent=2)+'\n');print('Recorded original Conv2d graph rejection')
