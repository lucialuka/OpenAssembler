"""Explicit current return-control boundaries, avoiding fabricated missing-return values."""
import hashlib,json
from pathlib import Path
from normalize import normalize
HERE=Path(__file__).resolve().parent
CASES=[
 ('missing_return','@device\ndef h(x:i64)->i64:\n    if x<i64(0):\n        return -x\n'),
 ('branch_complete','@device\ndef h(x:i64)->i64:\n    if x<i64(0):\n        return -x\n    else:\n        return x+i64(3)\n'),
 ('loop_else','@device\ndef h(x:i64)->i64:\n    for c in (1,2):\n        if x<i64(c):\n            return x\n    else:\n        return x+i64(3)\n    return x\n'),
 ('continue','@device\ndef h(x:i64)->i64:\n    for c in (1,2):\n        if x<i64(c):\n            continue\n        return x\n    return x+i64(3)\n'),
 ('bare_return','@device\ndef h(x:i64)->i64:\n    if x<i64(0):\n        return\n    return x\n'),
]
if __name__=='__main__':
 rows=[]
 for name,source in CASES:
  try:normalize('from openassembler.pyfront import *\n'+source)
  except ValueError as error:rows.append(dict(name=name,source=source,rejected=True,reason=str(error)))
  else:raise AssertionError('prototype cannot claim '+name)
 result=dict(schema=1,passed=True,results=rows,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),normalizer_sha256=hashlib.sha256((HERE/'normalize.py').read_bytes()).hexdigest(),scope='Five explicit boundaries of isolated terminal-fallback early-return normalization. Missing scalar returns reject; complete branch-only returns, loop else, original continue and bare returns remain unimplemented, not silently lowered or defaulted')
 (HERE/'control-boundaries.json').write_text(json.dumps(result,indent=2)+'\n');print('Retained five return-control boundaries without fabricating default values')
