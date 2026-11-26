"""Retain intended tuple-loop control semantics before selecting a production design."""
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from openassembler.pyfront import compile_file,NotSupported
CASES=[
 ('plain',(3,-5,7),'    for c in COEFFICIENTS:\n        value=value*x+i64(c)\n',None),
 ('empty',(),'    for c in COEFFICIENTS:\n        value+=i64(c)\n',None),
 ('break',(3,-5,7),'    for c in COEFFICIENTS:\n        if c<0:\n            break\n        value=value*x+i64(c)\n',None),
 ('continue',(3,-5,7),'    for c in COEFFICIENTS:\n        if c<0:\n            continue\n        value=value*x+i64(c)\n',None),
 ('nested',(3,-5,7),'    for c in COEFFICIENTS:\n        for d in (1,2):\n            value+=i64(c*d)\n',None),
 ('target_after',(3,-5,7),'    c=11\n    for c in COEFFICIENTS:\n        value+=i64(c)\n    value+=i64(c)\n',None),
 ('loop_else',(3,-5,7),'    for c in COEFFICIENTS:\n        value+=i64(c)\n    else:\n        value+=i64(17)\n',None),
 ('tuple_values',(3,5,7),'    for c in COEFFICIENTS:\n        value*=i64(c)\n',None),
]
def reference(name,x):
 if name=='plain':return (3*x-5)*x+7
 if name=='empty':return 0
 if name=='break':return 3
 if name=='continue':return 3*x+7
 if name=='nested':return 15
 if name=='target_after':return 12
 if name=='loop_else':return 22
 return 0
if __name__=='__main__':
 rows=[]
 with tempfile.TemporaryDirectory(prefix='oa-static-loop-controls-')as temporary:
  for name,values,body,_ in CASES:
   source='from typing import Final\nfrom openassembler.pyfront import *\nCOEFFICIENTS:Final='+repr(values)+'\n@device\ndef helper(x:i64)->i64:\n    value=i64(0)\n'+body+'    return value\n@kernel\ndef k(a:Array(i64),out:Array(i64)):\n    out[0]=helper(a[0])\n'
   path=Path(temporary)/(name+'.py');path.write_text(source)
   try:compile_file(path)
   except NotSupported as error:diagnostic=str(error)
   else:raise AssertionError('baseline unexpectedly admitted '+name)
   rows.append(dict(name=name,source=source,rejected=True,diagnostic=diagnostic,reference_domain=[-7,0,1,3,1<<31],expected_modulo64=[reference(name,x)&((1<<64)-1)for x in[-7,0,1,3,1<<31]]))
 result=dict(schema=1,passed=True,results=rows,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Eight original-source static tuple iteration control contracts and deterministic independent expected values, all existing production rejections retained. No implementation or execution claims for break/continue/nested/empty/target/else semantics')
 (HERE/'control-matrix.json').write_text(json.dumps(result,indent=2)+'\n');print('Retained eight tuple-loop control boundaries and exact independent references')
