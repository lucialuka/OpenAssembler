"""Generated integer trees with separately computed arbitrary-precision references."""
import ast,hashlib,json,random,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from openassembler.pyfront import compile_file
from openassembler.pyfront.modules import contract_from_ptx
rng=random.Random(42153)
def tree(depth):
 if depth==0:
  value=rng.randint(-(1<<62),(1<<62)-1);return str(value),value
 left,a=tree(depth-1);right,b=tree(depth-1);choice=rng.randrange(9)
 if choice==0:return f'({left}+{right})',a+b
 if choice==1:return f'({left}-{right})',a-b
 if choice==2:return f'({left}*{right})',a*b
 if choice==3:return f'({left}&{right})',a&b
 if choice==4:return f'({left}|{right})',a|b
 if choice==5:return f'({left}^{right})',a^b
 if choice==6:
  divisor=abs(b)+1;return f'({left}//{divisor})',a//divisor
 if choice==7:
  divisor=-(abs(b)+1);return f'({left}%({divisor}))',a%divisor
 shift=rng.randrange(64);return f'({left}>>{shift})',a>>shift
if __name__=='__main__':
 rows=[]
 with tempfile.TemporaryDirectory(prefix='oa-Final-trees-')as temp:
  path=Path(temp)/'source.py'
  for i in range(2048):
   expression,wanted=tree(rng.randrange(1,5))
   path.write_text('from typing import Final\nfrom openassembler.pyfront import *\nVALUE:Final[int]='+expression+'\n@kernel\ndef k(out:Array(u64)):\n    out[0]=u64(VALUE)\n')
   text,_=compile_file(path);graph=contract_from_ptx(text);actual=int(graph['static_values']['source.py:VALUE']['value']['value']);assert actual==wanted,(i,expression,wanted,actual)
   rows.append(dict(expression=expression,expected=str(wanted),passed=True))
 result=dict(schema=1,passed=True,seed=42153,results=rows,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),static_values_sha256=hashlib.sha256((HERE/'static_values.py').read_bytes()).hexdigest(),scope='2048 deterministically generated typed integer Final expression trees with independently constructed arbitrary-precision results, retained full expressions/results and modulo64 casts; not a proof of all expressions or hardware')
 (HERE/'property-probe.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',len(rows),'independent integer expression trees')
