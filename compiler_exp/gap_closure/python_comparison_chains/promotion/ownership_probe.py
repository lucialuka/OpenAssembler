import sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from openassembler.pyfront import compile_file
from openassembler.pyfront.modules import contract_from_ptx
with tempfile.TemporaryDirectory(prefix='oa-chain-owner-')as temporary:
 path=Path(temporary)/'source.py';path.write_text('from openassembler.pyfront import *\n@kernel\ndef k(a:Array(i32)):\n    if 0<a[0]<7:\n        a[0]=17\n');text,_=compile_file(path);graph=contract_from_ptx(text);assert graph and graph.get('comparison_chains'),'standalone chain source/expression ownership missing'
print('Passed standalone original chain ownership')
