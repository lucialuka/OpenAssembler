"""Original contextual literal bridge across adjacent fixed-width relations."""
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from openassembler.pyfront import compile_file
if __name__=='__main__':
 rows=[]
 with tempfile.TemporaryDirectory(prefix='oa-chain-context-')as temporary:
  for dtype in('i32','u32','i64','u64','f32'):
   source=f'from openassembler.pyfront import *\n@kernel\ndef k(a:Array({dtype}),out:Array(i32)):\n    x=a[0]\n    if 0<1<x:\n        out[0]=7\n'
   path=Path(temporary)/(dtype+'.py');path.write_text(source);text,_=compile_file(path);rows.append(dict(dtype=dtype,source=source,accepted=True,ptx_sha256=hashlib.sha256(text.encode()).hexdigest()))
 report=dict(schema=1,passed=True,results=rows,scope='Five original numeric literal bridges across adjacent comparison types; acceptance only, independent numerical and hardware gates remain separate',source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in(HERE/'candidate-pyfront.py',HERE/'candidate-modules.py',Path(__file__))})
 (HERE/'context-probe.json').write_text(json.dumps(report,indent=2)+'\n');print('Passed five contextual literal bridge admissions')
