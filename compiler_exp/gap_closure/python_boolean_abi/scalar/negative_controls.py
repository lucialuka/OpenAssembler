"""Source Boolean parameter admission preserves exact arithmetic/storage/effect limits."""
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from openassembler.pyfront import compile_file,NotSupported
CASES=[
 ('array','@kernel\ndef k(a:Array(bool)):\n    a[0]=True\n','annotation'),
 ('buffer','@kernel\ndef k(a:Buffer(bool,(4,))):\n    a[0]=True\n','annotation'),
 ('shared','@kernel\ndef k(a:Array(i32),flag:bool):\n    s=shared_array(4,bool)\n    a[0]=i32(flag)\n','annotation'),
 ('shadow','bool=i32\n@kernel\ndef k(a:Array(i32),flag:bool):\n    a[0]=i32(flag)\n','annotation'),
 ('default','@kernel\ndef k(a:Array(i32),flag:bool=True):\n    a[0]=i32(flag)\n','default'),
 ('numeric_reassign','@kernel\ndef k(a:Array(i32),flag:bool):\n    flag=1\n    a[0]=flag\n','different type'),
 ('arithmetic','@kernel\ndef k(a:Array(i32),flag:bool):\n    a[0]=flag+flag\n','arithmetic'),
 ('integer_comparison','@kernel\ndef k(a:Array(i32),flag:bool):\n    a[0]=i32(flag==1)\n','mismatched types'),
 ('unconverted_store','@kernel\ndef k(a:Array(i32),flag:bool):\n    a[0]=flag\n','storing'),
 ('union','@kernel\ndef k(a:Array(i32),flag:bool):\n    result=flag or a[0]\n    a[0]=i32(result)\n','one fixed scalar kind'),
 ('tuple_annotation','@kernel\ndef k(a:Array(i32),flag:(bool,bool)):\n    a[0]=1\n','annotation'),
 ('bool_numeric_helper','@device\ndef h(x:i32)->i32:\n    return x\n@kernel\ndef k(a:Array(i32),flag:bool):\n    a[0]=h(flag)\n','expected s32, got pred'),
 ('local_call_shadow','@kernel\ndef k(a:Array(i32),flag:bool):\n    bool=flag\n    a[0]=i32(bool(flag))\n','local value'),
 ('varying_barrier','@kernel\ndef k(a:Array(i32),flag:bool):\n    if flag and tid_x():\n        syncthreads()\n    a[tid_x()]=1\n','thread-dependent'),
]

if __name__=='__main__':
 rows=[]
 with tempfile.TemporaryDirectory(prefix='oa-bool-kernel-boundary-')as temporary:
  for name,body,reason in CASES:
   source='from openassembler.pyfront import *\n'+body;path=Path(temporary)/(name+'.py');path.write_text(source)
   try:compile_file(path)
   except NotSupported as error:
    assert reason in str(error),(name,reason,str(error));assert error.path and error.lineno;rows.append(dict(name=name,source=source,reason=reason,rejected=True,diagnostic=str(error)))
   else:raise AssertionError(name)
 paths=[HERE/('candidate-'+suffix+'.py')for suffix in json.loads((HERE/'inputs.json').read_text())]+[Path(__file__)]
 (HERE/'negative-controls.json').write_text(json.dumps(dict(schema=1,passed=True,results=rows,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in paths},scope='Fourteen original Boolean source array/storage/type/effect/annotation/convergence rejection boundaries; launch/metadata negatives have separate regression gates'),indent=2)+'\n');print('Passed fourteen original logical Boolean parameter boundaries')
