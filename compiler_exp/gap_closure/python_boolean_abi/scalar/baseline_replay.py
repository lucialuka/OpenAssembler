"""Freeze the original logical kernel ABI rejections across all source trees."""
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install('baseline')
from fixtures import configurations,write_tree,NAMES
from openassembler.pyfront import compile_kernel,find_kernel,NotSupported

if __name__=='__main__':
 rows=[]
 with tempfile.TemporaryDirectory(prefix='oa-bool-abi-baseline-')as temporary:
  for case in configurations():
   source=write_tree(Path(temporary)/case['name'],case)
   for name in NAMES:
    try:compile_kernel(find_kernel(source,name))
    except NotSupported as error:
     assert 'unsupported type annotation'in str(error)and error.path and error.lineno;rows.append(dict(configuration=case['name'],entry=name,rejected=True,diagnostic=str(error)))
    else:raise AssertionError(name)
 paths=[HERE/('baseline-'+suffix+'.py')for suffix in json.loads((HERE/'inputs.json').read_text())]+[HERE/'fixtures.py',Path(__file__)]
 result=dict(schema=1,passed=True,results=rows,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in paths},scope='Sixty original bool-annotated kernel rejections in three unexecuted trees using exact frozen pre-logical-ABI modules; no hardware claim')
 (HERE/'baseline-replay.json').write_text(json.dumps(result,indent=2)+'\n');print('Replayed sixty original logical Boolean kernel parameter rejections')
