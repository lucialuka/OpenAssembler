"""Preserve pre-candidate rejection for the exact original storage source trees."""
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install('baseline')
from fixtures import configurations,write_tree,NAMES
from openassembler.pyfront import compile_kernel,find_kernel,NotSupported
if __name__=='__main__':
 rows=[]
 with tempfile.TemporaryDirectory(prefix='oa-bool-storage-baseline-')as temporary:
  for case in configurations():
   source=write_tree(Path(temporary)/case['name'],case)
   for name in NAMES:
    try:compile_kernel(find_kernel(source,name),source_root=Path(temporary)/case['name']if case.get('explicit')else None)
    except NotSupported as error:
     assert 'annotation'in str(error)and "'bool'"in str(error)and error.path and error.lineno
     rows.append(dict(configuration=case['name'],entry=name,rejected=True,diagnostic=str(error)))
    else:raise AssertionError((case['name'],name))
 paths=[HERE/('baseline-'+suffix+'.py')for suffix in json.loads((HERE/'inputs.json').read_text())]+[HERE/name for name in('candidate.py','fixtures.py','baseline_replay.py')]
 data=dict(schema=1,passed=True,results=rows,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in paths},scope='54 exact original Boolean storage kernel annotation rejections using frozen production modules before candidate; no candidate or numerical/hardware claim')
 (HERE/'baseline-replay.json').write_text(json.dumps(data,indent=2)+'\n');print('Retained',len(rows),'original Boolean storage rejections')
