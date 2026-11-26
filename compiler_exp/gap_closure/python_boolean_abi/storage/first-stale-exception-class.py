"""Validate exact pre/post capability rejection fixtures before production changes."""
import hashlib,json,re,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
from openassembler.pyfront import compile_file,NotSupported
if __name__=='__main__':
 path=HERE/'test-contract-updates.json';record=json.loads(path.read_text());results=[]
 with tempfile.TemporaryDirectory(prefix='oa-bool-storage-contract-')as temp:
  for prefix in('baseline','candidate'):
   install(prefix)
   for index,row in enumerate(record['updates']):
    value=row['before'if prefix=='baseline'else'after'];source=Path(temp)/f'{prefix}_{index}.py';source.write_text(value['source'])
    try:compile_file(source)
    except NotSupported as error:
     assert re.search(value['reason'],str(error)),(prefix,value,str(error));results.append(dict(snapshot=prefix,index=index,diagnostic=str(error),rejected=True))
    else:raise AssertionError((prefix,value))
 assert all(hashlib.sha256((ROOT/row['file']).read_bytes()).hexdigest()==row['baseline_sha256']for row in record['files'])
 record.update(passed=True,results=results,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());path.write_text(json.dumps(record,indent=2)+'\n');print('All eight original and eight candidate contract rejections pass')
