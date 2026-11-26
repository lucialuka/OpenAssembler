"""Record source-exact cross-module helper rejections without importing user code."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from openassembler.pyfront import compile_kernel,find_kernel

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
if __name__=='__main__':
 rows=[]
 for name in('signed','unsigned','floating'):
  try:compile_kernel(find_kernel(HERE/'corpus/entry.py',name))
  except Exception as error:rows.append(dict(entry=name,accepted=False,error=type(error).__name__+': '+str(error)))
  else:rows.append(dict(entry=name,accepted=True))
 assert all(not r['accepted']and'RuntimeError'not in r['error']for r in rows)
 result=dict(schema=1,passed=True,results=rows,source_hashes={str(p.relative_to(HERE)):sha(p)for p in sorted((HERE/'corpus').rglob('*.py'))},frontend_sha256=sha(ROOT/'openassembler/pyfront/compile.py'),helpers_sha256=sha(ROOT/'openassembler/pyfront/helpers.py'),script_sha256=sha(Path(__file__)),scope='Concrete next G04 source graph: local package, direct/module aliases and transitive relative helpers; existing frontend rejects call targets without executing any entry/helper/package top-level code. No candidate or hardware claim')
 (HERE/'baseline.json').write_text(json.dumps(result,indent=2)+'\n');print('Recorded',len(rows),'cross-module helper rejections')
