"""Read-only complete original-link/merged-hypothesis research gate audit."""
import hashlib,json,random,struct,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.container.cubin import text_words,regcount
from openassembler.native_ownership import validate

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
 path=HERE/'research-a100-20261123T182610Z.json';report=json.loads(path.read_text());assert report['passed']and len(report['results'])==48
 assert report['harness_sha256']==sha((HERE/'run_modal.py').read_bytes())
 assert all(sha((ROOT/name).read_bytes())==digest for name,digest in report['source_hashes'].items())
 archive_path=HERE/report['archive']['path'];assert sha(archive_path.read_bytes())==report['archive']['sha256']
 rows=[];count=0;guard=b'\xa5'*256
 with zipfile.ZipFile(archive_path)as archive:
  for name in('entry','helper'):assert archive.read(name+'.cu')==(HERE/(name+'.cu')).read_bytes()
  entry=archive.read('entry.ptx').decode();helper=archive.read('helper.ptx').decode();merged=archive.read('merged.ptx').decode()
  assert merged==entry.replace('.extern .func','.visible .func')+'\n'+'\n'.join(line for line in helper.splitlines()if not line.startswith(('.version','.target','.address_size')))
  module=parse_module(merged);kernel=module.kernels['linked_probe'];binaries={'original-linked':archive.read('linked.cubin'),'merged-ptxas':archive.read('merged-ptxas.cubin')}
  for level in('safe','mined','sched','opt'):
   binary=archive.read(level+'/oa.cubin');manifest=json.loads(archive.read(level+'/manifest.json'));validate(manifest['entries'][0],binary)
   _,_,words,_,registers=build.generate(kernel,'sm_80',level);assert words==text_words(binary,'linked_probe')and registers==regcount(binary,'linked_probe')
   rows.append(dict(level=level,passed=True,registers=registers));binaries['oa-'+level]=binary
  domain={(compiler,n,seed)for compiler in binaries for n in(0,1,259,384)for seed in(19271,19272)}
  assert{(r['compiler'],r['n'],r['seed'])for r in report['results']}==domain
  for row in report['results']:
   compiler,n,seed=(row[k]for k in('compiler','n','seed'));prefix=f'{compiler}/{n}/{seed}/';rng=random.Random(seed)
   data=[0,1,0xffffffff,0x80000000]+[rng.getrandbits(32)for _ in range(380)];before=archive.read(prefix+'input-before.bin');unchanged=archive.read(prefix+'input.bin');actual=archive.read(prefix+'output.bin')
   assert before==guard+struct.pack('<384I',*data)+guard and unchanged==before
   expected=[(value*17+9)&0xffffffff if i<n else 0x13579bdf for i,value in enumerate(data)]
   assert actual==guard+struct.pack('<384I',*expected)+guard and row['passed']
   assert sha(actual)==row['output_sha256']and sha(unchanged)==row['input_sha256']and sha(binaries[compiler])==row['cubin_sha256']
   count+=len(actual)+len(unchanged)
 result=dict(schema=1,passed=True,report_sha256=sha(path.read_bytes()),archive_sha256=report['archive']['sha256'],launches=48,bytes_checked=count,native_comparisons=rows,script_sha256=sha(Path(__file__).read_bytes()),scope='Complete research-only original two-TU CUDA-linked control versus exact merged-source hypothesis/full bytes and OA-native custody; production linking remains unimplemented')
 (HERE/'research-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',48,count,len(rows))
