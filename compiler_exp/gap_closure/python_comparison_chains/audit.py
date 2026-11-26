"""Independent typed truth/atomic bytes and original native custody for chain research."""
import hashlib,json,struct,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.container.cubin import text_words,regcount
from openassembler.native_ownership import validate
def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
 path=HERE/'candidate-a100-20261124T111942Z.json';report=json.loads(path.read_text());assert report['passed']and report['error']is None
 assert report['harness_sha256']==sha((HERE/'run_modal.py').read_bytes())
 for name,digest in report['source_hashes'].items():assert sha((HERE/'candidate-pyfront.py'if name=='openassembler/pyfront/compile.py'else ROOT/name).read_bytes())==digest,name
 for name,digest in report['reference_hashes'].items():assert sha((HERE/name).read_bytes())==digest
 domains=json.loads((HERE/'domains.json').read_text());atomic=json.loads((HERE/'candidate-offline.json').read_text());cases={r['dtype']:r for r in domains['cases']};guard=b'\xa5'*64
 cases['atomic']=dict(source=atomic['source'],lanes=8,input_hex=(guard+struct.pack('<8i',*atomic['inputs'])+guard).hex())
 wanted={};before={};counters={}
 for name,case in cases.items():
  raw=bytes.fromhex(case['input_hex']);assert raw[:64]==raw[-64:]==guard;before[name]=raw
  fmt={'i32':'i','u32':'I','i64':'q','u64':'Q','f32':'f','atomic':'i'}[name];values=struct.unpack('<'+str(case['lanes'])+fmt,raw[64:-64]);expected=[];counts=[]
  for x in values:
   if name=='atomic':
    count=0;result=11
    if-3<x and x<7:result+=1
    if not(0<=x and x<=3):result+=2
    if x<0:
     old=count;count+=1
     if old>0:result+=4
    old=count;count+=1
    if-1<old and old<1:result+=8
    counts.append(count)
   else:
    low=0 if name.startswith('u')else-2;result=0
    if low<x and x<3:result+=1
    if low!=x and x!=3:result+=2
    if not(low<=x and x<=3):result+=4
    if low<=x and x<=3:result+=24
   expected.append(result)
  wanted[name]=guard+struct.pack('<'+str(len(values))+'i',*expected)+guard
  if name=='atomic':counters[name]=guard+struct.pack('<8i',*counts)+guard
 domain={(name,backend)for name in cases for backend in('ptxas','oa-safe','oa-mined','oa-sched','oa-opt')};keys=[(r['dtype'],r['backend'])for r in report['results']];assert len(keys)==len(set(keys))==30 and set(keys)==domain
 archive_path=HERE/report['archive']['path'];assert sha(archive_path.read_bytes())==report['archive']['sha256'];native=[];bytes_checked=0;binaries={}
 with zipfile.ZipFile(archive_path)as archive:
  for name,case in cases.items():
   assert archive.read(name+'/source.py')==case['source'].encode()
   for level in('safe','mined','sched','opt'):
    prefix=name+'/'+level+'/';text=archive.read(prefix+'source.ptx').decode();binary=archive.read(prefix+'oa.cubin');manifest=json.loads(archive.read(prefix+'manifest.json'));entry,=manifest['entries'];validate(entry,binary);assert entry['source_sha256']==sha(case['source'].encode());binaries[name,'oa-'+level]=binary
    if level=='safe':binaries[name,'ptxas']=archive.read(name+'/ptxas.cubin')
    _,_,words,_,registers=build.generate(parse_module(text).kernels['k'],'sm_80',level);assert words==text_words(binary,'k')and registers==regcount(binary,'k');native.append(dict(dtype=name,level=level,registers=registers,passed=True))
  for row in report['results']:
   name,backend=row['dtype'],row['backend'];actual=archive.read(f'{name}/{backend}/output.bin');after=archive.read(f'{name}/{backend}/input.bin');assert actual==wanted[name]and after==before[name]
   assert sha(actual)==row['output_sha256']and sha(after)==row['input_sha256']and sha(binaries[name,backend])==row['cubin_sha256']and row['resources']['num_regs']==regcount(binaries[name,backend],'k');bytes_checked+=len(actual)+len(after)
   if name in counters:
    count=archive.read(f'{name}/{backend}/counter.bin');assert count==counters[name]and sha(count)==row['counter_sha256'];bytes_checked+=len(count)
   else:assert row['counter_sha256']is None
  corrupt=bytearray(actual);corrupt[64]^=1;assert bytes(corrupt)!=wanted[name]
 assert len(native)==24 and bytes_checked==11080
 proof=dict(schema=1,passed=True,production_fix=False,report=path.name,report_sha256=sha(path.read_bytes()),archive_sha256=report['archive']['sha256'],launches=30,bytes_checked=bytes_checked,native_comparisons=native,corrupted_output_rejected=True,auditor_sha256=sha(Path(__file__).read_bytes()),scope='Independent all-input/output/counter/guard typed truth and atomic effect references, full original source/native ownership plus 24 exact native word/register matches; 30 finite A100 research launches only. No production, general mixed-type/first-class Boolean or complete source-graph/remote integration claim')
 (HERE/'candidate-audit.json').write_text(json.dumps(proof,indent=2)+'\n');print('Audited 30 original chain launches; 11080 bytes; 24 exact native outputs')
