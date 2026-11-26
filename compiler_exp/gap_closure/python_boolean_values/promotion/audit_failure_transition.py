"""Bind the actual stale-predicate failure to control-only repaired hardware outputs."""
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from fixtures import configurations,NAMES,raw_input,expected,expected_counter
from openassembler.container.cubin import text_words,regcount
from openassembler.isa.word import control,control_valid
def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
 old_path=HERE/'candidate-a100-20261124T115739Z.json';new_path=HERE/'candidate-a100-20261124T120418Z.json';old=json.loads(old_path.read_text());new=json.loads(new_path.read_text());assert old['passed']is False and old['error']is None and new['passed']and new['error']is None
 assert old['harness_sha256']==sha((HERE/'first-run_modal.py').read_bytes())and new['harness_sha256']==sha((HERE/'run_modal.py').read_bytes())
 for name in('openassembler/pyfront/compile.py','openassembler/pyfront/modules.py','compiler_exp/gap_closure/python_boolean_values/promotion/fixtures.py'):assert old['source_hashes'][name]==new['source_hashes'][name]
 old_archive,new_archive=(HERE/row['archive']['path']for row in(old,new));assert sha(old_archive.read_bytes())==old['archive']['sha256']and sha(new_archive.read_bytes())==new['archive']['sha256']
 mask=31<<105;native=[];failures=[];guard=b'\xa5'*256
 with zipfile.ZipFile(old_archive)as a,zipfile.ZipFile(new_archive)as b:
  for case in configurations():
   name=case['name']
   for path,text in case['files'].items():assert a.read(name+'/original/'+path)==b.read(name+'/original/'+path)==text.encode()
   for level in('safe','mined','sched','opt'):
    prefix=name+'/'+level+'/';assert a.read(prefix+'source.ptx')==b.read(prefix+'source.ptx');before,after=(z.read(prefix+'oa.cubin')for z in(a,b))
    for entry in NAMES:
     left,right=text_words(before,entry),text_words(after,entry);assert len(left)==len(right)and regcount(before,entry)==regcount(after,entry);changes=[]
     for index,(oldword,newword)in enumerate(zip(left,right)):
      if oldword==newword:continue
      assert(oldword&~mask)==(newword&~mask)and control_valid(control(newword))and control(newword).stall>=control(oldword).stall
      changes.append(dict(instruction=index,before=vars(control(oldword)),after=vars(control(newword))))
     native.append(dict(configuration=name,entry=entry,level=level,native_exact=not changes,stall_yield_only_changes=changes,passed=True))
  assert len(old['results'])==len(new['results'])==2400
  for before,after in zip(old['results'],new['results']):
   keys=('configuration','compiler','entry','seed','n');assert tuple(before[k]for k in keys)==tuple(after[k]for k in keys)
   case,backend,entry,seed,n=(after[k]for k in keys);prefix=f'{case}/{backend}/{entry}/{seed}/{n}/';wanted=guard+expected(entry,seed,n,384)+guard;counts=guard+expected_counter(entry,seed,n,384)+guard
   assert a.read(prefix+'before-a.bin')==b.read(prefix+'before-a.bin')==guard+raw_input(entry,seed,384)+guard
   assert a.read(prefix+'a.bin')==b.read(prefix+'a.bin')==a.read(prefix+'before-a.bin');assert a.read(prefix+'counter.bin')==b.read(prefix+'counter.bin')==counts
   previous=a.read(prefix+'output.bin');current=b.read(prefix+'output.bin');assert current==wanted and after['passed'];assert previous[:256]==previous[-256:]==guard;assert before['passed']==(previous==wanted)
   if previous!=wanted:
    assert entry=='predicate_while'and backend in('oa-mined','oa-sched','oa-opt')and n>0
    failures.append({k:before[k]for k in keys})
 assert len(native)==240 and len(failures)==54
 result=dict(schema=1,passed=True,original_failed_report=old_path.name,original_failed_report_sha256=sha(old_path.read_bytes()),repaired_report=new_path.name,repaired_report_sha256=sha(new_path.read_bytes()),original_failures=failures,original_failure_count=54,repaired_launches=2400,native_comparisons=native,changed_control_outputs=sum(not r['native_exact']for r in native),auditor_sha256=sha(Path(__file__).read_bytes()),scope='Exact original source/PTX/input/counter/guard identity and 54 actual mined/sched/opt loop-carried predicate mismatches repaired under unchanged independent references. All 240 native instruction/operand/register identities retain only explicitly recorded nondecreasing stall/yield changes; no opcode, allocation, tolerance, emulator or numerical reference repair')
 (HERE/'failure-transition-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Audited 54 original hardware failures -> 2400 repaired launches;',result['changed_control_outputs'],'of 240 outputs changed only stall/yield bits')
