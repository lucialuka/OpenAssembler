"""Fresh complete prior native-word replay under exact isolated narrow ABI snapshots."""
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.container.cubin import text_words,regcount
from openassembler.modelfront.kernels import canonical
from dataclasses import asdict

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
 validation_path=HERE.parent/'python_boolean_abi/helpers/validation.json';validation=json.loads(validation_path.read_text());assert validation['passed']
 base=HERE.parent/'python_boolean_values/promotion/application-continuity.json';previous=json.loads(base.read_text());assert previous['passed']and previous['native_comparison_count']==3254
 for name,digest in previous['retained_hashes'].items():assert sha((ROOT/name).read_bytes())==digest,name
 replacements=json.loads((HERE/'inputs.json').read_text());staged={name:HERE/('candidate-'+suffix+'.py')for suffix,name in replacements.items()}
 assert all(sha((HERE/('baseline-'+suffix+'.py')).read_bytes())==validation['source_hashes'][name]for suffix,name in replacements.items())
 backend={name:sha((staged.get(name,ROOT/name)).read_bytes())for name in set(previous['backend_hashes'])|set(staged)}
 files={};records=[];cache={}
 def checked(path):
  files[str(path.relative_to(ROOT))]=sha(path.read_bytes());return json.loads(path.read_text())
 def compare(source,binary,level,names,label):
  module=parse_module(source.decode());assert set(module.kernels)==set(names)
  for name,kernel in module.kernels.items():
   key=sha(canonical({'kernel':asdict(kernel),'level':level}))
   if key not in cache:
    _,_,words,_,registers=build.generate(kernel,'sm_80',level);cache[key]=(words,registers)
   words,registers=cache[key];old=text_words(binary,name);assert words==old and registers==regcount(binary,name),(label,name,level)
   from openassembler.isa.word import control,control_valid
   mask=31<<105;changes=[]
   for index,(current,retained)in enumerate(zip(words,old)):
    if current==retained:continue
    assert(current&~mask)==(retained&~mask),(label,name,level,index,'non-stall/yield native change')
    after,before=control(current),control(retained);assert control_valid(after)and after.stall>=before.stall,(label,name,level,index)
    changes.append(dict(instruction=index,before=asdict(before),after=asdict(after)))
   records.append(dict(source=label,entry=name,level=level,registers=registers,passed=True,native_exact=not changes,stall_yield_only_changes=changes))

 proofs=[ROOT/name for name in previous['numerical_proofs']]+[HERE.parent/'python_boolean_values/promotion/promotion-audit.json',HERE.parent/'python_boolean_abi/helpers/promotion-audit.json']

 for proof_path in proofs:
  print('Checking',str(proof_path.relative_to(ROOT)),flush=True)
  proof=checked(proof_path);assert proof['passed']
  reports=proof['reports']if isinstance(proof.get('reports'),list)else[dict(path=proof['report'],sha256=proof['report_sha256'])]
  for item in reports:
   path=ROOT/item['path'];report=checked(path);assert files[item['path']]==item['sha256']
   if item['path']in previous['retained_hashes']:
    assert files[item['path']]==previous['retained_hashes'][item['path']]
    assert files[str(proof_path.relative_to(ROOT))]==previous['retained_hashes'][str(proof_path.relative_to(ROOT))]
   else:
    foundation=json.loads((proof_path.parent/'validation.json').read_text());assert foundation['passed']
    assert foundation['evidence_sha256'][proof_path.name]==sha(proof_path.read_bytes())and foundation['evidence_sha256'][path.name]==sha(path.read_bytes())
   info=report.get('archive',report.get('artifact_archive'));archive_path=path.parent/info['path'];assert sha(archive_path.read_bytes())==info['sha256'];files[str(archive_path.relative_to(ROOT))]=info['sha256']
   with zipfile.ZipFile(archive_path)as archive:
    names=archive.namelist()
    if 'pressure.ptx'in names:
     source=archive.read('pressure.ptx');entries=list(parse_module(source.decode()).kernels)
     for level in('safe','mined','sched','opt'):compare(source,archive.read(level+'/candidate/native.cubin'),level,entries,item['path'])
    elif any(name in names for name in('generated.ptx','probes.ptx','fixtures.ptx','shuffle.ptx','local.ptx')):
     source_name=next(name for name in('generated.ptx','probes.ptx','fixtures.ptx','shuffle.ptx','local.ptx')if name in names);source=archive.read(source_name);entries=list(parse_module(source.decode()).kernels)
     for level in('safe','mined','sched','opt'):
      if source_name!='fixtures.ptx':compare(source,archive.read(level+'/oa.cubin'),level,entries,item['path'])
      else:
       for entry in entries:
        single_source=source
        single=parse_module(source.decode()).kernels[entry];_,_,words,_,registers=build.generate(single,'sm_80',level);binary=archive.read(f'{level}/{entry}/oa.cubin')
        retained=text_words(binary,entry);assert words==retained and registers==regcount(binary,entry)
        from openassembler.isa.word import control,control_valid
        mask=31<<105;changes=[]
        for index,(current,old)in enumerate(zip(words,retained)):
         if current!=old:
          assert(current&~mask)==(old&~mask)and control_valid(control(current))and control(current).stall>=control(old).stall
          changes.append(dict(instruction=index,before=asdict(control(old)),after=asdict(control(current))))
        records.append(dict(source=item['path'],entry=entry,level=level,registers=registers,passed=True,native_exact=not changes,stall_yield_only_changes=changes))

    elif any(name.endswith('/safe/source.ptx')for name in names):
     for source_name in [name for name in names if name.endswith('/safe/source.ptx')]:
      prefix=source_name.removesuffix('safe/source.ptx')
      for level in('safe','mined','sched','opt'):
       source=archive.read(prefix+level+'/source.ptx');entries=list(parse_module(source.decode()).kernels)
       compare(source,archive.read(prefix+level+'/oa.cubin'),level,entries,item['path']+':'+prefix)
    elif any(name.endswith('/source.ptx')for name in names):
     for source_name in [name for name in names if name.endswith('/source.ptx')]:
      prefix=source_name.removesuffix('source.ptx');source=archive.read(source_name);entries=list(parse_module(source.decode()).kernels)
      for level in('safe','mined','sched','opt'):
       binary_name=prefix+level+'/oa.cubin'
       if binary_name not in names:
        binary_name=prefix+'oa-'+level+'/native.cubin'
       compare(source,archive.read(binary_name),level,entries,item['path']+':'+prefix)
    elif any(name.endswith('/safe/merged.ptx')for name in names):
     for source_name in [name for name in names if name.endswith('/safe/merged.ptx')]:
      prefix=source_name.removesuffix('safe/merged.ptx')
      for level in('safe','mined','sched','opt'):
       source=archive.read(prefix+level+'/merged.ptx');entries=list(parse_module(source.decode()).kernels)
       compare(source,archive.read(prefix+level+'/oa.cubin'),level,entries,item['path']+':'+prefix)
    else:
     for name in names:
      if not name.endswith('/native-ledger.json'):continue
      prefix=name.removesuffix('native-ledger.json');ledger=json.loads(archive.read(name));plan=json.loads(archive.read(prefix+'plan.json'));assert sha(canonical(plan))==ledger['plan_sha256']
      source=archive.read(prefix+'sources/native.ptx');binary=archive.read(prefix+ledger['cubin']);assert sha(source)==ledger['source_sha256']and sha(binary)==ledger['cubin_sha256'];compare(source,binary,'opt',ledger['kernels'],item['path']+':'+prefix)
 assert len(records)==3734,len(records)
 assert all(row['native_exact'] for row in records)
 result=dict(schema=1,passed=True,native_comparison_count=3734,native_exact_count=3734,stall_yield_changed_count=0,backend_hashes=backend,numerical_proofs=[str(p.relative_to(ROOT))for p in proofs],retained_hashes=files,native_comparisons=records,previous_validation=str(validation_path.relative_to(ROOT)),previous_validation_sha256=sha(validation_path.read_bytes()),candidate_source_hashes={name:sha(path.read_bytes())for name,path in staged.items()},auditor_sha256=sha(Path(__file__).read_bytes()),scope='Fresh direct replay of every retained preceding 3734 native output under isolated narrow-ABI snapshots; all complete words/controls/register counts exact, with numerical proof/archive identities checked. Exact source snapshots must match production before integration binding; no previous numerical reports changed, GPU rerun or performance claim')
 (HERE/'application-continuity.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed 3734 direct native comparisons; every word/control/register exact',flush=True)
