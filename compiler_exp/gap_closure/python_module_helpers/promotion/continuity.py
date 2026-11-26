"""Current executable continuity across the complete preceding numerical proof chain."""
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.container.cubin import text_words,regcount
from openassembler.modelfront.kernels import canonical
from dataclasses import asdict

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
 validation_path=HERE.parents[1]/'wide_multiply/promotion/validation.json';validation=json.loads(validation_path.read_text());assert validation['passed']
 previous_path=HERE.parents[1]/'wide_multiply/promotion/application-continuity.json';previous=json.loads(previous_path.read_text());assert previous['passed']and previous['native_comparison_count']==1934
 changed={name for name,digest in validation['source_hashes'].items()if name.startswith('openassembler/')and sha((ROOT/name).read_bytes())!=digest}
 assert {'openassembler/pyfront/compile.py','openassembler/pyfront/helpers.py','openassembler/build.py','openassembler/native_ownership.py','openassembler/remote_build.py','openassembler/cli.py'}<=changed<={'openassembler/pyfront/compile.py','openassembler/pyfront/helpers.py','openassembler/build.py','openassembler/native_ownership.py','openassembler/remote_build.py','openassembler/cli.py','openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json'},changed
 baseline={'openassembler/pyfront/compile.py':'baseline-pyfront.py','openassembler/pyfront/helpers.py':'baseline-helpers.py','openassembler/build.py':'baseline-build.py','openassembler/native_ownership.py':'baseline-ownership.py','openassembler/remote_build.py':'baseline-remote.py','openassembler/cli.py':'baseline-cli.py'}
 assert all(sha((HERE/path).read_bytes())==validation['source_hashes'][name]for name,path in baseline.items())
 report_path=HERE/'candidate-a100-20261123T222132Z.json';admission=json.loads(report_path.read_text());assert admission['passed']
 backend={name:sha((ROOT/name).read_bytes())for name in set(previous['backend_hashes'])|set(baseline)|{'openassembler/ptx/link.py','openassembler/pyfront/modules.py'}}
 assert all(admission['source_hashes'][name]==digest for name,digest in backend.items())
 files={};records=[];cache={}
 def checked(path):
  files[str(path.relative_to(ROOT))]=sha(path.read_bytes());return json.loads(path.read_text())
 def compare(source,binary,level,names,label):
  module=parse_module(source.decode());assert set(module.kernels)==set(names)
  for name,kernel in module.kernels.items():
   key=sha(canonical({'kernel':asdict(kernel),'level':level}))
   if key not in cache:
    _,_,words,_,registers=build.generate(kernel,'sm_80',level);cache[key]=(words,registers)
   words,registers=cache[key];assert words==text_words(binary,name)and registers==regcount(binary,name),(label,name,level)
   records.append(dict(source=label,entry=name,level=level,registers=registers,passed=True))
 proofs=[ROOT/name for name in previous['numerical_proofs']]+[HERE.parents[1]/'wide_multiply/promotion/promotion-audit.json']
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
    assert proof_path==HERE.parents[1]/'wide_multiply/promotion/promotion-audit.json'
    assert files[item['path']]==validation['evidence_sha256'][path.name]and files[str(proof_path.relative_to(ROOT))]==validation['evidence_sha256'][proof_path.name]
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
        single=parse_module(source.decode()).kernels[entry];_,_,words,_,registers=build.generate(single,'sm_80',level);binary=archive.read(f'{level}/{entry}/oa.cubin')
        assert words==text_words(binary,entry)and registers==regcount(binary,entry);records.append(dict(source=item['path'],entry=entry,level=level,registers=registers,passed=True))
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
 assert len(records)==2090,len(records)
 result=dict(schema=1,passed=True,scope='Prior full numerical/ownership proof chain and complete current native words/registers; local-package Python helpers have their own original-source A100 admission',hardware='not run; exact executable continuity only',performance='not measured',backend_hashes=backend,changed_production_inputs=sorted(changed),added_production_inputs=['openassembler/pyfront/modules.py'],baseline_hashes={name:sha((HERE/path).read_bytes())for name,path in baseline.items()},previous_validation=str(validation_path.relative_to(ROOT)),previous_validation_sha256=sha(validation_path.read_bytes()),previous_application_chain=str(previous_path.relative_to(ROOT)),previous_application_chain_sha256=sha(previous_path.read_bytes()),validated_backend_report=str(report_path.relative_to(ROOT)),validated_backend_report_sha256=sha(report_path.read_bytes()),numerical_proofs=[str(p.relative_to(ROOT))for p in proofs],retained_hashes=files,native_comparisons=records,native_comparison_count=len(records),auditor_sha256=sha(Path(__file__).read_bytes()))
 (HERE/'application-continuity.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',len(records),'prior native occurrences')
