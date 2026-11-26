"""Bind unchanged backend evidence and regenerate the preceding tuple source corpus."""
import argparse,hashlib,json,sys,tempfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('report',type=Path);args=parser.parse_args();args.report=args.report.resolve();report=json.loads(args.report.read_text());assert report['passed']
 from openassembler import build
 from openassembler.pyfront import compile_file
 from openassembler.ptx.parse import parse_module
 from openassembler.container.cubin import text_words,regcount
 from openassembler.backend.storage import local_frame_inventory
 from openassembler.native_ownership import validate
 prior_dir=HERE.parents[1]/'python_range_control/promotion';validation_path=prior_dir/'validation.json';prior=json.loads(validation_path.read_text());assert prior['passed']
 chain_path=prior_dir/'application-continuity.json';chain=json.loads(chain_path.read_text());assert chain['passed']and chain['native_comparison_count']==2774
 assert sha(chain_path)==prior['evidence_sha256']['application-continuity.json']
 inherited_chain_path=ROOT/chain['inherited_application_chain'];assert sha(inherited_chain_path)==chain['inherited_application_chain_sha256']
 inherited_chain=json.loads(inherited_chain_path.read_text());assert inherited_chain['passed']and inherited_chain['native_comparison_count']==2534
 ancestor_path=ROOT/inherited_chain['inherited_application_chain'];assert sha(ancestor_path)==inherited_chain['inherited_application_chain_sha256']
 ancestor=json.loads(ancestor_path.read_text());assert ancestor['passed']and ancestor['native_comparison_count']==2294
 for name,digest in ancestor['retained_hashes'].items():assert sha(ROOT/name)==digest,name
 changed={name for name,digest in prior['source_hashes'].items()if name.startswith('openassembler/')and sha(ROOT/name)!=digest}
 allowed={'openassembler/pyfront/compile.py','openassembler/pyfront/modules.py'};assert changed==allowed,changed
 assert all(sha(HERE/('baseline-'+kind+'.py'))==prior['source_hashes'][name]for name,kind in(('openassembler/pyfront/compile.py','pyfront'),('openassembler/pyfront/modules.py','modules')))
 # All executable backend inputs from the preceding full replay remain exact.
 inherited={name:digest for name,digest in chain['unchanged_backend_hashes'].items()if name not in allowed}
 assert all(sha(ROOT/name)==digest==report['source_hashes'][name]for name,digest in inherited.items())
 assert all(sha(ROOT/name)==digest for name,digest in report['source_hashes'].items()if name.startswith(('openassembler/','specs/')))
 proof_path=prior_dir/'promotion-audit.json';proof=json.loads(proof_path.read_text());assert proof['passed']and sha(proof_path)==prior['evidence_sha256']['promotion-audit.json']
 old_report=ROOT/proof['report'];old=json.loads(old_report.read_text());assert sha(old_report)==proof['report_sha256'];archive_path=old_report.parent/old['archive']['path'];assert sha(archive_path)==old['archive']['sha256']
 rows=[];source_rows=[]
 with tempfile.TemporaryDirectory(prefix='oa-return-predecessor-')as temporary,zipfile.ZipFile(archive_path)as archive:
  configurations=sorted({name.split('/')[0]for name in archive.namelist()if '/original/'in name})
  for configuration in configurations:
   folder=Path(temporary)/configuration
   for name in archive.namelist():
    prefix=configuration+'/original/'
    if name.startswith(prefix):
     path=folder/name.removeprefix(prefix);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(archive.read(name))
   entry='kernels/entry.py'if configuration=='remote_package'else'entry.py';text,names=compile_file(folder/entry)
   for level in('safe','mined','sched','opt'):
    prefix=configuration+'/'+level+'/';assert text.encode()==archive.read(prefix+'source.ptx'),configuration
    binary=archive.read(prefix+'oa.cubin');manifest=json.loads(archive.read(prefix+'manifest.json'));validate(manifest['entries'][0],binary)
    for name,kernel in parse_module(text).kernels.items():
     resources={};_,_,words,_,registers=build.generate(kernel,'sm_80',level,resource_info=resources)
     assert words==text_words(binary,name)and registers==regcount(binary,name)and resources['local_bytes']==local_frame_inventory(binary)[name]==0
     rows.append(dict(configuration=configuration,entry=name,level=level,registers=registers,passed=True))
   source_rows.append(dict(configuration=configuration,kernels=names,ptx_sha256=hashlib.sha256(text.encode()).hexdigest()))
 assert len(rows)==240
 result=dict(schema=1,passed=True,native_comparison_count=3014,inherited_native_comparison_count=2774,replayed_native_comparison_count=240,native_comparisons=rows,source_comparisons=source_rows,unchanged_backend_hashes=inherited,changed_production_inputs=sorted(changed),added_production_inputs=[],previous_validation=str(validation_path.relative_to(ROOT)),previous_validation_sha256=sha(validation_path),inherited_application_chain=str(chain_path.relative_to(ROOT)),inherited_application_chain_sha256=sha(chain_path),preceding_hardware_report=str(old_report.relative_to(ROOT)),preceding_hardware_report_sha256=sha(old_report),preceding_archive_sha256=sha(archive_path),validated_backend_report=str(args.report.relative_to(ROOT)),validated_backend_report_sha256=sha(args.report),auditor_sha256=sha(Path(__file__)),scope='2774 earlier direct/inherited native replay results inherited only after exact retained proof/archive hashes and unchanged executable backend identities; 240 preceding range/else originals newly regenerate to identical full PTX/native words/registers/frames. New comparison-chain semantics have their separate original-source A100 gate. No repeat GPU execution or performance claim')
 (HERE/'application-continuity.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed 240 regenerated native outputs; 2774 inherited under unchanged backend identities')
