"""Promote only exact independently validated Boolean storage modules and fixtures."""
import hashlib,json,re,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=='__main__':
 previous_path=HERE.parent/'scalar/validation.json';previous=json.loads(previous_path.read_text());assert previous['passed']
 staged=json.loads((HERE/'inputs.json').read_text())
 for suffix,name in staged.items():assert sha(ROOT/name)==sha(HERE/('baseline-'+suffix+'.py'))==previous['source_hashes'][name]
 for report_name,proof_name,count,explicit,bytes_count,native in [('candidate-a100-20261126T065029Z.json','candidate-audit.json',4320,432,18247680,216),('shapes-a100-20261126T065356Z.json','shapes-audit.json',960,96,4055040,48)]:
  report=json.loads((HERE/report_name).read_text());proof=json.loads((HERE/proof_name).read_text());assert report['passed']and proof['passed']and proof['report_sha256']==sha(HERE/report_name)
  assert proof['launches']==count and proof['explicit_launches']==explicit and proof['bytes_checked']==bytes_count and len(proof['native_comparisons'])==native and proof['corrupted_output_rejected']
  assert sha(HERE/report['archive']['path'])==report['archive']['sha256']
  for suffix,name in staged.items():assert sha(HERE/('candidate-'+suffix+'.py'))==report['source_hashes'][name]
 for name,count,executions in [('candidate-offline.json',216,10368),('shapes-offline.json',48,2304)]:
  d=json.loads((HERE/name).read_text());assert d['passed']and len(d['results'])==count and d['executions']==executions and all(sha(HERE/path)==digest for path,digest in d['source_hashes'].items())
 assert re.search(r'208 passed in ',(HERE/'candidate-tests.log').read_text())and'FAILED 'not in(HERE/'candidate-tests.log').read_text()
 updates=json.loads((HERE/'test-contract-updates.json').read_text());assert updates['passed']and len(updates['updates'])==8 and len(updates['results'])==16
 assert all(sha(ROOT/row['file'])==row['baseline_sha256']and sha(HERE/('updated-'+Path(row['file']).name))==row['current_sha256']for row in updates['files'])
 for suffix,name in staged.items():shutil.copyfile(HERE/('candidate-'+suffix+'.py'),ROOT/name)
 for row in updates['files']:shutil.copyfile(HERE/('updated-'+Path(row['file']).name),ROOT/row['file'])
 added={'tests/test_pyfront_boolean_storage.py':'test_pyfront_boolean_storage.py','tests/test_pyfront_boolean_storage_contracts.py':'test_storage_contracts.py','tests/data/boolean_storage_fixtures.json':'boolean_storage_fixtures.json'}
 for target,source in added.items():assert not(ROOT/target).exists();shutil.copyfile(HERE/source,ROOT/target)
 result=dict(schema=1,passed=True,previous_validation_sha256=sha(previous_path),changed_production_inputs={name:sha(ROOT/name)for name in staged.values()},changed_test_inputs={row['file']:sha(ROOT/row['file'])for row in updates['files']},added_test_inputs={target:sha(ROOT/target)for target in added},evidence_sha256={name:sha(HERE/name)for name in('candidate-audit.json','shapes-audit.json','candidate-offline.json','shapes-offline.json','test-contract-updates.json','candidate-tests.log')},script_sha256=sha(Path(__file__)),scope='Exact five modules promoted after independent original-source/offline/A100/public-spec/type/host/storage gates; three explicit existing fixture contract updates and three added test/data files. Full integration/previous native/policy/installed/public remote/docs remain required')
 (HERE/'promotion.json').write_text(json.dumps(result,indent=2)+'\n');print('Promoted exact five storage modules; full integration pending')
