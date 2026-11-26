"""Bind final export check, wheel audit and docs after all docs/register edits."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=='__main__':
 result=subprocess.run([sys.executable,str(HERE/'check_exported_docs.py')],cwd=ROOT,stdout=open(HERE/'exported-docs-check.log','w'));assert result.returncode==0
 result=subprocess.run([sys.executable,str(HERE/'audit_wheel.py')],cwd=ROOT,stdout=open(HERE/'wheel-audit.log','w'));assert result.returncode==0
 result=subprocess.run([sys.executable,str(HERE/'audit_wheel_shapes.py')],cwd=ROOT,stdout=open(HERE/'wheel-shapes-audit.log','w'));assert result.returncode==0
 result=subprocess.run([sys.executable,str(ROOT/'tools/check_docs.py')],cwd=ROOT,stdout=open(HERE/'docs-check.log','w'));assert result.returncode==0
 record_path=HERE/'validation.json';record=json.loads(record_path.read_text());record['passed']=True;record['all_gaps_closed']=False;record['docs']={'status':'passed','checks':{name:'Documentation check passed:'in(HERE/name).read_text()for name in('docs-check.log','exported-docs-check.log')}}
 record['evidence_sha256'].update({name:sha(HERE/name)for name in('wheel-audit.json','wheel-shapes-audit.json')});record['logs_sha256']={p.name:sha(p)for p in HERE.glob('*.log')if p.name!='record-validation.log'}
 record['experiment_source_hashes']={str(p.relative_to(HERE)):sha(p)for p in sorted(HERE.rglob('*'))if p.suffix in('.py','.ptx','.cu')and'__pycache__'not in p.parts and'wheel'not in p.parts and'normal-remote-build'not in p.parts}
 record['source_hashes']={str(p.relative_to(ROOT)):sha(p)for base in('openassembler','tests')for p in sorted((ROOT/base).rglob('*'))if p.suffix in('.py','.json','.ptx')and'__pycache__'not in p.parts};record['source_hashes']['specs/sm_80/memory_templates.json']=sha(ROOT/'specs/sm_80/memory_templates.json')
 record['evidence_sha256'].update({name:sha(HERE/name)for name in('wheel-audit.json','wheel-shapes-audit.json')})
 record_path.write_text(json.dumps(record,indent=2)+'\n')
