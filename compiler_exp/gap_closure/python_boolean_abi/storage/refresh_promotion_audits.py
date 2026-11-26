"""Bind rerun observers after an independently passed policy identity update."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=='__main__':
 path=HERE/'promotion.json';d=json.loads(path.read_text());assert d['passed']
 for field in('changed_production_inputs','changed_test_inputs','added_test_inputs'):assert all(sha(ROOT/name)==value for name,value in d[field].items())
 for proof_name,report_name,script,count,explicit,byte_count in [('candidate-audit.json','candidate-a100-20261126T065029Z.json','audit.py',4320,432,18247680),('shapes-audit.json','shapes-a100-20261126T065356Z.json','audit_shapes.py',960,96,4055040)]:
  proof=json.loads((HERE/proof_name).read_text());assert proof['passed']and proof['report_sha256']==sha(HERE/report_name)and proof['auditor_sha256']==sha(HERE/script)and proof['launches']==count and proof['explicit_launches']==explicit and proof['bytes_checked']==byte_count
  d['evidence_sha256'][proof_name]=sha(HERE/proof_name)
 d['observer_refresh_script_sha256']=sha(Path(__file__));path.write_text(json.dumps(d,indent=2)+'\n');print('Exact production/test bytes unchanged; promotion binds full rerun current observers')
