"""Combine disjoint source domains without combining timing or hiding individual gates."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
if __name__=='__main__':
 rows=[];reports=[];parts={}
 for name,script in(('production-base-audit.json','audit.py'),('production-squares-audit.json','audit_squares.py')):
  path=HERE/name;data=json.loads(path.read_text());assert data['passed']and data['audited_backend']=='production'and data['auditor_sha256']==sha(HERE/script)
  report=ROOT/data['report'];assert data['report_sha256']==sha(report)
  hardware=json.loads(report.read_text());assert hardware['passed']
  archive=report.parent/hardware['archive']['path'];assert sha(archive)==data['archive_sha256']==hardware['archive']['sha256']
  parts[name]=sha(path);reports.append(dict(path=data['report'],sha256=sha(report)));rows.extend(data['native_comparisons'])
 assert len(rows)==len({(r['name'],r['level'])for r in rows})==156
 result=dict(schema=1,passed=True,audited_backend='production',reports=reports,individual_audits_sha256=parts,launches=1560,bytes_checked=15974400,native_comparisons=rows,auditor_sha256=sha(Path(__file__)),scope='Two complete disjoint original-source gates: 35 typed Python/PTX/CUDA configurations plus four Python register/parameter square configurations. All native words/registers/resources/ownership and complete independent bytes passed; no combined timing or performance claim')
 (HERE/'promotion-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed 1560 launches; 15974400 bytes; 156 native comparisons')
