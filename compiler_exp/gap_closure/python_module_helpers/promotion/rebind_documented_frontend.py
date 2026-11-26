"""Bind the completed exact replay to its final documented, hardware-tested frontend."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def program(path):
 tree=ast.parse(path.read_text());assert isinstance(tree.body[0],ast.Expr)and isinstance(tree.body[0].value,ast.Constant)
 tree.body=tree.body[1:];return ast.dump(tree,include_attributes=False)
if __name__=='__main__':
 path=HERE/'application-continuity.json';prior=json.loads(path.read_text());assert prior['passed']and prior['native_comparison_count']==2090
 assert prior['auditor_sha256']==sha(HERE/'continuity.py')
 previous=HERE/'pre-doc-pyfront.py';current=ROOT/'openassembler/pyfront/compile.py'
 assert sha(previous)==prior['backend_hashes']['openassembler/pyfront/compile.py']and program(previous)==program(current)
 assert all(sha(ROOT/name)==digest for name,digest in prior['backend_hashes'].items()if name!='openassembler/pyfront/compile.py')
 report_path=HERE/'candidate-a100-20261123T222916Z.json';report=json.loads(report_path.read_text());proof=json.loads((HERE/'promotion-audit.json').read_text());assert report['passed']and proof['passed']and proof['report_sha256']==sha(report_path)
 assert all(report['source_hashes'][name]==sha(ROOT/name)for name in prior['backend_hashes'])
 base=HERE/'pre-doc-application-continuity.json';assert not base.exists();base.write_bytes(path.read_bytes())
 prior.update(base_audit=str(base.relative_to(ROOT)),base_audit_sha256=sha(base),base_auditor_sha256=prior['auditor_sha256'],auditor_sha256=sha(Path(__file__)),documentation_only_frontend_ast_equal=True,
              validated_backend_report=str(report_path.relative_to(ROOT)),validated_backend_report_sha256=sha(report_path),scope=prior['scope']+'; final frontend differs only in its module contract docstring, exact non-doc AST equality and separate final source-exact A100 admission required')
 prior['backend_hashes']['openassembler/pyfront/compile.py']=sha(current)
 prior['retained_hashes'][str(base.relative_to(ROOT))]=sha(base)
 path.write_text(json.dumps(prior,indent=2)+'\n');print('Bound 2090 exact native replay records to final hardware-tested documented frontend; non-doc AST identical')
