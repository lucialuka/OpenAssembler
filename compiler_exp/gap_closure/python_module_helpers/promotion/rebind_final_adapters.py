"""Retain exact full replay while binding final independently validated entry adapters."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def program(path,omit=()):
 tree=ast.parse(path.read_text())
 tree.body=[node for node in tree.body if not(isinstance(node,ast.Expr)and isinstance(node.value,ast.Constant)and isinstance(node.value.value,str))and not(isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))and node.name in omit)]
 return ast.dump(tree,include_attributes=False)
if __name__=='__main__':
 path=HERE/'application-continuity.json';prior=json.loads(path.read_text());assert prior['passed']and prior['native_comparison_count']==2090
 assert prior['auditor_sha256']==sha(HERE/'continuity.py')
 old_frontend=HERE/'pre-doc-pyfront.py';current_frontend=ROOT/'openassembler/pyfront/compile.py'
 assert sha(old_frontend)==prior['backend_hashes']['openassembler/pyfront/compile.py']and program(old_frontend)==program(current_frontend)
 old_build=HERE/'pre-routing-candidate-build.py';assert sha(old_build)==prior['backend_hashes']['openassembler/build.py']
 assert program(old_build,('_build_source',))==program(ROOT/'openassembler/build.py',('_build_source',))
 current_remote=ROOT/'openassembler/remote_build.py';old_remote=HERE/'pre-scope-remote.py'
 source=current_remote.read_text();new="or source.is_dir() and (root / rel.removeprefix('source/')).is_relative_to(source) and (root / rel.removeprefix('source/') in commands";old="or source.is_dir() and (root / rel.removeprefix('source/') in commands";assert source.count(new)==1;old_remote.write_text(source.replace(new,old))
 assert sha(old_remote)==prior['backend_hashes']['openassembler/remote_build.py']
 assert program(old_remote,('compile_native',))==program(current_remote,('compile_native',))
 exceptions={'openassembler/pyfront/compile.py','openassembler/build.py','openassembler/remote_build.py'}
 assert all(sha(ROOT/name)==digest for name,digest in prior['backend_hashes'].items()if name not in exceptions)
 report_path=HERE/'candidate-a100-20261123T224420Z.json';report=json.loads(report_path.read_text());proof=json.loads((HERE/'promotion-audit.json').read_text());assert report['passed']and proof['passed']and proof['report_sha256']==sha(report_path)and proof['audited_backend']=='production'
 names=set(prior['backend_hashes'])|{'openassembler/frontend.py'}
 assert all(report['source_hashes'][name]==sha(ROOT/name)for name in names)
 base=HERE/'pre-final-adapter-application-continuity.json';assert not base.exists();base.write_bytes(path.read_bytes())
 result=dict(prior);result.update(base_audit=str(base.relative_to(ROOT)),base_audit_sha256=sha(base),base_auditor_sha256=prior['auditor_sha256'],auditor_sha256=sha(Path(__file__)),
  frontend_non_doc_ast_equal=True,build_ast_outside_entry_adapter_equal=True,remote_ast_outside_client_target_discovery_equal=True,
  validated_backend_report=str(report_path.relative_to(ROOT)),validated_backend_report_sha256=sha(report_path),
  scope=prior['scope']+'; final entry adapters and directory target discovery have a separate exact-source CLI/worker A100 gate, unchanged build code outside _build_source and remote code outside compile_native verified structurally; frontend executable AST is identical')
 result['backend_hashes']={name:sha(ROOT/name)for name in names};result['retained_hashes'][str(base.relative_to(ROOT))]=sha(base)
 path.write_text(json.dumps(result,indent=2)+'\n');print('Bound 2090 complete native replay records to final source-exact adapter admission; generator AST unchanged')
