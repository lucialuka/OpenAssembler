"""Verify retained artifacts and docstring-only post-hardware source changes."""
import ast
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent

def digest(raw):return hashlib.sha256(raw).hexdigest()

def semantic(raw):
    tree=ast.parse(raw)
    for node in ast.walk(tree):
        if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body=node.body[1:]
    return ast.dump(tree,include_attributes=False)


def main():
    records=[]
    for path in sorted(HERE.glob('*.json')):
        data=json.loads(path.read_text())
        if 'archive' not in data:continue
        artifact=HERE/data['archive']['path']
        assert digest(artifact.read_bytes())==data['archive']['sha256'],path
        report=data.get('report') or {}
        if report.get('passed'):
            assert data['returncode']==0,path
            assert all(row['passed'] for row in report.get('checks',[])),path
        records.append({'record':path.name,'archive':artifact.name,'sha256_matches':True,'returncode':data['returncode'],'passed':report.get('passed',False)})
    hardware=json.loads((HERE/'production-hardware-20261120T142112Z.json').read_text())
    sources=[]
    for item in json.loads((HERE/'promotion.json').read_text())['files']:
        name=item['path'];frozen=(HERE/'overlay'/name).read_bytes();current=(ROOT/name).read_bytes()
        assert digest(frozen)==hardware['compiler_hashes'][name],name
        assert semantic(frozen)==semantic(current),name
        sources.append({'path':name,'hardware_source_sha256':digest(frozen),'current_source_sha256':digest(current),
                        'unchanged_executable_ast':True,'byte_identical':frozen==current})
    result={'schema':1,'archives':records,'production_source_continuity':sources,
            'scope':'SHA-256 of every retained result archive; admitted result exit/check consistency; current modified modules equal hardware-frozen executable Python AST after removing docstrings only. Does not replace native numerical or performance gates.'}
    (HERE/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(len(records),'archives;',len(sources),'production modules verified')

if __name__=='__main__':main()
