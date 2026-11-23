"""Reproducible structural inventory; never imports repository source files."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def inventory():
    names = subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    rows=[]
    for name in sorted(filter(None,names)):
        path=ROOT/name
        if not path.is_file():
            continue
        raw=path.read_bytes()
        row={'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        if path.suffix in {'.py','.md','.toml','.sh','.ptx','.cu','.c','.cpp','.cuh','.json'}:
            row['lines']=len(raw.splitlines())
        if path.suffix=='.py':
            tree=ast.parse(raw,filename=name)
            row['docstring']=ast.get_docstring(tree)
            row['definitions']=[{'name':node.name,'line':node.lineno,'end_line':node.end_lineno,
                                 'kind':type(node).__name__}
                                for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
            row['imports']=sorted({node.module or '' for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)} |
                                  {alias.name for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names})
        rows.append(row)
    return {'schema':1,'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'scope':'all Git-tracked files read and hashed; AST inventory for all Python files; structural inventory is not a claim of manual semantic review of every historical experiment',
            'counts':dict(Counter(row['path'].split('/')[0] for row in rows)),
            'files':rows}


if __name__=='__main__':
    output=ROOT/'compiler_exp/gap_closure/repository-map.json'
    output.write_text(json.dumps(inventory(),indent=2)+'\n')
    print(output)
