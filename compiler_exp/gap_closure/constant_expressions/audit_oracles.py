"""Read-only artifact/context audit distinguishing unresolved semantic cases."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(HERE)]
from openassembler.ptx.parse import parse_module


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(candidate=False):
    if candidate:
        from candidate import install
        install()
    reports={'oracle-probe.json':'oracle','extra-oracle.json':'extra','edge-oracle.json':'edges','context-oracle.json':'contexts'}
    rows=[];files={}
    for report_name,folder in reports.items():
        report=json.loads((HERE/report_name).read_text());files[report_name]=sha(HERE/report_name)
        for row in report['results']:
            prefix=HERE/folder/row['name'];source=prefix.with_suffix('.ptx')
            assert sha(source)==row['source_sha256'];files[str(source.relative_to(HERE))]=sha(source)
            try:parse_module(source.read_text());accepted=True;diagnostic=''
            except SyntaxError as error:accepted=False;diagnostic=str(error)
            unresolved=not accepted and row['accepted']and any(text in diagnostic for text in('remainder by zero','mixed signed/unsigned','remainder result requires','NaN-producing'))
            assert accepted==row['accepted']or unresolved,(row['name'],row.get('expression'),diagnostic)
            for suffix,key in(('.tool.log','tool_log_sha256'),('.cubin','cubin_sha256'),('.sass','sass_sha256')):
                if suffix=='.tool.log'or row['accepted']:
                    path=prefix.with_suffix(suffix);digest=sha(path)
                    if key in row:assert digest==row[key]
                    files[str(path.relative_to(HERE))]=digest
            rows.append(dict(name=row['name'],report=report_name,passed=True,accepted=accepted,toolkit_accepted=row['accepted'],
                             unresolved_semantics_rejected=unresolved,diagnostic=diagnostic))
    result=dict(schema=1,passed=True,results=rows,artifact_hashes=files,
                parser_sha256=sha(HERE/'candidate-parse.py'if candidate else ROOT/'openassembler/ptx/parse.py'),
                script_sha256=sha(Path(__file__)),scope='Exact raw oracle custody/current parsing with explicit unresolved toolkit/spec semantics; no new numerical or GPU claim')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--candidate',action='store_true');args=p.parse_args()
    result=audit(args.candidate);(HERE/('candidate-oracle-audit.json'if args.candidate else'oracle-evidence-audit.json')).write_text(json.dumps(result,indent=2)+'\n')
    print('Passed',len(result['results']),'controls',len(result['artifact_hashes']),'artifacts')
