"""Retained oracle artifact inventory and production parse/context agreement."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from openassembler import oracle
from openassembler.ptx.parse import parse_module


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__=='__main__':
    groups={'oracle-matrix.json':'oracle','context-oracle.json':'contexts','range-oracle.json':'ranges',
            'widen-oracle.json':'widen','integer-context-oracle.json':'integer_contexts','store64-oracle.json':'store64'}
    files={};records=[]
    for report_name,folder in groups.items():
        report=json.loads((HERE/report_name).read_text());files[report_name]=sha(HERE/report_name)
        for row in report['results']:
            prefix=HERE/folder/row['name'];source=prefix.with_suffix('.ptx')
            assert sha(source)==row['source_sha256']
            try:parse_module(source.read_text());accepted=True
            except SyntaxError:accepted=False
            assert accepted==row['accepted'],(report_name,row['name'])
            for suffix,key in(('.cubin','cubin_sha256'),('.sass','sass_sha256'),('.tool.log','tool_log_sha256')):
                path=prefix.with_suffix(suffix)
                if row['accepted']or suffix=='.tool.log':
                    assert path.is_file(),path
                    digest=sha(path)
                    if key in row:assert digest==row[key],path
                    files[str(path.relative_to(HERE))]=digest
            if not accepted:assert 'error'in prefix.with_suffix('.tool.log').read_text().lower()
            files[str(source.relative_to(HERE))]=sha(source)
            records.append(dict(report=report_name,name=row['name'],expected_acceptance=accepted,passed=True))
    assert len(records)==110
    result=dict(schema=1,passed=True,results=records,artifact_hashes=files,parser_sha256=sha(ROOT/'openassembler/ptx/parse.py'),
                ptxas=oracle.run_tool('ptxas','--version').stdout,
                script_sha256=sha(Path(__file__)),scope='Read-only retained oracle source/binary/disassembly/diagnostic custody and current parse acceptance; no new toolkit numerical run or GPU claim')
    (HERE/'oracle-evidence-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',len(records),'controls',len(files),'artifacts')
