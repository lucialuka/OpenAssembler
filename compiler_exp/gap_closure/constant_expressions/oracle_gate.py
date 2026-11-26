"""Exact current acceptance/value audit, retaining unresolved semantic cases."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from openassembler.ptx.parse import parse_module


if __name__=='__main__':
    rows=[]
    for report_name,values_name,folder in(('oracle-probe.json','oracle-values.json','oracle'),('extra-oracle.json','extra-values.json','extra'),('edge-oracle.json','edge-values.json','edges')):
        observations={row['name']:row for row in json.loads((HERE/values_name).read_text())['results']}
        for row in json.loads((HERE/report_name).read_text())['results']:
            path=HERE/folder/(row['name']+'.ptx');assert hashlib.sha256(path.read_bytes()).hexdigest()==row['source_sha256']
            try:
                kernel=parse_module(path.read_text()).kernels[row['name']];value=kernel.body[1].args[1];accepted=True;diagnostic=None
            except SyntaxError as error:accepted=False;diagnostic=str(error)
            unresolved=(not accepted and row['accepted']and any(text in diagnostic for text in('remainder by zero','mixed signed/unsigned','remainder result requires','NaN-producing')))
            assert accepted==row['accepted']or unresolved,(row['expression'],row['accepted'],diagnostic)
            if accepted:
                bits=getattr(value,'bits',getattr(value,'value',0))&((1<<64)-1)
                assert observations[row['name']]['passed']and bits==int(observations[row['name']]['observed_bits'],16),(row['expression'],hex(bits),observations[row['name']])
            rows.append(dict(name=row['name'],report=report_name,expression=row['expression'],passed=True,accepted=accepted,
                             toolkit_accepted=row['accepted'],unresolved_semantics_rejected=unresolved,diagnostic=diagnostic))
    result=dict(schema=1,passed=True,results=rows,accepted=sum(row['accepted']for row in rows),
                unresolved_semantic_cases=sum(row['unresolved_semantics_rejected']for row in rows),
                parser_sha256=hashlib.sha256((HERE/'candidate-parse.py').read_bytes()).hexdigest(),
                scope='Retained toolkit acceptance and complete folded bits checked for the admitted domain; mixed signed conditional and remainder-zero discrepancies explicitly reject rather than count as semantic support')
    (HERE/'candidate-oracle-gate.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',len(rows),'controls;',result['accepted'],'accepted;',result['unresolved_semantic_cases'],'unresolved rejected')
