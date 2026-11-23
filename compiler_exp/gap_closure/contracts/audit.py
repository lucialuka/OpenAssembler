"""Check exhaustive form inventory, fixture identity and cross-stage custody."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from openassembler.backend.contracts import scalar_forms
from openassembler.ptx.parse import parse_module
HERE=Path(__file__).resolve().parent

def run():
    ledger=json.loads((HERE/'ledger.json').read_text())
    artifact=ROOT/ledger['fixture_artifact']
    assert hashlib.sha256(artifact.read_bytes()).hexdigest()==ledger['fixtures_sha256']
    fixtures=json.loads(artifact.read_text())
    declared={'.'.join((op,*mods)):arity for (op,mods),arity in scalar_forms().items()}
    assert {row['form']:row['arity'] for row in ledger['forms']}==declared
    rows=[*ledger['forms'],*(row['parameter_backed_variant'] for row in ledger['forms']),
          *ledger['boundary_and_negative_controls'],*ledger['memory_width_matrix']]
    for row in rows:
        source=fixtures[row['fixture_id']]
        assert hashlib.sha256(source.encode()).hexdigest()==row['fixture_sha256']
        module=parse_module(source)
        ins=next(i for i in module.kernels['probe'].body if i.line==row['target_line'])
        assert '.'.join((ins.op,*ins.mods))==row['form']
        if row['stages']['encoding']['status']=='passed':
            assert all(row['stages'][s]['status']=='passed' for s in
                       ('parse','admission','selection','register_allocation','predicate_allocation_and_lowering','scheduling'))
        assert all(row['stages'][s]['status']=='not-run' for s in
                   ('independent_cpu_semantics','toolkit','packaging','hardware','performance'))
    for name,want in ledger['source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==want,name
    return {'schema':1,'passed':True,'declared_forms':len(declared),'rows_checked':len(rows),
            'unique_fixtures':len(fixtures),'ledger_sha256':hashlib.sha256((HERE/'ledger.json').read_bytes()).hexdigest(),
            'hardware':'not run; audit verifies inventory and evidence custody only'}

if __name__=='__main__':
    result=run();(HERE/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
