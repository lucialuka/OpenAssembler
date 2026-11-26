"""Read-only complete source/stage/native custody audit of the width matrix."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(ROOT))
from openassembler.backend.contracts import scalar_forms,validate_scalar,ContractError
from openassembler.backend import ra
from openassembler.isa.encode import Encoder
from openassembler.ptx.parse import parse_module

def sha(data):return hashlib.sha256(data).hexdigest()

if __name__=='__main__':
    path=HERE/'width-matrix.json';matrix=json.loads(path.read_text());assert matrix['passed']
    assert all(sha((ROOT/name).read_bytes())==digest for name,digest in matrix['source_hashes'].items())
    fixture_path=HERE/'width-matrix-fixtures.json';assert sha(fixture_path.read_bytes())==matrix['fixtures_sha256']
    fixtures=json.loads(fixture_path.read_text());forms=scalar_forms();encoder=Encoder('sm_80');seen=set();encodings=0
    for row in matrix['results']:
        ident=row['fixture_id'];assert ident not in seen;seen.add(ident)
        source=fixtures[ident];assert sha(source.encode())==row['fixture_sha256']
        kernel=parse_module(source).kernels['probe'];ins=next(i for i in kernel.body if i.line==row['target_line'])
        assert '.'.join((ins.op,*ins.mods))==row['form']
        declared=(ins.op,ins.mods)in forms;assert declared==row['declared']
        if declared:
            validate_scalar(ins);code,stats=ra.compile_kernel(kernel);words=ra.encode_code(encoder,code,stats['labels'])
            stage=row['stages']['encoding'];assert stage['status']=='passed'
            assert stage['instruction_count']==len(words)and stage['machine_words_sha256']==sha(b''.join(w.to_bytes(16,'little')for w in words))
            encodings+=1
        else:
            try:validate_scalar(ins)
            except ContractError:pass
            else:raise AssertionError(row['form'])
            assert row['stages']['admission']['status']=='rejected'
        for stage in('independent_cpu_semantics','toolkit','packaging','hardware','performance'):assert row['stages'][stage]['status']=='not-run'
    assert len(seen)==len(fixtures)==336 and encodings==172
    assert all(dict(Counter(row['stages'][stage]['status']for row in matrix['results']))==counts for stage,counts in matrix['counts'].items())
    result=dict(schema=1,passed=True,matrix_sha256=sha(path.read_bytes()),fixtures_sha256=sha(fixture_path.read_bytes()),
                rows=336,native_comparisons=encodings,rejected_forms=164,script_sha256=sha(Path(__file__).read_bytes()),
                scope='Read-only retained matrix input/stage/fixture identity and complete encoded-word replay; numerical/toolkit/package/hardware/performance remain explicitly not run for these fixtures')
    (HERE/'width-matrix-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',len(seen),'width rows;',encodings,'exact native comparisons')
