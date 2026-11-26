"""Retain a fresh memory-width admission/lowering matrix without rewriting history."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(ROOT))
from openassembler.backend.contracts import scalar_forms
from openassembler.isa.encode import Encoder

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

if __name__=='__main__':
    spec=importlib.util.spec_from_file_location('width_fixture_ledger',HERE.parent/'contracts/ledger.py')
    ledger=importlib.util.module_from_spec(spec);spec.loader.exec_module(ledger)
    encoder=Encoder('sm_80');forms=scalar_forms();rows=[]
    for op in('ld','st'):
        for space in('global','shared'):
            for dtype in('b8','u8','s8','b16','u16','s16','b32','u32','s32','f32','b64','u64','s64','f64'):
                for width in(1,2,4):
                    mods=(space,)+(('v'+str(width),)if width>1 else())+(dtype,)
                    for registers in(True,False):
                        row=ledger.evaluate(op,mods,2,encoder,declared=(op,mods)in forms,register_backed=registers)
                        assert row['stages']['parse']['status']=='passed',row
                        assert(row['stages']['admission']['status']=='passed')==row['declared'],row
                        if row['declared']:assert row['stages']['encoding']['status']=='passed',row
                        row['fixture_artifact']='compiler_exp/gap_closure/shared_widths/width-matrix-fixtures.json';rows.append(row)
    assert len(rows)==336
    fixtures_path=HERE/'width-matrix-fixtures.json';fixtures_path.write_text(json.dumps(ledger.FIXTURES,indent=2)+'\n')
    inputs=[ROOT/'openassembler/backend'/n for n in('ra.py','contracts.py','compile.py','sched.py','storage.py','spill.py','pressure.py')]+[ROOT/'openassembler/isa/encode.py',ROOT/'openassembler/ptx/parse.py',ROOT/'specs/sm_80/memory_templates.json',HERE.parent/'contracts/ledger.py',Path(__file__)]
    result=dict(schema=1,passed=True,scope='Global/shared scalar/vector width matrix for register and parameter-backed representative operands; admission through encoding only, semantics/toolkit/package/hardware/performance explicitly not run for these fixtures',
                results=rows,counts={stage:dict(Counter(row['stages'][stage]['status']for row in rows))for stage in rows[0]['stages']},
                source_hashes={str(p.relative_to(ROOT)):sha(p)for p in inputs},fixtures_sha256=sha(fixtures_path))
    (HERE/'width-matrix.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',len(rows),'width rows;',result['counts']['admission'],result['counts']['encoding'])
