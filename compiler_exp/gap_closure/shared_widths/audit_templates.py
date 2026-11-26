"""Exact oracle/template/field-sweep data custody before normal-path promotion."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from openassembler import oracle
from openassembler.emu.program import parse_insn
from openassembler.isa.encode import Encoder,key


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__=='__main__':
    report=json.loads((HERE/'oracle-probe.json').read_text());templates=json.loads((HERE/'candidate-templates.json').read_text())
    sweep=json.loads((HERE/'field-sweep.json').read_text());assert sweep['passed']and len(sweep['results'])==12096
    assert sweep['template_sha256']==sha(HERE/'candidate-templates.json')
    files={};keys=[]
    for row in templates:
        source=next(row_ for row_ in report['results']if row_['dtype']==row['probe'])
        assert source['cubin_sha256']==row['oracle_cubin_sha256']
        for suffix in('.ptx','.cubin','.sass','.tool.log'):
            path=HERE/('shared_'+row['probe']+suffix);files[path.name]=sha(path)
        assert files['shared_'+row['probe']+'.ptx']==source['source_sha256']
        assert files['shared_'+row['probe']+'.cubin']==source['cubin_sha256']
        actual=[ins for ins in oracle.parse_listing((HERE/('shared_'+row['probe']+'.sass')).read_text())if ins.text==row['text']]
        assert len(actual)==1 and f'{actual[0].word:032x}'==row['word']
        instruction=parse_insn(0,row['text'],int(row['word'],16));template=Encoder._learn([instruction],canonical=False)
        assert template and template.slots==[('mem','mem+scale'),('reg','rb')]
        keys.append(repr(key(instruction)))
    assert len(set(keys))==4
    result=dict(schema=1,passed=True,templates=keys,field_sweep_words=12096,oracle_artifacts=files,
                template_sha256=sha(HERE/'candidate-templates.json'),field_sweep_sha256=sha(HERE/'field-sweep.json'),
                recovery_sha256=sha(HERE/'template-recovery.json'),script_sha256=sha(Path(__file__)),
                scope='Exact oracle-derived template data, unique learned mappings, full varied-disassembly corpus custody; hardware semantics/scheduling are separate')
    (HERE/'template-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',len(keys),'recovered templates;',result['field_sweep_words'],'field words')
