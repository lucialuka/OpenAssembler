"""Learn missing narrow-store templates from retained oracle instruction words."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from openassembler import oracle
from openassembler.isa.encode import Encoder,key
from openassembler.emu.program import parse_insn
from openassembler.isa.word import control


if __name__=='__main__':
    report=json.loads((HERE/'oracle-probe.json').read_text());encoder=Encoder('sm_80');rows=[];maps=[]
    for row in report['results']:
        dtype=row['dtype']
        if dtype not in('u8','s8','u16','s16'):continue
        listing=HERE/f'shared_{dtype}.sass';binary=HERE/f'shared_{dtype}.cubin'
        assert hashlib.sha256(binary.read_bytes()).hexdigest()==row['cubin_sha256']
        instructions=[ins for ins in oracle.parse_listing(listing.read_text())if ins.text.split()[0].startswith('STS.')]
        assert len(instructions)==1
        observed=instructions[0];parsed=parse_insn(observed.addr,observed.text,observed.word)
        template=encoder._learn([parsed],canonical=False)
        assert template is not None and template.slots==[('mem','mem+scale'),('reg','rb')],template
        encoder.templates[key(parsed)]=template
        # Re-encode the complete observed instruction/control word. This proves
        # source custody and the known field mapping on the real oracle case;
        # varied disassembly and hardware remain separate gates.
        reconstructed=encoder.encode(observed.text,control(observed.word))
        assert reconstructed==observed.word
        rows.append(dict(probe=dtype,text=observed.text,word=f'{observed.word:032x}',oracle_cubin_sha256=row['cubin_sha256']))
        maps.append(dict(dtype=dtype,key=repr(key(parsed)),slots=template.slots,observed_word=f'{observed.word:032x}',reencoded_exact=True))
    result=dict(schema=1,passed=True,scope='Candidate template learning from exact oracle words using existing independently recovered memory/scale/register fields; no generated production tables modified, no varied-field or hardware proof',
                results=maps,oracle_report_sha256=hashlib.sha256((HERE/'oracle-probe.json').read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/'candidate-templates.json').write_text(json.dumps(rows,indent=2)+'\n')
    (HERE/'template-recovery.json').write_text(json.dumps(result,indent=2)+'\n');print('Recovered',len(rows),'exact oracle narrow-store templates')
