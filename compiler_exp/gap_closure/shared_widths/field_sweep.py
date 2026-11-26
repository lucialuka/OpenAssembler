"""Varied disassembly for exact recovered shared narrow-store templates."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from openassembler import oracle
from openassembler.emu.program import parse_insn
from openassembler.emu.operands import Mem
from openassembler.isa.encode import Encoder,key
from openassembler.isa.word import Control


if __name__=='__main__':
    encoder=Encoder('sm_80');rows=json.loads((HERE/'candidate-templates.json').read_text())
    for row in rows:
        instruction=parse_insn(0,row['text'],int(row['word'],16))
        template=encoder._learn([instruction],canonical=False);assert template and all(field is not None for _,field in template.slots)
        encoder.templates[key(instruction)]=template
    ctl=Control(6,False,7,7,0,0);records=[];words=[]
    for dtype in('U8','S8','U16','S16'):
        for base in(0,3,37,126,254,255):
            for value in(0,2,19,95,253,255):
                for offset in(-16,0,2,127,4096,8388607,-8388608):
                    for scale in(1,4,8,16):
                        # Deliberately unaligned offsets are encoding controls,
                        # not admitted aligned memory accesses or executions.
                        suffix=f'{offset:+#x}'if offset else''
                        indexed=f'R{base}'+(f'.X{scale}'if scale!=1 else'')
                        text=f'STS.{dtype} [{indexed}{suffix}], R{value}'
                        for guard in('', '@P2 ','@!P5 '):
                            instruction=guard+text;word=encoder.encode(instruction,ctl)
                            records.append(dict(text=instruction,word=f'{word:032x}'));words.append(word)
    decoded=oracle.disasm_words(words)
    assert len(decoded)==len(words)
    for row,text in zip(records,decoded):
        assert text is not None,row
        wanted=parse_insn(0,row['text'],int(row['word'],16));actual=parse_insn(0,text.replace('+-','-'),int(row['word'],16))
        assert key(wanted)==key(actual)and wanted.guard==actual.guard,(row['text'],text,key(wanted),key(actual),wanted.guard,actual.guard)
        for left,right in zip(wanted.ops,actual.ops):
            if isinstance(left,Mem):
                lb=left.base.idx if left.base is not None else 255;rb=right.base.idx if right.base is not None else 255
                assert lb==rb and(left.off&0xffffff)==(right.off&0xffffff)
                if lb!=255:assert left.scale==right.scale
            else:assert left==right,(row['text'],text,left,right)
        row.update(disassembled=text,passed=True)
    (HERE/'field-sweep.json').write_text(json.dumps(dict(schema=1,passed=True,results=records,
        template_sha256=hashlib.sha256((HERE/'candidate-templates.json').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Recovered-template register, full signed offset, scale and predicate disassembly sweep; includes deliberately unaligned non-executed words, no hardware scheduling or memory semantics claim'),indent=2)+'\n')
    print('Passed',len(records),'varied native words')
