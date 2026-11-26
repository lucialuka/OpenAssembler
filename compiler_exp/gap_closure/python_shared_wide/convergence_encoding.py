"""Custody and varied disassembly for the existing B0 convergence encoding."""
import hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from openassembler import oracle
from openassembler.isa.encode import Encoder
from openassembler.isa.word import Control,control

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
    corpus=ROOT/'corpus/sm_80/pairs.jsonl';seeds={}
    for line in corpus.open():
        row=json.loads(line);name=row['text'].split()[0]
        if name in('BSSY','BSYNC')and name not in seeds:seeds[name]=row
    assert set(seeds)=={'BSSY','BSYNC'}
    enc=Encoder('sm_80');words=[];rows=[]
    for wait in(0,1,21,63):
        for distance in(16,32,48,256,4096,65536,1048576):
            for name in('BSSY','BSYNC'):
                address=16*len(words);ctl=Control(6,False,7,7,wait,0)
                word=enc.encode_branch('BSSY B0, `(join)',address,address+distance,ctl)if name=='BSSY'else enc.encode('BSYNC B0',ctl)
                rows.append(dict(op=name,address=address,target=address+distance if name=='BSSY'else None,word=f'{word:032x}',wait=wait));words.append(word)
                assert control(word)==ctl
    actual=oracle.disasm_words(words)
    for row,text in zip(rows,actual):
        assert text is not None
        if row['op']=='BSSY':
            match=re.fullmatch(r'BSSY B0, (0x[0-9a-f]+)',text);assert match and int(match[1],16)==row['target'],(row,text)
        else:assert text=='BSYNC B0',text
        row.update(disassembled=text,passed=True)
    result=dict(schema=1,passed=True,results=rows,seeds=seeds,corpus_sha256=sha(corpus.read_bytes()),encoder_sha256=sha((ROOT/'openassembler/isa/encode.py').read_bytes()),script_sha256=sha(Path(__file__).read_bytes()),
                scope='Existing real-corpus B0 BSSY/BSYNC words, exact controls and varied resolved forward targets; no new guessed fields, token indices, scheduling or general convergence proof')
    (HERE/'convergence-encoding.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',len(rows),'convergence words')
