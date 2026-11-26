"""Compare exact installed rowwise output before changing compiler identities."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from openassembler import build,rowwise_dispatch as rd
from openassembler.ptx.parse import parse_module
from openassembler.kernels.reductions import ReductionPlan,generate

def sha(data):return hashlib.sha256(data).hexdigest()

def outputs():
    result=[]
    for entry in json.loads(rd.TABLE.read_text())['entries']:
        c=entry['contract']
        plan=ReductionPlan(rows=c['rows'],width=c['width'],op=c['op'],epsilon=c['epsilon_f32'],arch=c['arch'],warps_per_cta=entry['warps_per_cta'])
        source=generate(plan);kernel=next(iter(parse_module(source).kernels.values()))
        code,labels,words,exits,registers=build.generate(kernel,'sm_80','opt')
        result.append(dict(contract=c,policy=entry['warps_per_cta'],words=words,code=[(t,vars(ctl))for t,ctl in code],labels=labels,exits=exits,registers=registers))
    return result
if __name__=='__main__':
    before=outputs()
    from candidate import install
    install();after=outputs()
    rows=[dict(contract=a['contract'],policy=a['policy'],passed=a==b,before_words_sha256=sha(b''.join(w.to_bytes(16,'little')for w in a['words'])),after_words_sha256=sha(b''.join(w.to_bytes(16,'little')for w in b['words'])))for a,b in zip(before,after)]
    (HERE/'rowwise-policy-probe.json').write_text(json.dumps(dict(schema=1,passed=all(r['passed']for r in rows),results=rows),indent=2)+'\n');print(rows)
