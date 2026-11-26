"""Deterministic independent layout/extraction algebra across arbitrary scalar mixes."""
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(HERE)]
from openassembler.ptx.parse import parameter_layout


if __name__=='__main__':
    rng=random.Random(78319)
    types=('b8','u8','s8','b16','u16','s16','b32','u32','s32','f32','b64','u64','s64','f64')
    rows=[]
    for index in range(4096):
        fields=[(rng.choice(types),'p'+str(i))for i in range(rng.randint(1,32))]
        packed=bytearray();expected={};values={}
        for dtype,name in fields:
            size=int(dtype[1:])//8
            while len(packed)%size:packed.append(0xa5)
            expected[name]=(len(packed),size)
            values[name]=rng.getrandbits(size*8)
            packed.extend(values[name].to_bytes(size,'little'))
        assert parameter_layout(fields)==expected
        checks=[]
        for dtype,name in fields:
            start,size=expected[name]
            for width in (1,2):
                if width > size:continue
                for offset in range(0,size,width):
                    raw=int.from_bytes(packed[start+offset:start+offset+width],'little')
                    absolute=start+offset
                    aligned=absolute&~3
                    block=packed[aligned:aligned+4]+bytes(max(0,aligned+4-len(packed)))
                    word=int.from_bytes(block,'little')
                    shift=(absolute&3)*8;bits=width*8
                    unsigned=(word>>shift)&((1<<bits)-1)
                    signed=raw-(1<<bits)if raw&(1<<(bits-1))else raw
                    left=(word<<(32-bits-shift))&0xffffffff
                    arithmetic=left-(1<<32)if left&(1<<31)else left
                    arithmetic>>=32-bits
                    assert raw==unsigned and signed==arithmetic
                    checks.append([name,offset,width,unsigned,signed])
        rows.append(dict(index=index,parameters=fields,packed_hex=packed.hex(),layout={name:list(v)for name,v in expected.items()},extractions=checks,passed=True))
    report=dict(schema=1,passed=True,seed=78319,results=rows,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                parser_sha256=hashlib.sha256((ROOT/'openassembler/ptx/parse.py').read_bytes()).hexdigest(),
                scope='4096 deterministic mixed scalar layouts with independently packed padding/bytes, exact offsets and every naturally aligned byte/halfword extraction; signed/zero extension algebra independently checked. CPU layout/algebra only, no additional native/hardware or universal ABI claim')
    (HERE/'layout-properties.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Passed 4096 independent mixed layouts and',sum(len(row['extractions'])for row in rows),'byte/halfword extraction controls')
