"""Independent operation-order invariants and exact allocation/resource controls."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from hardware_probes import cases
from openassembler.backend import ra
from openassembler.backend.pressure import schedule
from openassembler.backend.spill import allocate_kernel,_allocate_kernel
from openassembler.ptx.parse import parse_module


def check_order(original,ordered):
    assert len(original)==len(ordered)and{id(op)for op in original}=={id(op)for op in ordered}
    position={id(op):index for index,op in enumerate(ordered)}
    reads=[ra._all_uses(op)|(ra._defs(op)if op.guard is not None else set())for op in original]
    writes=[ra._defs(op)for op in original]
    loads=[]
    for i,op in enumerate(original):
        opcode=op.fmt.split()[0].split('.')[0]if op.fmt else ''
        if opcode in('LDG','LDS','LDL','LDC'):loads.append(position[id(op)])
        if op.side or op.label or op.target or opcode in('BAR','MEMBAR','ERRBAR','CCTL','DEPBAR','S2R','ULDC','SHFL'):
            assert position[id(op)]==i,('moved fixed boundary',i,op.fmt)
        for j in range(i+1,len(original)):
            if writes[i]&reads[j]or reads[i]&writes[j]or writes[i]&writes[j]:
                assert position[id(op)]<position[id(original[j])],('register/predicate dependence',i,j)
    assert loads==sorted(loads),'changed load order'


def transfers(ops):
    result=Counter();block=0
    for op in ops:
        if op.label:block+=1
        opcode=op.fmt.split()[0].split('.')[0]if op.fmt else ''
        if opcode in('LDL','STL'):result[block,opcode]+=1
        if op.target or opcode in('EXIT','RET','BRA','BRX','JMP','CALL'):block+=1
    return result


if __name__=='__main__':
    records=[]
    for case in cases():
        selector=ra.Selector(parse_module(case['source']).kernels[case['name']]);ops=ra.coalesce(ra.dce(selector.run()))
        snapshot=[asdict(op)for op in ops];ordered=schedule(ops);check_order(ops,ordered)
        assert snapshot==[asdict(op)for op in ops]
        try:check_order(ops,list(reversed(ordered)))
        except AssertionError:pass
        else:raise AssertionError('corrupt order accepted')
        baseline=_allocate_kernel(selector,ops);candidate=allocate_kernel(selector,ops)
        before,after=transfers(baseline[0]),transfers(candidate[0])
        assert candidate[2]['spill_slots']<=baseline[2]['spill_slots']
        assert candidate[2]['local_bytes']<=baseline[2]['local_bytes']
        assert all(count<=before[key]for key,count in after.items())
        if candidate[2]['spill_slots']==baseline[2]['spill_slots']:
            assert candidate==baseline,'equal-cost plan changed the prior allocation'
        records.append(dict(name=case['name'],domain=case['domain'],passed=True,
                            reordered=sum(a is not b for a,b in zip(ops,ordered)),before=baseline[2],after=candidate[2],
                            before_transfers={str(key):value for key,value in before.items()},
                            after_transfers={str(key):value for key,value in after.items()},corrupt_order_rejected=True))
        print(records[-1],flush=True)
    result=dict(schema=1,passed=True,results=records,source_hashes={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()for name in('pressure.py','spill.py','controls.py')},
                scope='Independent pairwise RAW/WAR/WAW/guard and fixed-boundary/load-order constraints, unchanged operation identities, real allocations and non-increased local transfers per basic block; no hardware or speed claim')
    (HERE/'final-controls.json').write_text(json.dumps(result,indent=2)+'\n')
