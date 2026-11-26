"""Exact scalar executable continuity and bounded impossible-resource controls."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(HERE)]
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.backend import ra
from openassembler.backend.storage import LocalFrame


def generated(case,level):
    resources={}
    try:result=build.generate(parse_module(case['source']).kernels[case['name']],'sm_80',level,resource_info=resources)
    except ra.NotSupported as exc:return('rejected',type(exc).__name__,str(exc))
    return 'generated',result,resources


if __name__=='__main__':
    cases=json.loads((ROOT/'tests/data/spill_fixtures.json').read_text())
    ordinary=['dot_rows','saxpy_gs','softmax_rows','nested2']
    for name in ordinary:
        path=ROOT/f'tests/data/backend_loops/{name}.ptx'
        if path.is_file():
            for entry in parse_module(path.read_text()).kernels:
                cases.append(dict(name=entry,source=path.read_text()))
    baseline={(case['name'],level):generated(case,level)for case in cases for level in('safe','mined','sched','opt')}
    from candidate import install
    install()
    from openassembler.backend.spill import allocate_kernel
    records=[]
    for case in cases:
        for level in('safe','mined','sched','opt'):
            assert generated(case,level)==baseline[case['name'],level]
            records.append(dict(kind='exact-prior-native-and-resources',name=case['name'],level=level,passed=True))
    stack=ra.V(1);prologue=ra.Op('MOV {d0}, c[0x0][0x28]',[stack],[],side=True)
    selector=SimpleNamespace(k=SimpleNamespace(directives={'maxnreg':(16,)}),r1=stack,local_frame=LocalFrame({}, {}, 0))
    groups=[tuple(ra.V(2+4*i+j)for j in range(4))for i in range(4)]
    impossible=[prologue]+[ra.Op('define {d0}',[group])for group in groups]+[ra.Op('consume',[],groups,side=True)]
    for kind,ops,diagnostic in(
        ('simultaneous-wide-operands',impossible,'register pressure exceeds bounded'),
        ('simultaneous-predicates',[prologue]+[ra.Op('define',[ra.PV(i)])for i in range(1,9)]+[ra.Op('consume',[],[(ra.PV(i),False)for i in range(1,9)],side=True)],'predicate pressure exceeds bounded'),
    ):
        try:allocate_kernel(selector,ops)
        except ra.NotSupported as exc:assert diagnostic in str(exc);records.append(dict(kind=kind,diagnostic=str(exc),passed=True))
        else:raise AssertionError(kind+' incorrectly accepted')
    selector.local_frame=LocalFrame({}, {}, 65536)
    try:allocate_kernel(selector,impossible)
    except ra.NotSupported as exc:assert 'frame exceeds'in str(exc);records.append(dict(kind='combined-frame-bound',diagnostic=str(exc),passed=True))
    else:raise AssertionError('oversized frame accepted')
    from probes import cases as pressure_cases
    for case in pressure_cases():
        s=ra.Selector(parse_module(case['source']).kernels[case['name']]);ops=ra.coalesce(ra.dce(s.run()))
        rewritten,phys,resources=allocate_kernel(s,ops)
        for group in set(ra._groups(rewritten).values()):
            assert len(group)in(1,2,4)
            assert [phys[v]for v in group]==list(range(phys[group[0]],phys[group[0]]+len(group)))
            assert phys[group[0]]%len(group)==0
        assert len(set(ra.allocate_preds(rewritten).values()))<=7
        records.append(dict(kind='whole-group-alignment-and-predicate-budget',name=case['name'],resources=resources,passed=True))
    result=dict(schema=1,passed=True,results=records,allocator_sha256=hashlib.sha256((HERE/'spill.py').read_bytes()).hexdigest())
    (HERE/'candidate-controls.json').write_text(json.dumps(result,indent=2)+'\n');print('passed',len(records))
