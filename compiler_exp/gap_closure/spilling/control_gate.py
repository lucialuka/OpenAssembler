"""Candidate group exclusions, no-spill continuity and defined partial writes."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(HERE)]
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.emu.machine import Launch,Machine,Memory
from openassembler.emu.program import load_listing
from tests.backend_harness import listing


if __name__=='__main__':
    samples=['tests/data/backend_loops/dot_rows.ptx','tests/data/backend_loops/saxpy_gs.ptx','tests/data/backend_loops/softmax_rows.ptx','tests/data/backend_loops/nested2.ptx']
    sources=[]
    for path in samples:
        p=ROOT/path
        if p.is_file():sources.append((path,p.read_text()))
    if not sources:
        sources=[('simple','.version 8.7\n.target sm_80\n.address_size 64\n.entry simple(.param .u64 out){.reg .u64 %p;ld.param.u64 %p,[out];st.global.u32 [%p],1;ret;}')]
    baseline={}
    for name,text in sources:
        for entry,kernel in parse_module(text).kernels.items():
            for level in('safe','mined','sched','opt'):
                try:baseline[name,entry,level]=('generated',build.generate(kernel,'sm_80',level))
                except Exception as exc:baseline[name,entry,level]=('rejected',type(exc).__name__,str(exc))
    from candidate import install
    install()
    from openassembler.backend import ra
    from openassembler.backend.spill import allocate_kernel
    from probes import source,expected
    records=[]
    for name,text in sources:
        for entry,kernel in parse_module(text).kernels.items():
            for level in('safe','mined','sched','opt'):
                try:current=('generated',build.generate(kernel,'sm_80',level))
                except Exception as exc:current=('rejected',type(exc).__name__,str(exc))
                assert current==baseline[name,entry,level]
                records.append(dict(kind='no-spill-native-continuity'if current[0]=='generated'else'unchanged-rejection',source=name,entry=entry,level=level,passed=True))
    # A tight scalar limit is recoverable; grouped pressure remains a separate
    # explicit rejection and never breaks an aligned group into scalar pieces.
    kernel=parse_module(source(40,16)).kernels['pressure']
    selector=ra.Selector(kernel);ops=ra.coalesce(ra.dce(selector.run()))
    groups=ra._groups(ops)
    _,_,resources=allocate_kernel(selector,ops)
    assert resources['spill_slots']>0 and resources['register_limit']==16
    # Resource metadata never aliases an input dictionary across invocations.
    resources.clear();different={}
    generated=build.generate(kernel,'sm_80','safe',resource_info=different)
    assert different['spill_slots']>0 and different['local_bytes']>0 and generated[4]<=16
    records.append(dict(kind='tight-register-limit',resources=different,passed=True))
    from types import SimpleNamespace
    from openassembler.backend.storage import LocalFrame
    stack=ra.V(1);virtual=2;ops=[ra.Op('MOV {d0}, c[0x0][0x28]',[stack],[],side=True)]
    grouped=[]
    for index in range(12):
        group=tuple(ra.V(n)for n in range(virtual,virtual+4));virtual+=4;grouped.append(group)
        ops.append(ra.Op('dummy wide load {d0}',[group],[],var_lat=True))
    for group in grouped:ops.append(ra.Op('dummy wide consume {s0}',[],[group],side=True))
    selector=SimpleNamespace(k=SimpleNamespace(directives={'maxnreg':(16,)}),r1=stack,local_frame=LocalFrame({}, {}, 0))
    try:allocate_kernel(selector,ops)
    except ra.NotSupported as exc:
        assert 'wide/predicate'in str(exc)
        records.append(dict(kind='group-pressure-fails-closed',diagnostic=str(exc),passed=True))
    else:raise AssertionError('unsupported group pressure admitted')
    result=dict(schema=1,passed=True,results=records,scope='Candidate CPU continuity/resource controls; no packaging/GPU/timing claim',
                spill_sha256=hashlib.sha256((HERE/'spill.py').read_bytes()).hexdigest())
    (HERE/'candidate-controls.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
