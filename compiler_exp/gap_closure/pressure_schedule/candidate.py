"""Isolate the pressure pass and allocation wrapper from production."""
from pathlib import Path
import importlib.util
import sys

HERE=Path(__file__).resolve().parent


def install(*,frozen=False):
    for leaf in('pressure','spill'):
        name='openassembler.backend.'+leaf
        path=HERE/('hardware-spill.py'if frozen and leaf=='spill'else leaf+'.py')
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)


if __name__=='__main__':
    source=(HERE/'baseline-spill.py').read_text().replace('def allocate_kernel(selector,ops):','def _allocate_kernel(selector,ops):',1)
    source+='''

def allocate_kernel(selector,ops):
    from openassembler.backend import ra
    from openassembler.backend.pressure import schedule
    from collections import Counter
    def transfers(result):
        counts=Counter();block=0
        for op in result[0]:
            if op.label:block+=1
            opcode=op.fmt.split()[0].split('.')[0]if op.fmt else ''
            if opcode in('LDL','STL'):counts[block,opcode]+=1
            if op.target or opcode in('EXIT','RET','BRA','BRX','JMP','CALL'):block+=1
        return counts
    failure=None
    try:
        baseline=_allocate_kernel(selector,ops)
        if not baseline[2]['spill_slots']:return baseline
    except ra.NotSupported as error:
        if not str(error).startswith(('register pressure exceeds','predicate pressure exceeds','spill frame exceeds')):raise
        failure=error
    ordered=schedule(ops)
    if any(a is not b for a,b in zip(ops,ordered)):
        try:
            candidate=_allocate_kernel(selector,ordered)
            if failure is not None:return candidate
            before,after=transfers(baseline),transfers(candidate)
            if(candidate[2]['spill_slots']<baseline[2]['spill_slots']and candidate[2]['local_bytes']<=baseline[2]['local_bytes']
               and all(count<=before[key]for key,count in after.items())):return candidate
        except ra.NotSupported:pass
    if failure is not None:raise failure
    return baseline
'''
    (HERE/'spill.py').write_text(source)
