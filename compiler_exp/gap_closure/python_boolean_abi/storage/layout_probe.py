"""Record exact existing byte primitives and unsupported Boolean storage width rules."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from openassembler.backend.contracts import scalar_forms
from openassembler.pyfront.types import Scalar
from openassembler.pyfront.buffers import Buffer

if __name__=='__main__':
    forms=scalar_forms()
    memory=[]
    for operation in('ld','st'):
        for space in('global','shared'):
            for dtype in('u8','b8'):
                name=space+'.'+dtype
                memory.append(dict(operation=operation,modifiers=name,declared=(operation,tuple(name.split('.')))in forms))
    rows=[]
    for logical_type in(Scalar('pred'),Scalar('bool')):
        try:Buffer(logical_type,(4,7),access='readwrite')
        except ValueError as error:rows.append(dict(logical_kind=logical_type.ptx,rejected=True,diagnostic=str(error)))
        else:raise AssertionError(logical_type)
    # Public element layouts are distinct from the internal predicate-register
    # kind. Do not construct a synthetic accepted Bool Buffer to bypass the gate.
    inputs=('openassembler/pyfront/buffers.py','openassembler/pyfront/compile.py','openassembler/verify.py','openassembler/runner.py','openassembler/backend/contracts.py')
    result=dict(schema=1,passed=True,production_fix=False,buffer_rejections=rows,memory_forms=memory,
                source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()for name in inputs},
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Read-only scalar memory-form inventory and actual predicate/Boolean shaped-buffer constructor rejection. Declared byte memory forms are existing primitives, not evidence of Boolean source storage semantics, numerical/hardware execution or supported Buffer ABI')
    (HERE/'layout-baseline.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Recorded byte-form inventory and two actual logical Boolean shaped-buffer rejections')
