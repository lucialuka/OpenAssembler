"""Retain original narrow entry/form rejections after validated scalar ABI promotion."""
import builtins
import hashlib
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
STAGE = HERE.parent/'narrow_parameters'
sys.path.insert(0,str(ROOT))
for name,suffix in [('openassembler.ptx.parse','parse'),('openassembler.backend.contracts','contracts')]:
    module = importlib.import_module(name)
    path = STAGE/('baseline-'+suffix+'.py')
    exec(builtins.compile(path.read_bytes(),str(path),'exec'),module.__dict__)
from openassembler.ptx.parse import parse_module, Instr, Var, Addr
from openassembler.backend.contracts import validate_scalar,ContractError


if __name__ == '__main__':
    rows = []
    for path in (HERE/'narrow.ptx',HERE/'parameter_layout/source.ptx'):
        try:parse_module(path.read_text())
        except SyntaxError as error:
            assert 'scalar 32/64-bit parameters required' in str(error)
            rows.append(dict(path=str(path.relative_to(HERE)),rejected=True,diagnostic=str(error)))
        else:raise AssertionError(path)
    for dtype in ('u8','s8','b8','u16','s16','b16'):
        try:validate_scalar(Instr('ld',('param',dtype),[Var('%r1'),Addr('value')],None))
        except ContractError as error:rows.append(dict(dtype=dtype,rejected=True,diagnostic=str(error)))
        else:raise AssertionError(dtype)
    paths = [HERE/'narrow.ptx',HERE/'parameter_layout/source.ptx',STAGE/'baseline-parse.py',STAGE/'baseline-contracts.py',Path(__file__)]
    records = [HERE/'narrow-baseline.json',HERE/'parameter_layout/baseline-layout.json',HERE/'parameter_layout/layout-audit.json']
    data = dict(schema=1,passed=True,results=rows,
                source_hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                original_records_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in records},
                scope='Two original parser and six scalar-form rejections replayed with exact pre-narrow-ABI parser/contracts; historical toolkit/numerical records unchanged. Current narrow ABI is separately admitted and hardware validated')
    (HERE/'parameter-boundary-replay.json').write_text(json.dumps(data,indent=2)+'\n')
    print('Replayed two original parser and six original scalar-load-form boundaries')
