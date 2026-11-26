"""Locate actual CUDA-emitted scalar boundaries after byte-parameter admission."""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from openassembler.ptx.parse import parse_module
from openassembler import build
from openassembler.backend.ra import NotSupported


if __name__=='__main__':
    prior=HERE.parent/'python_boolean_abi'
    path=prior/'baseline-toolkit.json'
    report=json.loads(path.read_text());assert report['passed']
    assert all(hashlib.sha256((prior/name).read_bytes()).hexdigest()==digest for name,digest in report['artifact_hashes'].items())
    source=prior/'source.ptx'
    module=parse_module(source.read_text())
    rows=[]
    for name,kernel in module.kernels.items():
        try:build.generate(kernel,'sm_80','safe')
        except NotSupported as error:
            assert 'setp.'in str(error)and'.s16'in str(error)
            rows.append(dict(entry=name,rejected=True,stage='scalar contract',diagnostic=str(error),parameters=kernel.params))
        else:raise AssertionError(name)
    inputs=('openassembler/backend/contracts.py','openassembler/backend/ra.py','openassembler/ptx/parse.py','openassembler/ptx/semantics.py')
    data=dict(schema=1,passed=True,production_fix=False,results=rows,
              original_toolkit_record=str(path.relative_to(ROOT)),original_toolkit_record_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              original_artifact_hashes={str((prior/name).relative_to(ROOT)):digest for name,digest in report['artifact_hashes'].items()},
              source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()for name in inputs},
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              scope='Exact independent original nvcc/ptxas Boolean scalar/array controls now pass entry parsing but reject at actual setp.s16 scalar admission. No source adaptation, new compilation, numerical execution, GPU claim or generic CUDA support claim')
    (HERE/'baseline.json').write_text(json.dumps(data,indent=2)+'\n')
    print('Two original CUDA entries now reach the actual setp.s16 scalar-contract boundary')
