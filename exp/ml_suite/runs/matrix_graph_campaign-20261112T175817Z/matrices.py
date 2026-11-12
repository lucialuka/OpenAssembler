"""Immutable exact-contract selector from two independent A100 kernel campaigns."""
import hashlib
import json
from pathlib import Path
from openassembler.modelfront.kernels import canonical
from openassembler.modelfront import kernels as _kernels

_ORIGINAL_EMIT_KERNEL=_kernels.emit_kernel


def selected_kernel(baseline,pointer,tiled):
    if baseline.op not in ('linear','linear_dx','linear_dw'):return baseline
    path=Path(__file__).with_name('matrix-admission.json')
    data=json.loads(path.read_text())
    # Compiler plans retain m/n/k as redundant operator metadata. The campaign
    # admitted exact operand/output shapes with no attrs; reconstruct that exact
    # canonical contract so PTX identity is stable across plan serialization.
    contract=_ORIGINAL_EMIT_KERNEL(baseline.op,[{'shape':list(s)} for s in baseline.inputs],
                                   {'shape':list(baseline.output)}, {})
    identity=hashlib.sha256(canonical(contract.signature)).hexdigest()
    matches=[r for r in data['rows'] if r['contract_sha256']==identity]
    if len(matches)!=1:return baseline
    row=matches[0]
    if row['variant']=='pointer128':candidate=pointer(contract,128)
    elif row['variant']=='pointer32':candidate=pointer(contract,32)
    elif row['variant']=='tiled':candidate=tiled(contract)
    else:return baseline
    if hashlib.sha256(candidate.ptx.encode()).hexdigest()!=row['candidate_ptx_sha256']:
        raise ValueError('measured FP32 matrix implementation identity changed')
    return candidate
