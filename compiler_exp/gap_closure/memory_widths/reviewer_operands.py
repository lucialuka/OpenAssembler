"""Retained pre-repair f64 store immediate diagnostic proof against ptxas."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(HERE)]
from cases import source
from openassembler import build
import openassembler.backend as backend
from openassembler.ptx.parse import parse_module

if __name__=='__main__':
    raw=(HERE/'reviewer-before-ra.py').read_bytes()
    module=types.ModuleType('openassembler.backend.ra');module.__file__=str(HERE/'reviewer-before-ra.py');module.__package__='openassembler.backend'
    sys.modules[module.__name__]=module;backend.ra=module;exec(compile(raw,module.__file__,'exec'),module.__dict__)
    ptx=source('st.global.f64 [%dst],0d3FF0000000000000;')
    expected='3ff0000000000000'
    binary=build.ptxas_build(ptx,'sm_80')
    try:build.generate(parse_module(ptx).kernels['probe'],'sm_80','safe')
    except Exception as exc:result={'status':'failed','exception':type(exc).__name__,'message':str(exc)}
    else:raise AssertionError('Expected the retained pre-repair operand diagnostic failure')
    report={'scope':'Legal ptxas operand admission and pre-repair OA diagnostic failure; no hardware execution',
            'before_ra_sha256':hashlib.sha256(raw).hexdigest(),'ptx':ptx,'expected_store_bits_u64_hex':expected,
            'ptxas':{'status':'accepted','cubin_bytes':len(binary),'cubin_sha256':hashlib.sha256(binary).hexdigest()},'openassembler':result}
    target=HERE/'reviewer-f64-immediate-negative.json'
    if target.exists():raise FileExistsError('Preserve original review evidence')
    target.write_text(json.dumps(report,indent=2)+'\n');print(target)
