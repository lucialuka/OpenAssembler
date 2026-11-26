"""Locate the original logical/physical boundary after narrow entry admission."""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from openassembler.pyfront import compile_kernel,find_kernel,NotSupported
from openassembler.ptx.parse import parse_module
from openassembler import build


if __name__=='__main__':
    rows=[]
    for name in ('scalar_truth','scalar_helper','scalar_guard'):
        try:compile_kernel(find_kernel(HERE/'source.py',name))
        except NotSupported as error:
            assert 'unsupported type annotation' in str(error) and error.path and error.lineno
            rows.append(dict(entry=name,rejected=True,diagnostic=str(error)))
        else:raise AssertionError(name)
    # Physical byte slot admission is now independently established; the missing
    # logical contract must not be disguised as a remaining PTX parser rejection.
    control='.entry physical(.param .u8 flag,.param .u8 neighbor,.param .u64 out){.reg .u32 %r;.reg .u64 %p;ld.param.u8 %r,[flag];ld.param.u64 %p,[out];st.global.u32 [%p],%r;ret;}'
    kernel=parse_module(control).kernels['physical']
    native=[]
    for level in ('safe','mined','sched','opt'):
        _,_,words,_,registers=build.generate(kernel,'sm_80',level)
        native.append(dict(level=level,registers=registers,words_sha256=hashlib.sha256(b''.join(word.to_bytes(16,'little')for word in words)).hexdigest()))
    names=('openassembler/pyfront/compile.py','openassembler/pyfront/helpers.py','openassembler/verify.py','openassembler/runner.py','openassembler/ptx/parse.py','openassembler/native_ownership.py')
    data=dict(schema=1,passed=True,production_fix=False,results=rows,physical_byte_control=control,physical_native_controls=native,
              source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()for name in names},
              artifact_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in (HERE/'source.py',Path(__file__))},
              scope='Three original bool-annotated kernel rejections with exact source diagnostics and no user execution; physical byte slots compile at all four levels. Missing logical annotation/representation/metadata/host-binding contract remains explicit; no new GPU or numerical claim')
    (HERE/'baseline.json').write_text(json.dumps(data,indent=2)+'\n')
    print('Recorded three original logical Boolean parameter rejections; physical byte entry compiles at four levels')
