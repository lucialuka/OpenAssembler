"""Original entry, load, physical-layout and device-call boundaries."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE)]
from candidate import install
install()
from fixtures import HEADER, CASES, source
from openassembler.ptx.parse import parse_module
from openassembler import build
from openassembler.backend.ra import NotSupported

NEGATIVE = [
    ('aggregate', '.entry k(.param .align 8 .b8 v[8]) {ret;}'),
    ('float16', '.entry k(.param .f16 v) {ret;}'),
    ('predicate_slot', '.entry k(.param .pred v) {ret;}'),
    ('narrow_pointer', '.entry k(.param .u8 .ptr v) {ret;}'),
    ('array', '.entry k(.param .b8 v[4]) {ret;}'),
    ('scalar_alignment_override', '.entry k(.param .align 16 .u16 v) {ret;}'),
    ('cross_slot', '.entry k(.param .u8 v,.param .u8 other){.reg .u32 %r;ld.param.u16 %r,[v];ret;}'),
    ('negative_offset', '.entry k(.param .u16 v){.reg .u32 %r;ld.param.u8 %r,[v-1];ret;}'),
    ('out_of_slot', '.entry k(.param .u16 v){.reg .u32 %r;ld.param.u8 %r,[v+2];ret;}'),
    ('unaligned_halfword', '.entry k(.param .b32 v){.reg .u32 %r;ld.param.u16 %r,[v+1];ret;}'),
    ('undeclared', '.entry k(.param .u8 v){.reg .u32 %r;ld.param.u8 %r,[other];ret;}'),
    ('parameter_vector', '.entry k(.param .b32 v){.reg .u32 %r<3>;ld.param.v2.u8 {%r1,%r2},[v];ret;}'),
    ('parameter_store', '.entry k(.param .u8 v){.reg .u32 %r;mov.u32 %r,0;st.param.u8 [v],%r;ret;}'),
    ('narrow_device_slots', '.func (.param .u8 result) h(.param .u8 flag){ret;} .entry k(){ret;}'),
]

if __name__ == '__main__':
    rows = []
    for name, text in NEGATIVE:
        try:
            parsed = parse_module(HEADER+text)
            for kernel in parsed.kernels.values():build.generate(kernel,'sm_80','safe')
        except (SyntaxError, NotSupported, ValueError) as error:
            rows.append(dict(name=name, source=HEADER+text, rejected=True, diagnostic=str(error), exception=type(error).__name__))
        else:
            raise AssertionError(name)
    paths = [HERE/('candidate-'+suffix+'.py') for suffix in json.loads((HERE/'inputs.json').read_text())]
    paths += [Path(__file__), HERE/'inputs.json']
    data = dict(schema=1, passed=True, results=rows,
                source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                scope='Fourteen exact original aggregate/float/predicate/pointer/offset/alignment/load/store/vector/device-call ABI rejection boundaries; no numerical/GPU claim')
    (HERE/'negative-controls.json').write_text(json.dumps(data,indent=2)+'\n')
    print('Passed 14 original narrow ABI rejection controls')
