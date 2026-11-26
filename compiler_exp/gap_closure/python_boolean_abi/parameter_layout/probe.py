"""Observe actual toolkit narrow scalar layout without bypassing OA admission."""
import hashlib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from openassembler import oracle
from openassembler.ptx.parse import parse_module
from openassembler.container.elf import Elf
from openassembler.container import nvinfo
from openassembler.backend.contracts import ContractError, validate_scalar
from openassembler.ptx.parse import Instr, Var, Addr


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    source, binary = HERE/'source.ptx', HERE/'reference.cubin'
    text = source.read_text()
    try:
        parse_module(text)
    except SyntaxError as error:
        assert 'scalar 32/64-bit parameters required' in str(error)
        rejection = str(error)
    else:
        raise AssertionError('baseline narrow parameter unexpectedly admitted')
    forms = []
    for dtype in ('u8', 's8', 'b8', 'u16', 's16', 'b16'):
        instruction = Instr('ld', ('param', dtype), [Var('%r1'), Addr('first')], None)
        try:
            validate_scalar(instruction)
        except ContractError as error:
            forms.append(dict(dtype=dtype, rejected=True, diagnostic=str(error)))
        else:
            raise AssertionError(dtype)
    result = oracle.run_tool('ptxas', '-arch=sm_80', '-o', oracle.tool_path(binary), oracle.tool_path(source), check=False)
    (HERE/'ptxas.log').write_text(result.stdout+result.stderr)
    assert result.returncode == 0, result.stderr
    (HERE/'reference.sass').write_text(oracle.disasm_text(binary))
    records = nvinfo.parse(Elf.parse(binary.read_bytes()).section('.nv.info.adjacent').data)
    raw = [dict(name=row.name, format=row.fmt, payload_hex=row.data.hex(),
                u32_words=list(struct.unpack('<'+'I'*(len(row.data)//4), row.data)) if len(row.data)%4 == 0 else None)
           for row in records]
    report = dict(schema=1, passed=True, production_fix=False, parser_rejection=rejection,
                  scalar_load_rejections=forms, toolkit_parameter_records=raw,
                  artifact_hashes={p.name: sha(p) for p in (source, binary, HERE/'reference.sass', HERE/'ptxas.log')},
                  source_hashes={name: sha(ROOT/name) for name in ('openassembler/ptx/parse.py', 'openassembler/backend/contracts.py', 'openassembler/backend/ra.py', 'openassembler/backend/compile.py', 'openassembler/verify.py')},
                  script_sha256=sha(Path(__file__)),
                  scope='Actual ptxas mixed adjacent 8/16/32/64-bit parameter metadata and native listing, plus explicit original OA parser/scalar-form rejection boundaries; no OA candidate, admitted miscompile, numerical execution or GPU claim')
    (HERE/'baseline-layout.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Recorded original parser/six load-form rejections and actual mixed-width toolkit metadata')
    print(json.dumps([row for row in raw if row['name'] in ('KPARAM_INFO', 'CBANK_PARAM_SIZE')], indent=2))
