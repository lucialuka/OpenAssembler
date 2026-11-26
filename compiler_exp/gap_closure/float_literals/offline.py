"""Exact rational controls for decimal parse, use rounding and raw conversions."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import struct
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from rounding_probe import rounded_bits,rational
from openassembler.ptx.parse import parse_operand,parse_instr,ImmF,parse_module
from openassembler import build


def independent32(bits):
    sign=(bits>>63)<<31;exponent=bits>>52&2047;fraction=bits&((1<<52)-1)
    if exponent==2047:return sign|0x7f800000|((fraction>>29)|0x400000 if fraction else 0)
    return rounded_bits(rational(bits,64),32,bool(bits>>63))


if __name__=='__main__':
    records=[];rng=random.Random(8127)
    literals=['0.0','-0.0','+1.5','.5','2.','1e-3','1E+3','0e9999','-0e-9999','1.0000000596046448',
              '1.0000001788139343','3.4028235677973366e38','1.401298464324817e-45','7.006492321624085e-46',
              '2.2250738585072014e-308','1.7976931348623157e308']
    literals.extend(f'{rng.randrange(-10**17,10**17)}.{rng.randrange(10**12):012d}e{rng.randrange(-300,289):+d}'for _ in range(5000))
    for text in literals:
        expected64=rounded_bits(Fraction(text),64,text.startswith('-'))
        value=parse_operand(text);assert isinstance(value,ImmF)and value.width==64 and value.bits==expected64
        value32=parse_instr('mov.f32 %value,'+text,1).args[1]
        assert value32==ImmF(independent32(expected64),32),(text,value32,hex(expected64))
    records.append(dict(kind='independent-rational-decimal-and-use-rounding',cases=len(literals),passed=True))
    words=[0,1,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff0000000000001,0x7ffabcdef1234567,0xfff8000000000000]
    words.extend(rng.getrandbits(64)for _ in range(5000))
    for bits in words:
        actual=parse_instr(f'mov.f32 %value,0d{bits:016x}',1).args[1]
        assert actual==ImmF(independent32(bits),32),(hex(bits),actual)
        assert parse_instr(f'mov.f64 %value,0d{bits:016x}',1).args[1]==ImmF(bits,64)
    records.append(dict(kind='raw64-payload-use-conversion-and-preservation',cases=len(words),passed=True))
    for bits in(0,1,0x3fc00000,0x80000000,0x7fc12345,0x7f800001,0xffcabcde):
        assert parse_instr(f'mov.f32 %value,0f{bits:08x}',1).args[1]==ImmF(bits,32)
        assert parse_instr(f'mov.f64 %value,0f{bits:08x}',1).args[1]==ImmF(bits,64)
    records.append(dict(kind='raw32-exact-preservation-and-wide-move-zero-extension',cases=7,passed=True))
    reports=('oracle-matrix.json','context-oracle.json','range-oracle.json','widen-oracle.json','integer-context-oracle.json','store64-oracle.json')
    for report_name in reports:
        report=json.loads((HERE/report_name).read_text());folder={'oracle-matrix.json':'oracle','context-oracle.json':'contexts','range-oracle.json':'ranges','widen-oracle.json':'widen','integer-context-oracle.json':'integer_contexts','store64-oracle.json':'store64'}[report_name]
        for row in report['results']:
            path=HERE/folder/(row['name']+'.ptx')
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row['source_sha256']
            try:module=parse_module(path.read_text());parsed=True
            except SyntaxError:parsed=False
            assert parsed==row['accepted'],(row['name'],parsed,row['accepted'])
            if parsed:
                for kernel in module.kernels.values():
                    if row['name']in('sqrt_rn_f32','rcp_rn_f32'):
                        # Exact backend modifier support is checked separately;
                        # syntax acceptance alone does not widen that whitelist.
                        continue
                    build.generate(kernel,'sm_80','safe')
        records.append(dict(kind='retained-toolkit-contexts',report=report_name,cases=len(report['results']),passed=True))
    result=dict(schema=1,passed=True,results=records,source_sha256=hashlib.sha256((HERE/'candidate-parse.py').read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Independent rational decimal/binary rounding, raw payloads, preserved/mismatched context rejection and ordinary code generation; no GPU proof')
    (HERE/'candidate-offline.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
