"""Integer rational IEEE rounding and retained ptxas-SASS execution controls."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from openassembler.emu.machine import Launch,Machine,Memory
from openassembler.emu.program import load_listing


def rounded_bits(value,width,negative_zero=False):
    exponent_bits,fraction_bits=(8,23)if width==32 else(11,52)
    sign=int(value<0 or(value==0 and negative_zero));value=abs(value)
    if not value:return sign<<(width-1)
    bias=(1<<(exponent_bits-1))-1;emin=1-bias;emax=bias
    exponent=value.numerator.bit_length()-value.denominator.bit_length()
    power=Fraction(1<<exponent)if exponent>=0 else Fraction(1,1<<-exponent)
    if value<power:exponent-=1
    quantum=max(exponent-fraction_bits,emin-fraction_bits)
    scaled=value/(Fraction(1<<quantum)if quantum>=0 else Fraction(1,1<<-quantum))
    significand,remainder=divmod(scaled.numerator,scaled.denominator)
    if 2*remainder>scaled.denominator or(2*remainder==scaled.denominator and significand%2):significand+=1
    if not significand:return sign<<(width-1)
    if significand>=1<<(fraction_bits+1):significand>>=1;quantum+=1
    actual_exponent=quantum+significand.bit_length()-1
    if actual_exponent>emax:return(sign<<(width-1))|(((1<<exponent_bits)-1)<<fraction_bits)
    field=actual_exponent+bias if actual_exponent>=emin else 0
    fraction=significand-(1<<fraction_bits)if field else significand
    return(sign<<(width-1))|(field<<fraction_bits)|fraction


def rational(bits,width):
    exponent_bits,fraction_bits=(8,23)if width==32 else(11,52)
    exponent=(bits>>fraction_bits)&((1<<exponent_bits)-1);fraction=bits&((1<<fraction_bits)-1)
    assert exponent!=(1<<exponent_bits)-1
    bias=(1<<(exponent_bits-1))-1
    value=Fraction(fraction+(1<<fraction_bits)if exponent else fraction)
    power=(exponent-bias if exponent else 1-bias)-fraction_bits
    value*=Fraction(1<<power)if power>=0 else Fraction(1,1<<-power)
    return-value if bits>>(width-1)else value


if __name__=='__main__':
    oracle=json.loads((HERE/'oracle-matrix.json').read_text());records=[]
    for row in oracle['results']:
        if not row['accepted']:continue
        source=(HERE/'oracle'/(row['name']+'.ptx')).read_text()
        literal=row['spelling']if row['name'].startswith('f32_')and row['name']!='f32_raw64'else None
        if literal and not literal.startswith('0d'):
            value=Fraction(literal);bits64=rounded_bits(value,64,literal.startswith('-'))
        elif row['name']in('f32_raw64','f32_13'):bits64=int('3ff8000000000000',16)
        elif row['name']in('f64_decimal','b64_decimal'):bits64=int('3ff8000000000000',16)
        else:continue
        width=64 if row['name']in('f64_decimal','b64_decimal')else 32
        expected=bits64 if width==64 else rounded_bits(rational(bits64,64),32,bool(bits64>>63))
        listing=(HERE/'oracle'/(row['name']+'.sass')).read_text()
        memory=Memory();out=0x100000000;guard=b'\xa5'*64;memory.write_bytes(out-64,guard+bytes(width//8)+guard)
        try:
            Machine(load_listing(listing),Launch(block=(1,1,1),params=struct.pack('<Q',out)),memory).run()
            actual=memory.read(out,width//8)
            assert actual==expected,(row['name'],hex(actual),hex(expected))
            assert bytes(memory.read(out-64+i,1)for i in range(64))==guard
            assert bytes(memory.read(out+width//8+i,1)for i in range(64))==guard
            record=dict(name=row['name'],passed=True,width=width,parsed_binary64=hex(bits64),expected_bits=hex(expected),actual_bits=hex(actual))
        except Exception as error:record=dict(name=row['name'],passed=False,error=type(error).__name__+': '+str(error),expected_bits=hex(expected))
        records.append(record)
    tie=Fraction('1.0000000596046448');staged=rounded_bits(rational(rounded_bits(tie,64),64),32);direct=rounded_bits(tie,32)
    assert staged==0x3f800000 and direct==0x3f800001
    result=dict(schema=1,passed=all(row['passed']for row in records),results=records,
                two_stage_counterexample=dict(literal=str(tie),staged_bits=hex(staged),direct_bits=hex(direct)),
                oracle_matrix_sha256=hashlib.sha256((HERE/'oracle-matrix.json').read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Exact rational rounding and CPU execution of retained ptxas SASS; no production implementation or native GPU numerical gate')
    (HERE/'rounding-probe.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
