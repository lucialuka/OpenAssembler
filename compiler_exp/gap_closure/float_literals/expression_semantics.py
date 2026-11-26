"""Next-cycle independent binary64 intermediate-rounding observations."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(HERE)]
from rounding_probe import rounded_bits,rational
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing


def f64(value):return rational(rounded_bits(value,64),64)


if __name__=='__main__':
    baseline=json.loads((HERE/'expression-baseline.json').read_text())
    references={'expr_0':Fraction(-3,2),'expr_1':f64(Fraction(3,2)+Fraction(1,2)),
                'expr_2':f64(Fraction(3,2)+Fraction(1,2)),
                'expr_4':f64(Fraction(1,3)),
                'expr_7':f64(f64(f64(Fraction('1e20'))+Fraction(1))-f64(Fraction('1e20'))),
                'expr_8':f64(Fraction(1)+Fraction(1)),
                'expr_11':f64(Fraction(3,2)*Fraction(2))}
    assert set(references)=={row['name']for row in baseline['results']if row['toolkit_accepted']}
    records=[];guard=b'\xa5'*64;out=0x100000000
    for row in baseline['results']:
        if row['name']not in references:continue
        listing=HERE/'expressions'/(row['name']+'.sass')
        wanted=rounded_bits(references[row['name']],32)
        for poison in(None,0xdeadbeef):
            memory=Memory();memory.write_bytes(out-64,guard+b'\xdf\x9b\x57\x13'+guard)
            Machine(load_listing(listing.read_text()),Launch(block=(1,1,1),params=struct.pack('<Q',out),poison=poison),memory).run()
            assert bytes(memory.read(out-64+i,1)for i in range(132))==guard+struct.pack('<I',wanted)+guard
        records.append(dict(name=row['name'],expression=row['expression'],passed=True,expected_bits=hex(wanted),
                            source_sha256=row['source_sha256'],sass_sha256=hashlib.sha256(listing.read_bytes()).hexdigest()))
    assert references['expr_7']==0
    result=dict(schema=1,passed=True,results=records,executions=14,
                baseline_sha256=hashlib.sha256((HERE/'expression-baseline.json').read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Pending expression research only: exact rational binary64 intermediate rounding checked against normal/poisoned retained ptxas SASS; no expression implementation or native GPU gate')
    (HERE/'expression-semantics.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',len(records),'references',14,'CPU executions')
