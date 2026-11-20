"""Retain an unrelated exact-division emulator counterexample; never waive it."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import diff_emu
from openassembler.emu.probe import run_probe


if __name__=='__main__':
    source=Path(sys.argv[1]);listing=Path(sys.argv[2]);output=Path(sys.argv[3])
    inputs=[(0x8147ec48,0x21dea823,0x2bae9e99,0xc5bae70c)]
    expected=diff_emu.run_ptx(source.read_text(),'p_div_rn_f32',inputs,0,0)
    actual=run_probe(listing.read_text(),inputs,0,0)
    result={'scope':'SASS emulator vs PTX interpreter, same frozen corpus; not a new hardware observation',
            'inputs':[[hex(x) for x in row] for row in inputs],'actual':actual,'expected':expected,
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'sass_sha256':hashlib.sha256(listing.read_bytes()).hexdigest(),
            'matches':diff_emu.same('div.rn.f32',actual[0],expected[0][0],False)}
    output.write_text(json.dumps(result,indent=2)+'\n');print(output,'matches',result['matches'])
