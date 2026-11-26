"""CPU observation of complete constant bits from retained ptxas listings."""
import hashlib
import json
from pathlib import Path
import struct
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from openassembler.emu.machine import Launch,Machine,Memory
from openassembler.emu.program import load_listing


if __name__=='__main__':
    report=json.loads((HERE/'edge-oracle.json').read_text());rows=[];guard=b'\xa5'*64;out=0x100000000
    for row in report['results']:
        if not row['accepted']:continue
        path=HERE/'edges'/(row['name']+'.sass');text=path.read_text();assert hashlib.sha256(text.encode()).hexdigest()==row['sass_sha256']
        try:
            observed=[]
            for poison in(None,0xdeadbeef):
                memory=Memory();memory.write_bytes(out-64,guard+bytes(8)+guard)
                Machine(load_listing(text),Launch(block=(1,1,1),params=struct.pack('<Q',out),poison=poison),memory).run()
                assert bytes(memory.read(out-64+i,1)for i in range(64))==guard
                assert bytes(memory.read(out+8+i,1)for i in range(64))==guard
                observed.append(memory.read(out,8))
            assert observed[0]==observed[1]
            result=dict(row,passed=True,observed_bits=f'{observed[0]:016x}',observed_float=repr(struct.unpack('<d',struct.pack('<Q',observed[0]))[0]))
        except Exception as error:result=dict(row,passed=False,error=type(error).__name__+': '+str(error))
        rows.append(result);print(row['expression'],result.get('observed_bits'),result.get('observed_float'),result.get('error'),flush=True)
    result=dict(schema=1,passed=all(row['passed']for row in rows),results=rows,
                oracle_report_sha256=hashlib.sha256((HERE/'edge-oracle.json').read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='CPU normal/poisoned execution of retained ptxas listings exposes folded constant bits with guards; independent semantics and GPU gates remain separate')
    (HERE/'edge-values.json').write_text(json.dumps(result,indent=2)+'\n')
