"""Source-only baseline and complete independent shared-wide byte executions."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(HERE)]
from fixtures import cases,values,expected,delta,SENTINEL

def sha(data):return hashlib.sha256(data).hexdigest()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',action='store_true');args=parser.parse_args()
    if args.candidate:
        from candidate import install
        install()
    from openassembler.pyfront import compile_file
    from openassembler import build
    from openassembler.ptx.parse import parse_module
    from openassembler.ptx.interp import Interp,Launch as PLaunch
    from openassembler.emu.machine import Memory,Machine,Launch
    from openassembler.emu.program import load_listing
    from tests.backend_harness import listing
    rows=[];lanes=128;inp,out=0x100010000,0x100000000;guard=b'\xa5'*64
    with tempfile.TemporaryDirectory(prefix='oa-python-shared-')as temporary:
        for case in cases():
            path=Path(temporary)/(case['name']+'.py');path.write_text(case['source'])
            try:ptx,names=compile_file(path)
            except Exception as error:
                rows.append(dict(name=case['name'],status='rejected',error=type(error).__name__+': '+str(error)));continue
            assert args.candidate,case
            assert names==[case['name']]
            module=parse_module(ptx);kernel=module.kernels[case['name']]
            from openassembler.backend.ra import Selector
            assert Selector(kernel).smem_bytes==case['shared_bytes']
            for level in('safe','mined','sched','opt'):
                code,labels,words,_,registers=build.generate(kernel,'sm_80',level)
                assert words
                for seed in(14301,14302):
                    data=values(seed,lanes);before=guard+struct.pack('<'+str(lanes)+'Q',*data)+guard
                    d=delta(case,seed)
                    for n in(0,91,128):
                        wanted=guard+expected(data,case,n,d)+guard
                        for engine in('ptx','normal','poisoned'):
                            memory=Memory();memory.write_bytes(inp-64,before);memory.write_bytes(out-64,guard+struct.pack('<Q',SENTINEL)*(lanes*4)+guard)
                            if engine=='ptx':Interp(module,case['name'],PLaunch(block=(128,1,1),params={'out':out,'a':inp,'delta':d,'n':n}),memory).run()
                            else:Machine(load_listing(listing(code,labels)),Launch(block=(128,1,1),params=struct.pack('<QQQI',inp,out,d&((1<<64)-1),n),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
                            actual=bytes(memory.read(out-64+i,1)for i in range(len(wanted)));assert actual==wanted,(case['name'],level,n,seed,engine,next((i,a,b)for i,(a,b)in enumerate(zip(actual,wanted))if a!=b))
                            assert bytes(memory.read(inp-64+i,1)for i in range(len(before)))==before
                rows.append(dict(name=case['name'],level=level,status='passed',executions=18,registers=registers,shared_bytes=case['shared_bytes'],source_sha256=sha(case['source'].encode()),ptx_sha256=sha(ptx.encode())))
                print(rows[-1],flush=True)
    result=dict(schema=1,passed=all(row['status']==('passed'if args.candidate else'rejected')for row in rows),results=rows,
                frontend_sha256=sha((HERE/('candidate-compile.py'if args.candidate else'baseline-compile.py')).read_bytes()),fixtures_sha256=sha((HERE/'fixtures.py').read_bytes()),
                scope='Independent complete modulo64 references, guards/inputs/tails and PTX/normal/poisoned SASS; no hardware or performance claim')
    result['backend_hashes']={name:sha((HERE/('candidate-'+stem+'.py')).read_bytes()if args.candidate else(ROOT/name).read_bytes())for name,stem in [('openassembler/backend/ra.py','ra'),('openassembler/backend/compile.py','backend-compile')]}
    (HERE/('candidate-offline.json'if args.candidate else'baseline.json')).write_text(json.dumps(result,indent=2)+'\n');print('Passed',result['passed'],len(rows))
