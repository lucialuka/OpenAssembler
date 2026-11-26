"""Candidate/baseline direct-call selection and independent complete byte gates."""
import argparse,hashlib,json,struct,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from fixtures import expanded_cases as cases,values,expected,SENTINEL

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',action='store_true');args=parser.parse_args()
    if args.candidate:
        from candidate import install
        install()
    from openassembler import build
    from openassembler.ptx.parse import parse_module
    from openassembler.ptx.interp import Interp,Launch as PLaunch
    from openassembler.emu.machine import Machine,Memory,Launch
    from openassembler.emu.program import load_listing
    from tests.backend_harness import listing
    rows=[];guard=b'\xa5'*64;inp,out=0x100010000,0x100000000;lanes=128
    for case in cases():
        try:
            module=parse_module(case['source']);kernel=module.kernels[case['entry']]
            for level in('safe','mined','sched','opt'):
                code,labels,words,_,registers=build.generate(kernel,'sm_80',level)
                if not args.candidate:raise AssertionError('baseline unexpectedly admitted '+case['name'])
                for seed in(17341,17342):
                    data=values(seed,lanes,case['mode']);before=guard+struct.pack('<128Q',*data)+guard
                    for n in(0,91,128):
                        wanted=guard+expected(data,case,n)+guard
                        for engine in('ptx','normal','poisoned'):
                            memory=Memory();memory.write_bytes(inp-64,before);memory.write_bytes(out-64,guard+struct.pack('<Q',SENTINEL)*lanes+guard)
                            if engine=='ptx':Interp(module,case['entry'],PLaunch(block=(128,1,1),params={kernel.params[0][1]:out,kernel.params[1][1]:inp,kernel.params[2][1]:n}),memory).run()
                            else:Machine(load_listing(listing(code,labels)),Launch(block=(128,1,1),params=struct.pack('<QQI',out,inp,n),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
                            actual=bytes(memory.read(out-64+i,1)for i in range(len(wanted)));assert actual==wanted,(case['name'],level,n,seed,engine,next((i,a,b)for i,(a,b)in enumerate(zip(actual,wanted))if a!=b))
                            assert bytes(memory.read(inp-64+i,1)for i in range(len(before)))==before
                rows.append(dict(name=case['name'],level=level,status='passed',executions=18,registers=registers,calls=module.inlined_calls,source_sha256=sha(case['source'].encode())))
                print(rows[-1],flush=True)
        except Exception as error:
            rows.append(dict(name=case['name'],status='rejected'if not args.candidate else'failed',error=type(error).__name__+': '+str(error)));print(rows[-1],flush=True)
    result=dict(schema=1,passed=all(r['status']==('passed'if args.candidate else'rejected')for r in rows),results=rows,source_hashes={name:sha((HERE/name).read_bytes())for name in('candidate-parse.py','inlining.py','fixtures.py','functions.ptx')},scope='Source-defined direct scalar calls, independent complete buffers/guards/input and functional models; no hardware or native call-stack proof')
    (HERE/('candidate-offline.json'if args.candidate else'baseline.json')).write_text(json.dumps(result,indent=2)+'\n');print('Passed',result['passed'],len(rows))
