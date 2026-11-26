"""Next G11 cycle: full-bit wide selection with alias/false-guard/source snapshots."""
import argparse,hashlib,json,random,struct,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
HEADER='.version 8.7\n.target sm_80\n.address_size 64\n'
SENTINEL=0x13579bdf2468ace0

def cases():
    for dtype in('b64','u64','s64','f64'):
        for mode in('register','immediate','aliased','false_guard'):
            a='%a';b='%b';dest='%r';initial='mov.b64 %r,0xdeadbeef13579bdf;'
            if mode=='immediate':b='0dfff8123456789abc'if dtype=='f64'else'0xdeadbeef87654321'
            if mode=='aliased':dest='%a';initial=''
            instruction=f'selp.{dtype} {dest},{a},{b},%take;'
            if mode=='false_guard':instruction='@%writer '+instruction
            source=HEADER+f'''.entry probe(.param .u64 out,.param .u64 input,.param .u32 n){{
.reg .u64 %out,%in,%off,%src,%dst,%a,%b,%r;.reg .u32 %i,%n,%lo;.reg .pred %take,%writer,%active;
ld.param.u64 %out,[out];ld.param.u64 %in,[input];ld.param.u32 %n,[n];mov.u32 %i,%tid.x;
mul.wide.u32 %off,%i,16;add.u64 %src,%in,%off;ld.global.b64 %a,[%src];ld.global.b64 %b,[%src+8];
and.b32 %lo,%i,1;setp.ne.u32 %take,%lo,0;and.b32 %lo,%i,2;setp.ne.u32 %writer,%lo,0;
{initial}{instruction}
'''+('mov.b64 %b,0;'if mode=='aliased'else'mov.b64 %a,0;mov.b64 %b,0;')+f'''
mul.wide.u32 %off,%i,8;add.u64 %dst,%out,%off;setp.lt.u32 %active,%i,%n;@%active st.global.b64 [%dst],{dest};ret;}}'''
            yield dict(dtype=dtype,mode=mode,source=source)

def data(seed):
    rng=random.Random(seed);edge=[0,(1<<64)-1,1<<63,0x7ff0000000000000,0xfff0000000000000,0x7ff0123456789abc,0xfff8123456789abc]
    return[edge[i]if i<len(edge)else rng.getrandbits(64)for i in range(256)]

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',action='store_true');args=parser.parse_args()
    if args.candidate:
        from candidate import install
        install()
    from openassembler import build
    from openassembler.ptx.parse import parse_module
    from openassembler.ptx.interp import Interp,Launch as PL
    from openassembler.emu.machine import Machine,Memory,Launch
    from openassembler.emu.program import load_listing
    from tests.backend_harness import listing
    rows=[];inp,out=0x100010000,0x100000000;guard=b'\xa5'*64
    for case in cases():
        try:
            module=parse_module(case['source']);kernel=module.kernels['probe']
            for level in('safe','mined','sched','opt'):
                code,labels,words,_,registers=build.generate(kernel,'sm_80',level);assert args.candidate
                for seed in(27391,27392):
                    values=data(seed);before=guard+struct.pack('<256Q',*values)+guard
                    for n in(0,91,128):
                        expected=[SENTINEL]*128
                        for i in range(n):
                            left,right=values[2*i:2*i+2]
                            if case['mode']=='immediate':right=0xfff8123456789abc if case['dtype']=='f64'else 0xdeadbeef87654321
                            expected[i]=(left if i&1 else right)if case['mode']!='false_guard'or i&2 else 0xdeadbeef13579bdf
                        wanted=guard+struct.pack('<128Q',*expected)+guard
                        for engine in('ptx','normal','poisoned'):
                            memory=Memory();memory.write_bytes(inp-64,before);memory.write_bytes(out-64,guard+struct.pack('<Q',SENTINEL)*128+guard)
                            if engine=='ptx':Interp(module,'probe',PL(block=(128,1,1),params={'out':out,'input':inp,'n':n}),memory).run()
                            else:Machine(load_listing(listing(code,labels)),Launch(block=(128,1,1),params=struct.pack('<QQI',out,inp,n),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
                            assert bytes(memory.read(out-64+i,1)for i in range(len(wanted)))==wanted,(case['dtype'],case['mode'],level,n,seed,engine)
                            assert bytes(memory.read(inp-64+i,1)for i in range(len(before)))==before
                rows.append(dict(dtype=case['dtype'],mode=case['mode'],level=level,passed=True,executions=18))
        except Exception as error:rows.append(dict(dtype=case['dtype'],mode=case['mode'],passed=False,error=type(error).__name__+': '+str(error)))
    result=dict(schema=1,passed=all(r['passed']for r in rows)if args.candidate else not any(r['passed']for r in rows),results=rows,scope='Independent full-bit scalar64 select CPU/normal/poisoned SASS hypotheses with exact input/output guards; no toolkit/hardware or production claim',source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in(HERE/'candidate-ra.py',HERE/'candidate-contracts.py',Path(__file__))})
    (HERE/('candidate-offline.json'if args.candidate else'baseline.json')).write_text(json.dumps(result,indent=2)+'\n');print(result['passed'],len(rows))
