"""Independent integer bit-reference vs typed PTX and all native optimization levels."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import platform
import struct
import sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from openassembler import build
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import run_kernel,Launch as PTXLaunch
from openassembler.pyfront import compile_file,NotSupported
from bench_build import listing
FORMAT={'s64':'q','u64':'Q','s32':'i','u32':'I'}
MASK=(1<<64)-1


def signed(x,bits=64):
    x&=(1<<bits)-1
    return x-(1<<bits) if x&(1<<(bits-1)) else x


def cases(n=14, delta=0):
    a=[0,1,-1,-(1<<63),(1<<63)-1,1<<32,-(1<<32),0x7fffffff,-0x80000000,12345678901234,-12345678901234,7,-9,1<<62,-(1<<62),33]
    b=[0,-1,1,(1<<63)-1,1,1,-1,0xffffffff,0x80000000,-23456789012345,23456789012345,7,9,-(1<<62),1<<62,-33]
    a=[signed(x+delta) for x in a];b=[signed(x-delta) for x in b]
    au=[x&MASK for x in a];bu=[x&MASK for x in b]
    o=[-900]*128;flags=[-900]*16;offset=-0x123456789abcdef
    for i,(x,y) in enumerate(zip(a[:n],b[:n])):
        v=x
        for _ in range(i%4):v=signed(v+y if v<y else v-y)
        o[i*8:i*8+8]=[signed(x+y),signed(x-y),signed(x&y),signed(x|y),signed(x^y),signed(~x),signed(-x+offset),v]
        flags[i]=int(x<y)+2*int(x<=y)+4*int(x>y)+8*int(x>=y)+16*int(x==y)+32*int(x!=y)+64*int(bool(x))
    yield {'name':'signed','arrays':{'a':('s64',a),'b':('s64',b),'out':('s64',[-900]*128),'flags':('s32',[-900]*16)},'scalars':{'offset':offset,'n':n},'expected':{'a':a,'b':b,'out':o,'flags':flags}}
    o=[0xdeadbeef12345678]*96;narrow=[-900]*32;offset=0xfedcba9876543210
    for i,(x,y) in enumerate(zip(au[:n],bu[:n])):
        o[i*6:i*6+6]=[(x+y+offset)&MASK,(x-y)&MASK,x&y,x|y,x^y,(~x)&MASK];narrow[i*2]=signed(x,32);narrow[i*2+1]=int(x<y)+2*int(x<=y)+4*int(x>y)+8*int(x>=y)+16*int(x==y)+32*int(x!=y)+64*int(bool(x))
    yield {'name':'unsigned','arrays':{'a':('u64',au),'b':('u64',bu),'out':('u64',[0xdeadbeef12345678]*96),'narrow':('s32',[-900]*32)},'scalars':{'offset':offset,'n':n},'expected':{'a':au,'b':bu,'out':o,'narrow':narrow}}
    o=[-900]*128;uo=[0xdeadbeef12345678]*128
    for i,(x,y) in enumerate(zip(a[:n],au[:n])):
        o[i*8:i*8+8]=[x,x>>1,x>>31,x>>32,x>>33,x>>63,signed(x<<32),signed(x<<63)]
        uo[i*8:i*8+8]=[y,y>>1,y>>31,y>>32,y>>33,y>>63,(y<<32)&MASK,(y<<63)&MASK]
    yield {'name':'shifts','arrays':{'a':('s64',a),'b':('u64',au),'out':('s64',[-900]*128),'uout':('u64',[0xdeadbeef12345678]*128)},'scalars':{'n':n},'expected':{'a':a,'b':au,'out':o,'uout':uo}}
    ai=[signed(x,32) for x in a];bi=[x&0xffffffff for x in b];o=[-900]*48;uo=[0xdeadbeef12345678]*48
    for i,(x,y) in enumerate(zip(ai[:n],bi[:n])):
        o[i*3:i*3+3]=[x,y,-1];uo[i*3:i*3+3]=[x&MASK,y,0x23456789abcdef01]
    yield {'name':'casts','arrays':{'a':('s32',ai),'b':('u32',bi),'wide':('s64',[-900]*48),'uwide':('u64',[0xdeadbeef12345678]*48)},'scalars':{'n':n},'expected':{'a':ai,'b':bi,'wide':o,'uwide':uo}}

    o=[-900]*48
    for i,(x,y) in enumerate(zip(ai[:n],a[:n])):
        o[i*3:i*3+3]=[x,signed(y,32),y]
    yield {'name':'conversion_snapshots','arrays':{'a':('s32',ai),'b':('s64',a),'out':('s64',[-900]*48)},'scalars':{'n':n},'expected':{'a':ai,'b':a,'out':o}}


def run(compiler):
    text=(HERE/'fixture.py').read_text();ptx=compiler(HERE/'fixture.py');module=parse_module(ptx);rows=[]
    for case in cases():
        kernel=module.kernels[case['name']];arrays=case['arrays'];scalars=case['scalars'];addresses={name:0x100000000+0x2000*i+256 for i,name in enumerate(arrays)}
        def fresh():
            memory=Memory()
            for name,(kind,values) in arrays.items():
                payload=struct.pack('<'+str(len(values))+FORMAT[kind],*values)
                memory.write_bytes(addresses[name]-256,b'\xa5'*256+payload+b'\xa5'*256)
            return memory
        def read(memory):
            result={}
            for name,(kind,values) in arrays.items():
                size=struct.calcsize(FORMAT[kind]);raw=bytes(memory.read(addresses[name]+i,1) for i in range(len(values)*size))
                result[name]=list(struct.unpack('<'+str(len(values))+FORMAT[kind],raw))
                for offset in (-256,len(raw)):
                    assert bytes(memory.read(addresses[name]+offset+i,1) for i in range(256))==b'\xa5'*256
            return result
        memory=fresh();run_kernel(module,case['name'],PTXLaunch(block=(16,1,1),params={**addresses,**scalars}),memory)
        interpreted=read(memory);assert interpreted==case['expected'],(case['name'],'PTX',interpreted,case['expected'])
        blob=b''
        for kind,name in kernel.params:
            alignment=8 if kind.endswith('64') else 4
            blob+=b'\0'*((-len(blob))%alignment)
            blob+=struct.pack('<'+FORMAT[kind],addresses.get(name,scalars.get(name)))
        variants=[]
        for level in ('safe','mined','sched','opt'):
            code,labels,words,_,registers=build.generate(kernel,'sm_80',level)
            memory=fresh();Machine(load_listing(listing(code,labels)),Launch(block=(16,1,1),params=blob,param_base=0x160),memory).run()
            actual=read(memory);assert actual==case['expected'],(case['name'],level,actual,case['expected'])
            variants.append({'level':level,'registers':registers,'words':len(words),'native_text_sha256':hashlib.sha256(b''.join(w.to_bytes(16,'little') for w in words)).hexdigest(),'passed':True})
        rows.append({'name':case['name'],'inputs':case['arrays'],'scalars':scalars,'expected':case['expected'],'native':variants,'passed':True})
    return {'python':platform.python_version(),'reference':'independent Python integer bit operations with explicit modulo64/signed interpretation after every operation',
            'source_sha256':hashlib.sha256(text.encode()).hexdigest(),'ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest(),
            'cases':rows,'passed':True,'hardware':'not_run','performance':'not_measured'}

if __name__=='__main__':
    if '--backend-candidate' in sys.argv:
        import importlib.util
        path=HERE.parent/'scalar_wide'/'candidate.py'
        spec=importlib.util.spec_from_file_location('backend_wide_candidate',path)
        overlay=importlib.util.module_from_spec(spec);spec.loader.exec_module(overlay);overlay.install()
    if '--production' in sys.argv:
        result={'production':run(lambda path:compile_file(path)[0])};target=HERE/'production-result.json'
    else:
        try:compile_file(HERE/'fixture.py')
        except NotSupported as exc:baseline={'status':'rejected','diagnostic':str(exc)}
        else:baseline={'status':'accepted'}
        from install import install
        install();from openassembler.pyfront import compile_file as candidate
        result={'baseline':baseline,'candidate':run(lambda path:candidate(path)[0])};target=HERE/'candidate-result.json'
    target.write_text(json.dumps(result,indent=2)+'\n');print(target)
