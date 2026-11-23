"""Independent linearized CPU atomic reference for Python->PTX->native path."""
from pathlib import Path
import hashlib,json,struct,sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from openassembler import build
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from openassembler.ptx.parse import parse_module
from openassembler.pyfront import compile_file,NotSupported
from bench_build import listing


def cases():
    coords=[-2**31,-1,0,1,2,3,4,5,6,7,8,2**31-1,-3,10,-2,9];s=[10]*8;out=[]
    for index in coords:
        if 0<=index<8:out.append(s[index]);s[index]+=2
        else:out.append(0)
    yield{'name':'checked','arrays':{'state':[10]*8,'coords':coords,'returned':[-999]*16},'expected':{'state':s,'coords':coords,'returned':out},'scalar':{},'threads':16}
    for name,value in(('checked_exchange',7),('checked_compare_exchange',9)):
        result=[10 if 0<=index<8 else 0 for index in coords]
        yield{'name':name,'arrays':{'state':[10]*8,'coords':coords,'returned':[-999]*16},'expected':{'state':[value]*8,'coords':coords,'returned':result},'scalar':{},'threads':16}
    yield{'name':'evaluation_order','arrays':{'state':[1,5,9],'out':[99]},'expected':{'state':[0,12,10],'out':[7]},'scalar':{},'threads':32}
    for name in('add','exchange','compare_exchange'):
        n=16;state=[0];returned=list(range(n))if name=='add'else([0]+[i for i in range(1,n)])if name=='exchange'else([0]+[1]*(n-1))
        arrays={'state':state,'returned':[-999]*32};expected={'state':[16 if name=='add'else 16 if name=='exchange'else 1],'returned':returned+[-999]*16}
        if name=='add':arrays={'values':[1]*32,**arrays};expected={'values':[1]*32,**expected}
        yield{'name':name,'arrays':arrays,'expected':expected,'scalar':{'n':n},'threads':32}


def run(compiler):
    ptx=compiler(HERE/'fixture.py');module=parse_module(ptx);records=[]
    for case in cases():
        addresses={name:0x100000000+i*8192+256 for i,name in enumerate(case['arrays'])}
        def fresh():
            memory=Memory()
            for name,values in case['arrays'].items():memory.write_bytes(addresses[name]-256,b'\xa5'*256+struct.pack('<'+str(len(values))+'I',*(x&0xffffffff for x in values))+b'\xa5'*256)
            return memory
        def read(memory):
            result={}
            for name,values in case['arrays'].items():
                raw=bytes(memory.read(addresses[name]+i,1)for i in range(4*len(values)));result[name]=list(struct.unpack('<'+str(len(values))+'I',raw))
                for off in(-256,len(raw)):assert bytes(memory.read(addresses[name]+off+i,1)for i in range(256))==b'\xa5'*256
            return result
        expected={name:[x&0xffffffff for x in values]for name,values in case['expected'].items()};blob=b''
        for kind,name in module.kernels[case['name']].params:blob+=struct.pack('<Q'if kind=='u64'else'<I',addresses.get(name,case['scalar'].get(name)))
        rows=[]
        for level in('safe','mined','sched','opt'):
            code,labels,words,_,regs=build.generate(module.kernels[case['name']],'sm_80',level)
            for poison in(None,0xdeadbeef):
                memory=fresh();Machine(load_listing(listing(code,labels)),Launch(block=(case['threads'],1,1),params=blob,param_base=0x160,poison=poison),memory).run();actual=read(memory);assert actual==expected,(case['name'],level,actual,expected)
            rows.append({'level':level,'registers':regs,'native_sha256':hashlib.sha256(b''.join(w.to_bytes(16,'little')for w in words)).hexdigest(),'passed':True})
        records.append({**case,'native':rows,'passed':True})
    return{'passed':True,'ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest(),'cases':records,'hardware':'not_run','performance':'not_measured'}

if __name__=='__main__':
    if '--production'in sys.argv:result={'production':run(lambda p:compile_file(p)[0])};target=HERE/'production-result.json'
    else:
        try:compile_file(HERE/'fixture.py')
        except NotSupported as exc:baseline={'status':'rejected','reason':str(exc)}
        else:baseline={'status':'accepted'}
        from install import install
        install();from openassembler.pyfront import compile_file as candidate
        result={'baseline':baseline,'candidate':run(lambda p:candidate(p)[0])};target=HERE/'candidate-result.json'
    target.write_text(json.dumps(result,indent=2)+'\n');print(target)
