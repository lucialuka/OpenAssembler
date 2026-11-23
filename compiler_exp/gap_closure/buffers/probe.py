"""Independent bounded shape/stride/index reference vs checked PTX and all native levels."""
from pathlib import Path
import hashlib,json,struct,sys,platform
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from openassembler import build
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from openassembler.ptx.interp import run_kernel,Launch as PTXLaunch
from openassembler.ptx.parse import parse_module
from openassembler.pyfront import compile_file,NotSupported
from bench_build import listing
FORMAT={'f32':'f','i32':'i','i64':'q'}


def cases():
    coords=[-1,0,0,-1,0,0,0,3,1,2,2,0,2,3,3,0,0,4,2**31-1,0,-2**31,1,1,1,1,3,2,2,0,2,2,1]
    a=[float(i) for i in range(16)];o=[-999.0]*16
    for row,col in zip(coords[::2],coords[1::2]):
        if 0<=row<3 and 0<=col<4:o[row*6+col]=a[row*6+col]+.5
    yield {'name':'transpose','arrays':{'a':('f32',a),'out':('f32',[-999.0]*16),'coords':('i32',coords)},'expected':{'a':a,'out':o,'coords':coords},'threads':16}
    coords=[-2**31,-1,0,1,2,3,4,5,6,7,8,2**31-1,0,7,-20,20];a=[-(1<<63),-1,0,1,1<<32,(1<<63)-1,99,-33];o=[-999]*8;loads=[]
    for index in coords:
        value=a[index] if 0<=index<8 else 0;loads.append(value)
        if 0<=index<8:
            value=(value+1)&((1<<64)-1);o[index]=value-(1<<64) if value&(1<<63) else value
    yield {'name':'checked_wide','arrays':{'a':('i64',a),'out':('i64',[-999]*8),'coords':('i32',coords),'loads':('i64',[-999]*16)},'expected':{'a':a,'out':o,'coords':coords,'loads':loads},'threads':16}
    a=[11,22,33,44];o=[11,22,33,44]*3
    yield {'name':'broadcast','arrays':{'a':('i32',a),'out':('i32',[-999]*12)},'expected':{'a':a,'out':o},'threads':32}


def run(compiler):
    source=(HERE/'fixture.py').read_bytes();ptx=compiler(HERE/'fixture.py');module=parse_module(ptx);records=[]
    for case in cases():
        addresses={name:0x100000000+8192*i+256 for i,name in enumerate(case['arrays'])}
        def fresh():
            memory=Memory()
            for name,(kind,values) in case['arrays'].items():memory.write_bytes(addresses[name]-256,b'\xa5'*256+struct.pack('<'+str(len(values))+FORMAT[kind],*values)+b'\xa5'*256)
            return memory
        def read(memory):
            actual={}
            for name,(kind,values) in case['arrays'].items():
                size=struct.calcsize(FORMAT[kind])*len(values);data=bytes(memory.read(addresses[name]+i,1) for i in range(size))
                actual[name]=list(struct.unpack('<'+str(len(values))+FORMAT[kind],data))
                for off in (-256,size):assert bytes(memory.read(addresses[name]+off+i,1) for i in range(256))==b'\xa5'*256,(name,'guard',off)
            return actual
        memory=fresh();run_kernel(module,case['name'],PTXLaunch(block=(case['threads'],1,1),params=addresses),memory);assert read(memory)==case['expected'],(case['name'],'PTX',read(memory),case['expected'])
        blob=b''.join(struct.pack('<Q',addresses[name]) for _,name in module.kernels[case['name']].params);variants=[]
        for level in ('safe','mined','sched','opt'):
            code,labels,words,_,regs=build.generate(module.kernels[case['name']],'sm_80',level)
            for poison in (None,0xdeadbeef):
                memory=fresh();Machine(load_listing(listing(code,labels)),Launch(block=(case['threads'],1,1),params=blob,param_base=0x160,poison=poison),memory).run()
                assert read(memory)==case['expected'],(case['name'],level,poison,read(memory),case['expected'])
            variants.append({'level':level,'native_sha256':hashlib.sha256(b''.join(w.to_bytes(16,'little')for w in words)).hexdigest(),'registers':regs,'passed':True})
        records.append({'name':case['name'],'arrays':case['arrays'],'expected':case['expected'],'native':variants,'passed':True})
    return {'passed':True,'python':platform.python_version(),'source_sha256':hashlib.sha256(source).hexdigest(),'ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest(),'cases':records,'hardware':'not_run','performance':'not_measured'}

if __name__=='__main__':
    if '--production' in sys.argv:result={'production':run(lambda p:compile_file(p)[0])};target=HERE/'production-result.json'
    else:
        try:compile_file(HERE/'fixture.py')
        except NotSupported as exc:baseline={'status':'rejected','reason':str(exc)}
        else:baseline={'status':'accepted'}
        from install import install
        install();from openassembler.pyfront import compile_file as candidate
        result={'baseline':baseline,'candidate':run(lambda p:candidate(p)[0])};target=HERE/'candidate-result.json'
    target.write_text(json.dumps(result,indent=2)+'\n');print(target)
