"""Checked shape/stride/index/device and host binding contracts."""
import hashlib,json,struct
from pathlib import Path
import pytest
from openassembler import build,runner
from openassembler.pyfront import compile_file,compile_kernel,find_kernel,NotSupported,Buffer,i32,i64
from openassembler.pyfront.buffers import contract_from_ptx
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import run_kernel,Launch as PL
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing

ROOT=Path(__file__).resolve().parent
CASES=json.loads((ROOT/'data/pyfront_buffers.json').read_text());FORMAT={'i32':'i','i64':'q','f32':'f'}

@pytest.mark.parametrize('case',CASES,ids=[c['name']for c in CASES])
@pytest.mark.parametrize('level',['safe','mined','sched','opt'])
def test_checked_accesses_strides_holes_and_extreme_indices(case,level):
    ptx=compile_kernel(find_kernel(ROOT/'data/pyfront_buffers.py',case['name']));module=parse_module(ptx)
    addresses={name:0x100000000+8192*i+256 for i,name in enumerate(case['arrays'])}
    def fresh():
        memory=Memory()
        for name,(kind,values)in case['arrays'].items():memory.write_bytes(addresses[name]-256,b'\xa5'*256+struct.pack('<'+str(len(values))+FORMAT[kind],*values)+b'\xa5'*256)
        return memory
    def read(memory):
        result={}
        for name,(kind,values)in case['arrays'].items():
            size=struct.calcsize(FORMAT[kind])*len(values);raw=bytes(memory.read(addresses[name]+i,1)for i in range(size));result[name]=list(struct.unpack('<'+str(len(values))+FORMAT[kind],raw))
            for offset in(-256,size):assert bytes(memory.read(addresses[name]+offset+i,1)for i in range(256))==b'\xa5'*256
        return result
    memory=fresh();run_kernel(module,case['name'],PL(block=(case['threads'],1,1),params=addresses),memory);assert read(memory)==case['expected']
    code,labels,*_=build.generate(module.kernels[case['name']],'sm_80',level)
    for poison in(None,0xdeadbeef):
        memory=fresh();Machine(load_listing(listing(code,labels)),Launch(block=(case['threads'],1,1),params=b''.join(struct.pack('<Q',addresses[n])for _,n in module.kernels[case['name']].params),param_base=0x160,poison=poison),memory).run();assert read(memory)==case['expected']

@pytest.mark.parametrize('annotation,diagnostic',[
 ('Buffer(i32,(0,4))','positive int32'),('Buffer(i32,(3,4),strides=(0,1))','may overlap'),
 ('Buffer(i64,(3,),alignment=4)','alignment'),('Buffer(i32,(3,),alignment=12)','alignment'),
 ('Buffer(i32,(2,),strides=(-1,))','literal constants'),('Buffer(i32,(2,),strides=(2147483647,))','byte span'),
 ('Buffer(i32,(2,3),strides=(1,))','matching nonnegative'),('Buffer(i32,(2,),access="write")','access'),
 ('Buffer(i32,(2,),unknown=1)','unsupported'),('Buffer(i32,(1,2,3,4,5))','one to four'),
])
def test_invalid_views_reject(tmp_path,annotation,diagnostic):
    p=tmp_path/'bad.py';p.write_text(f'@kernel\ndef k(a:{annotation}):\n    pass\n')
    with pytest.raises(NotSupported,match=diagnostic):compile_file(p)

@pytest.mark.parametrize('body,diagnostic',[
 ('out[0] = a[0,1]','requires 1 scalar'),('out[0] = a[i64(1)]','int32 expressions'),
 ('a[0] = 1','read-only'),('out[0] += 1','augmented stores'),
])
def test_invalid_accesses_reject(tmp_path,body,diagnostic):
    p=tmp_path/'bad.py';p.write_text(f'@kernel\ndef k(a:Buffer(i32,(2,),access="read"),out:Buffer(i32,(2,))):\n    {body}\n')
    with pytest.raises(NotSupported,match=diagnostic):compile_file(p)


def bindings(tmp_path):
    ptx=compile_kernel(find_kernel(ROOT/'data/pyfront_buffers.py','checked_wide'));kernel=parse_module(ptx).kernels['checked_wide'];contract=contract_from_ptx(ptx,kernel)
    (tmp_path/'source.ptx').write_text(ptx)
    target=runner.Target({'ptx':'source.ptx','native_ownership':{'ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest()}},
                         {'name':'checked_wide','params':kernel.params,'buffer_contract':contract},tmp_path/'k.py')
    return target

@pytest.mark.parametrize('bad', ['short','unaligned','alias','raw_pointer','tampered'])
def test_host_bindings_fail_closed_before_launch(tmp_path,bad):
    target=bindings(tmp_path);buffers={'x':64,'y':64,'coords':64,'loads':128};args=[('buf','x',0),('buf','y',0),('buf','coords',0),('buf','loads',0)]
    if bad=='short':buffers['x']=63
    elif bad=='unaligned':buffers['x']=128;args[0]=('buf','x',8)
    elif bad=='alias':args[1]=('buf','x',0)
    elif bad=='raw_pointer':args[0]=('u64',0x1234)
    else:(tmp_path/'source.ptx').write_text('tampered')
    with pytest.raises(runner.RunError,match='checked Buffer'):
        runner._checked_buffer_bindings(target,tmp_path,args,buffers)


def test_host_nonoverlap_views_and_readonly_broadcast_are_valid(tmp_path):
    target=bindings(tmp_path);buffers={'storage':128,'coords':64,'loads':128};args=[('buf','storage',0),('buf','storage',64),('buf','coords',0),('buf','loads',0)]
    runner._checked_buffer_bindings(target,tmp_path,args,buffers)
    assert Buffer(i32,(3,4),strides=(0,1),access='read').storage_elements==4


def test_buffer_metadata_and_verifier_fail_closed(tmp_path):
    target=bindings(tmp_path);source=(tmp_path/'source.ptx').read_text();kernel=parse_module(source).kernels['checked_wide']
    from openassembler.verify import parameter_contract_from_ptx
    with pytest.raises(ValueError,match='Buffer|buffer'):parameter_contract_from_ptx(source,kernel)
    marker=next(line for line in source.splitlines()if line.startswith('// openassembler-pyfront-buffer-v1 '))
    with pytest.raises(ValueError,match='duplicate'):contract_from_ptx(source+'\n'+marker,kernel)


def test_removing_buffer_contract_breaks_native_ownership():
    from openassembler import oracle
    from openassembler.native_ownership import validate
    if not oracle.available('ptxas'):pytest.skip('requires CUDA ptxas')
    result=build.build_source(ROOT/'data/pyfront_buffers.py','python','buffer',build.Options(no_sass=True),single=True)
    assert result.rc==0,result.error
    entry=result.entry(True);binary=result.files[entry['cubin']]
    validate(entry,binary)
    entry['kernels'][0].pop('buffer_contract')
    with pytest.raises(ValueError,match='inventory differs'):validate(entry,binary)
