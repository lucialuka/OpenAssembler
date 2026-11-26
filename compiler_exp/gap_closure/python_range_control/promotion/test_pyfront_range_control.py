"""Original pure helper return control, independent bits, source and ownership contracts."""
import json,struct
from pathlib import Path
import pytest
from openassembler import build,oracle
from openassembler.backend import ra
from openassembler.pyfront import compile_file,compile_kernel,find_kernel,NotSupported
from openassembler.pyfront.modules import contract_from_ptx,validate_sources,PREFIX
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing
DATA=json.loads((Path(__file__).parent/'range_control_fixtures.json').read_text())
HEADER='from typing import Final\nfrom openassembler.pyfront import *\n'

def write(root,case):
 for name,text in case['files'].items():
  path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
 return root/case['entry']

@pytest.mark.parametrize('case',DATA['configurations'],ids=lambda c:c['name'])
@pytest.mark.parametrize('name',list(DATA['values']))
@pytest.mark.parametrize('level',['safe','opt'])
def test_finite_range_overflow_loop_else_and_original_source_identity(case,name,level,tmp_path):
 source=write(tmp_path,case);text,names=compile_file(source);graph=contract_from_ptx(text);validate_sources(source,graph)
 assert{r['path']for r in graph['files']}==set(case['files'])and graph['helpers']
 single=compile_kernel(find_kernel(source,name));single_graph=contract_from_ptx(single)
 assert all(set(row['kernels'])=={name}for row in single_graph['helpers'].values())
 module=parse_module(text);code,labels,words,_,registers=build.generate(module.kernels[name],'sm_80',level)
 _,_,single_words,_,single_regs=build.generate(parse_module(single).kernels[name],'sm_80',level);assert words==single_words and registers==single_regs
 data=DATA['values'][name];before,initial,wanted=(bytes.fromhex(data[key])for key in('input_hex','initial_hex','expected_hex'))
 ap,out=0x100010000,0x100000000;guard=b'\xa5'*64
 for engine in('ptx','normal','poisoned'):
  memory=Memory();memory.write_bytes(ap-64,guard+before+guard);memory.write_bytes(out-64,guard+initial+guard)
  if engine=='ptx':Interp(module,name,PLaunch(block=(128,1,1),params={'a':ap,'out':out,'n':91}),memory).run()
  else:Machine(load_listing(listing(code,labels)),Launch(block=(128,1,1),params=struct.pack('<QQI',ap,out,91),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
  assert bytes(memory.read(out-64+i,1)for i in range(len(wanted)+128))==guard+wanted+guard
  assert bytes(memory.read(ap-64+i,1)for i in range(len(before)+128))==guard+before+guard

@pytest.mark.parametrize('case',DATA['negative_controls'],ids=lambda c:c['name'])
def test_loop_range_else_domains_initialization_and_convergence_reject(case,tmp_path):
 source=tmp_path/(case['name']+'.py');source.write_text(case['source'])
 with pytest.raises((NotSupported,ra.NotSupported),match=case['reason']):
  text,_=compile_file(source)
  for kernel in parse_module(text).kernels.values():build.generate(kernel,'sm_80','safe')

@pytest.mark.parametrize('body,wanted',[
 ('    value=0\n    for i in range(0,3,1):\n        value+=i\n    else:\n        value+=7\n    out[0]=i64(value)\n',10),
 ('    value=0\n    for i in range(3):\n        for j in range(0):\n            pass\n        else:\n            if i==1:\n                continue\n        value+=i\n    else:\n        value+=7\n    out[0]=i64(value)\n',9),
 ('    value=0\n    i=0\n    while i<3:\n        i+=1\n        for j in range(0):\n            pass\n        else:\n            break\n    else:\n        value=7\n    out[0]=i64(value+i)\n',1),
 ('    value=0\n    for i in range(3):\n        j=0\n        while j<0:\n            j+=1\n        else:\n            break\n    else:\n        value=7\n    out[0]=i64(value)\n',0),
 ('    for i in range(2147483646,2147483647,2):\n        return\n    else:\n        out[0]=i64(7)\n',-991),
])
@pytest.mark.parametrize('level',['safe','opt'])
def test_kernel_return_unit_step_and_nearest_loop_else_targets(body,wanted,level,tmp_path):
 source=tmp_path/'loop.py';source.write_text(HEADER+'@kernel\ndef k(out:Array(i64)):\n'+body);text,_=compile_file(source);module=parse_module(text);code,labels,words,_,regs=build.generate(module.kernels['k'],'sm_80',level);out=0x100000000
 for engine in('ptx','normal','poisoned'):
  memory=Memory();memory.write_bytes(out,struct.pack('<q',-991))
  if engine=='ptx':Interp(module,'k',PLaunch(params={'out':out}),memory).run(max_steps=5000)
  else:Machine(load_listing(listing(code,labels)),Launch(params=struct.pack('<Q',out),poison=0xdeadbeef if engine=='poisoned'else None),memory).run(max_steps=5000)
  assert bytes(memory.read(out+i,1)for i in range(8))==struct.pack('<q',wanted)

@pytest.mark.parametrize('loop',[
 'for i in range(3):\n        pass\n    else:',
 'i=0\n    while i<3:\n        i+=1\n    else:',
])
def test_uniform_runtime_loop_else_barrier_remains_admitted(loop,tmp_path):
 source=tmp_path/'uniform.py';source.write_text(HEADER+'@kernel\ndef k(out:Array(i32)):\n    '+loop+'\n        syncthreads()\n    out[tid_x()]=7\n');text,_=compile_file(source);assert 'bar.sync'in text

def test_returned_else_value_definite_initialization_is_path_sensitive(tmp_path):
 source=tmp_path/'live.py';source.write_text(HEADER+'''@device
def h(x:i32)->i32:
    for i in range(x):
        if x:
            return x
    else:
        value=7
    return value
@kernel
def k(out:Array(i32)):
    out[0]=h(0)
''');text,_=compile_file(source);module=parse_module(text);memory=Memory();out=0x100000000;memory.write_bytes(out,bytes(4));Interp(module,'k',PLaunch(params={'out':out}),memory).run();assert memory.read(out,4)==7

@pytest.mark.oracle
def test_normal_build_retains_loop_control_source_and_native_ownership(tmp_path):
 if not oracle.available('ptxas'):pytest.skip('needs CUDA toolkit')
 source=write(tmp_path,DATA['configurations'][0]);out=tmp_path/'output';assert build.compile_paths(source,out,build.Options(no_sass=True)).rc==0
 entry,=json.loads((out/'manifest.json').read_text())['entries'];binary=(out/entry['cubin']).read_bytes()
 from openassembler.native_ownership import validate
 validate(entry,binary);next(iter(entry['kernels'][0]['python_source_graph']['helpers'].values()))['body_sha256']='0'*64
 with pytest.raises(ValueError,match='inventory'):validate(entry,binary)

