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
DATA=json.loads((Path(__file__).parent/'helper_return_fixtures.json').read_text())
HEADER='from typing import Final\nfrom openassembler.pyfront import *\n'

def write(root,case):
 for name,text in case['files'].items():
  path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
 return root/case['entry']

@pytest.mark.parametrize('case',DATA['configurations'],ids=lambda c:c['name'])
@pytest.mark.parametrize('name',list(DATA['values']))
@pytest.mark.parametrize('level',['safe','opt'])
def test_complete_original_helper_return_paths_and_caller_isolation(case,name,level,tmp_path):
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
def test_original_helper_return_type_completeness_and_effect_domains_reject(case,tmp_path):
 source=tmp_path/(case['name']+'.py');source.write_text(case['source'])
 with pytest.raises((NotSupported,ra.NotSupported),match=case['reason']):
  text,_=compile_file(source)
  for kernel in parse_module(text).kernels.values():build.generate(kernel,'sm_80','safe')

@pytest.mark.parametrize('dtype,fmt,values',[
 ('i32','i',[-19,-1,0,1,3,19]),('u32','I',[0,1,3,19,2147483648,4294967295]),
 ('i64','q',[-19,-1,0,1,3,19]),('u64','Q',[0,1,3,19,1<<63,(1<<64)-1]),
 ('f32','f',[-19.0,-1.0,-0.0,0.0,1.0,19.0]),
])
@pytest.mark.parametrize('level',['safe','opt'])
def test_scalar_return_types_and_branch_complete_integer_conditions(dtype,fmt,values,level,tmp_path):
 one,three=('1.0','3.0')if dtype=='f32'else(f'{dtype}(1)',f'{dtype}(3)')
 body=f'    if x:\n        return {one}\n    else:\n        return {three}\n'
 source=tmp_path/'return.py';source.write_text(HEADER+f'@device\ndef h(x:{dtype})->{dtype}:\n'+body+f'@kernel\ndef k(a:Array({dtype}),out:Array({dtype})):\n    i=tid_x()\n    out[i]=h(a[i])\n')
 text,_=compile_file(source);graph=contract_from_ptx(text);assert graph and graph['helpers'];validate_sources(source,graph)
 module=parse_module(text);code,labels,words,_,regs=build.generate(module.kernels['k'],'sm_80',level)
 before=struct.pack('<6'+fmt,*values);wanted=struct.pack('<6'+fmt,*[1 if x else 3 for x in values]);ap,out=0x100010000,0x100000000
 for engine in('ptx','normal','poisoned'):
  memory=Memory();memory.write_bytes(ap,before);memory.write_bytes(out,bytes(len(wanted)))
  if engine=='ptx':Interp(module,'k',PLaunch(block=(6,1,1),params={'a':ap,'out':out}),memory).run()
  else:Machine(load_listing(listing(code,labels)),Launch(block=(6,1,1),params=struct.pack('<QQ',ap,out),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
  assert bytes(memory.read(out+i,1)for i in range(len(wanted)))==wanted
  assert bytes(memory.read(ap+i,1)for i in range(len(before)))==before

def test_imported_return_change_refreshes_original_source_snapshot(tmp_path):
 source=write(tmp_path,DATA['configurations'][0]);ref=find_kernel(source,'classify');before=compile_kernel(ref)
 helper=tmp_path/'library/helpers.py';text=helper.read_text();assert 'return i64(17)'in text;helper.write_text(text.replace('return i64(17)','return i64(29)',1))
 after=compile_kernel(ref);assert after!=before
 with pytest.raises(ValueError,match='dependency changed'):validate_sources(source,contract_from_ptx(before))

@pytest.mark.oracle
def test_normal_build_retains_original_return_body_and_native_ownership(tmp_path):
 if not oracle.available('ptxas'):pytest.skip('needs CUDA toolkit')
 source=write(tmp_path,DATA['configurations'][0]);out=tmp_path/'output';assert build.compile_paths(source,out,build.Options(no_sass=True)).rc==0
 entry,=json.loads((out/'manifest.json').read_text())['entries'];binary=(out/entry['cubin']).read_bytes()
 from openassembler.native_ownership import validate
 validate(entry,binary);next(iter(entry['kernels'][0]['python_source_graph']['helpers'].values()))['body_sha256']='0'*64
 with pytest.raises(ValueError,match='inventory'):validate(entry,binary)

