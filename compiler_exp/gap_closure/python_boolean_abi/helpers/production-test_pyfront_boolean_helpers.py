"""Logical Boolean helper inputs/returns with independent complete-bit effects and ownership."""
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
DATA=json.loads((Path(__file__).parent/'data/boolean_helper_fixtures.json').read_text())
HEADER='from typing import Final\nfrom openassembler.pyfront import *\n'

def write(root,case):
 for name,text in case['files'].items():
  path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
 return root/case['entry']

@pytest.mark.parametrize('case',DATA['configurations'],ids=lambda c:c['name'])
@pytest.mark.parametrize('name',list(DATA['values']))
@pytest.mark.parametrize('level',['safe','opt'])
def test_original_boolean_helpers_argument_copies_return_paths_and_atomic_snapshots(case,name,level,tmp_path):
 source=write(tmp_path,case);text,names=compile_file(source);graph=contract_from_ptx(text);validate_sources(source,graph)
 assert{r['path']for r in graph['files']}==set(case['files'])and graph['boolean_values']
 single=compile_kernel(find_kernel(source,name));single_graph=contract_from_ptx(single)
 assert all(set(row['kernels'])=={name}for row in single_graph['boolean_values'].values())
 module=parse_module(text);code,labels,words,_,registers=build.generate(module.kernels[name],'sm_80',level)
 _,_,single_words,_,single_regs=build.generate(parse_module(single).kernels[name],'sm_80',level);assert words==single_words and registers==single_regs
 data=DATA['values'][name];before,initial,wanted=(bytes.fromhex(data[key])for key in('input_hex','initial_hex','expected_hex'))
 ap,out,cp=0x100010000,0x100000000,0x100020000;guard=b'\xa5'*64
 for engine in('ptx','normal','poisoned'):
  memory=Memory();memory.write_bytes(ap-64,guard+before+guard);memory.write_bytes(out-64,guard+initial+guard);memory.write_bytes(cp-64,guard+bytes(len(bytes.fromhex(data['counter_hex'])))+guard)
  if engine=='ptx':Interp(module,name,PLaunch(block=(128,1,1),params={'a':ap,'out':out,'count':cp,'n':91}),memory).run()
  else:Machine(load_listing(listing(code,labels)),Launch(block=(128,1,1),params=struct.pack('<QQQI',ap,out,cp,91),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
  assert bytes(memory.read(out-64+i,1)for i in range(len(wanted)+128))==guard+wanted+guard
  assert bytes(memory.read(ap-64+i,1)for i in range(len(before)+128))==guard+before+guard
  assert bytes(memory.read(cp-64+i,1)for i in range(len(bytes.fromhex(data['counter_hex']))+128))==guard+bytes.fromhex(data['counter_hex'])+guard

@pytest.mark.parametrize('case',DATA['negative_controls'],ids=lambda c:c['name'])
def test_boolean_helper_type_effect_storage_and_return_domains_reject(case,tmp_path):
 source=tmp_path/(case['name']+'.py');source.write_text(case['source'])
 with pytest.raises((NotSupported,ra.NotSupported),match=case['reason']):
  text,_=compile_file(source)
  for kernel in parse_module(text).kernels.values():build.generate(kernel,'sm_80','safe')

@pytest.mark.parametrize('mutate',[
 lambda row:row.update(inputs='pred'),lambda row:row.update(inputs=[True]),
 lambda row:row.update(inputs=['f64']),lambda row:row.update(inputs=[['pred']]),
 lambda row:row.update(output='f64'),lambda row:row.update(output=True),
 lambda row:row.update(output=['pred']),lambda row:row.pop('inputs'),
 lambda row:row.pop('output'),
])
def test_invalid_logical_helper_signature_inventory_rejects(mutate,tmp_path):
 source=write(tmp_path,DATA['configurations'][0]);text,_=compile_file(source);graph=contract_from_ptx(text)
 mutate(next(iter(graph['helpers'].values())))
 with pytest.raises(ValueError,match='scalar signature'):contract_from_ptx(PREFIX+json.dumps(graph)+'\n')

@pytest.mark.parametrize('level',['safe','opt'])
def test_standalone_identity_helper_owns_predicate_signature_without_expression_marker(level,tmp_path):
 source=tmp_path/'identity.py';source.write_text(HEADER+'@device\ndef forward(flag:bool)->bool:\n    return flag\n@kernel\ndef k(a:Array(i32),out:Array(i32)):\n    i=tid_x()\n    flag=a[i]<7\n    out[i]=i32(forward(flag))\n')
 text,_=compile_file(source);graph=contract_from_ptx(text);validate_sources(source,graph)
 assert graph and 'boolean_values' not in graph and set(graph['helpers'])=={'identity.py:forward'}
 record=graph['helpers']['identity.py:forward'];assert record['inputs']==['pred'] and record['output']=='pred' and record['kernels']=={'k':1}
 module=parse_module(text);code,labels,words,_,regs=build.generate(module.kernels['k'],'sm_80',level)
 ap,out=0x100010000,0x100000000;before=struct.pack('<6i',-2147483648,-1,0,6,7,2147483647);wanted=struct.pack('<6i',1,1,1,1,0,0)
 for engine in ('ptx','normal','poisoned'):
  memory=Memory();memory.write_bytes(ap,before);memory.write_bytes(out,bytes(len(wanted)))
  if engine=='ptx':Interp(module,'k',PLaunch(block=(6,1,1),params={'a':ap,'out':out}),memory).run()
  else:Machine(load_listing(listing(code,labels)),Launch(block=(6,1,1),params=struct.pack('<QQ',ap,out),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
  assert bytes(memory.read(out+i,1)for i in range(len(wanted)))==wanted
  assert bytes(memory.read(ap+i,1)for i in range(len(before)))==before

def test_original_predicate_helper_body_refresh_updates_signature_ownership(tmp_path):
 source=write(tmp_path,DATA['configurations'][0]);ref=find_kernel(source,'invert_value');before=compile_kernel(ref);old=contract_from_ptx(before)
 helper=tmp_path/'library/helpers.py';helper.write_text(helper.read_text().replace('return not flag','return flag',1));after=compile_kernel(ref);new=contract_from_ptx(after)
 assert before!=after and old['helpers']['library/helpers.py:flip']['body_sha256']!=new['helpers']['library/helpers.py:flip']['body_sha256']
 with pytest.raises(ValueError,match='dependency changed'):validate_sources(source,old)

@pytest.mark.oracle
def test_normal_builder_binds_logical_predicate_signatures_and_rejects_mutated_manifest(tmp_path):
 if not oracle.available('ptxas'):pytest.skip('needs CUDA toolkit')
 source=write(tmp_path,DATA['configurations'][0]);out=tmp_path/'output';assert build.compile_paths(source,out,build.Options(no_sass=True)).rc==0
 entry,=json.loads((out/'manifest.json').read_text())['entries'];binary=(out/entry['cubin']).read_bytes()
 from openassembler.native_ownership import validate
 validate(entry,binary);next(iter(entry['kernels'][0]['python_source_graph']['helpers'].values()))['output']='s32'
 with pytest.raises(ValueError,match='inventory'):validate(entry,binary)

