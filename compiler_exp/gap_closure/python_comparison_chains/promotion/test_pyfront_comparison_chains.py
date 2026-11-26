"""Original typed condition chains and exact short-circuit effects/source ownership."""
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
DATA=json.loads((Path(__file__).parent/'comparison_chain_fixtures.json').read_text())
HEADER='from typing import Final\nfrom openassembler.pyfront import *\n'

def write(root,case):
 for name,text in case['files'].items():
  path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
 return root/case['entry']

@pytest.mark.parametrize('case',DATA['configurations'],ids=lambda c:c['name'])
@pytest.mark.parametrize('name',list(DATA['values']))
@pytest.mark.parametrize('level',['safe','opt'])
def test_original_condition_chains_types_short_circuit_and_atomic_snapshots(case,name,level,tmp_path):
 source=write(tmp_path,case);text,names=compile_file(source);graph=contract_from_ptx(text);validate_sources(source,graph)
 assert{r['path']for r in graph['files']}==set(case['files'])and graph['comparison_chains']
 single=compile_kernel(find_kernel(source,name));single_graph=contract_from_ptx(single)
 assert all(set(row['kernels'])=={name}for row in single_graph['comparison_chains'].values())
 module=parse_module(text);code,labels,words,_,registers=build.generate(module.kernels[name],'sm_80',level)
 _,_,single_words,_,single_regs=build.generate(parse_module(single).kernels[name],'sm_80',level);assert words==single_words and registers==single_regs
 data=DATA['values'][name];before,initial,wanted=(bytes.fromhex(data[key])for key in('input_hex','initial_hex','expected_hex'))
 ap,out,cp=0x100010000,0x100000000,0x100020000;guard=b'\xa5'*64
 for engine in('ptx','normal','poisoned'):
  memory=Memory();memory.write_bytes(ap-64,guard+before+guard);memory.write_bytes(out-64,guard+initial+guard);memory.write_bytes(cp-64,guard+bytes(len(initial))+guard)
  if engine=='ptx':Interp(module,name,PLaunch(block=(128,1,1),params={'a':ap,'out':out,'count':cp,'n':91}),memory).run()
  else:Machine(load_listing(listing(code,labels)),Launch(block=(128,1,1),params=struct.pack('<QQQI',ap,out,cp,91),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
  assert bytes(memory.read(out-64+i,1)for i in range(len(wanted)+128))==guard+wanted+guard
  assert bytes(memory.read(ap-64+i,1)for i in range(len(before)+128))==guard+before+guard
  assert bytes(memory.read(cp-64+i,1)for i in range(len(initial)+128))==guard+bytes.fromhex(data['counter_hex'])+guard

@pytest.mark.parametrize('case',DATA['negative_controls'],ids=lambda c:c['name'])
def test_condition_chain_type_value_initialization_and_convergence_domains_reject(case,tmp_path):
 source=tmp_path/(case['name']+'.py');source.write_text(case['source'])
 with pytest.raises((NotSupported,ra.NotSupported),match=case['reason']):
  text,_=compile_file(source)
  for kernel in parse_module(text).kernels.values():build.generate(kernel,'sm_80','safe')

@pytest.mark.parametrize('mutate',[
 lambda row:row.update(path='../escape.py'),lambda row:row.update(function='bad name'),
 lambda row:row.update(line=0),lambda row:row.update(column=-1),
 lambda row:row.update(operators=[]),lambda row:row.update(operators=['Lt','In']),
 lambda row:row.update(specialized_expression_sha256='bad'),
 lambda row:row.update(kernels={'ordinary':0}),lambda row:row.update(kernels={'ordinary':True}),
 lambda row:row.update(kernels={}),
])
def test_invalid_comparison_chain_expression_source_use_metadata_rejects(tmp_path,mutate):
 source=write(tmp_path,DATA['configurations'][0]);text,_=compile_file(source);graph=contract_from_ptx(text);mutate(next(iter(graph['comparison_chains'].values())))
 with pytest.raises(ValueError):contract_from_ptx(PREFIX+json.dumps(graph)+'\n')

@pytest.mark.parametrize('dtype,fmt,values',[
 ('i32','i',[-19,-1,0,1,3,19]),('u32','I',[0,1,3,19,1<<31,(1<<32)-1]),
 ('i64','q',[-(1<<63),-1,0,1,3,(1<<63)-1]),('u64','Q',[0,1,3,19,1<<63,(1<<64)-1]),
 ('f32','f',[float('nan'),float('-inf'),-0.0,0.0,3.0,float('inf')]),
])
@pytest.mark.parametrize('level',['safe','opt'])
def test_adjacent_literal_context_and_standalone_chain_original_ownership(dtype,fmt,values,level,tmp_path):
 source=tmp_path/'chain.py';source.write_text(HEADER+f'@kernel\ndef k(a:Array({dtype}),out:Array(i32)):\n    i=tid_x()\n    value=11\n    if 0<1<a[i]:\n        value=7\n    out[i]=value\n')
 text,_=compile_file(source);graph=contract_from_ptx(text);assert graph and not graph['helpers']and graph['comparison_chains'];validate_sources(source,graph)
 module=parse_module(text);code,labels,words,_,regs=build.generate(module.kernels['k'],'sm_80',level);before=struct.pack('<6'+fmt,*values);wanted=struct.pack('<6i',*[7 if 1<x else 11 for x in values]);ap,out=0x100010000,0x100000000
 for engine in('ptx','normal','poisoned'):
  memory=Memory();memory.write_bytes(ap,before);memory.write_bytes(out,bytes(24))
  if engine=='ptx':Interp(module,'k',PLaunch(block=(6,1,1),params={'a':ap,'out':out}),memory).run()
  else:Machine(load_listing(listing(code,labels)),Launch(block=(6,1,1),params=struct.pack('<QQ',ap,out),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
  assert bytes(memory.read(out+i,1)for i in range(24))==wanted
  assert bytes(memory.read(ap+i,1)for i in range(len(before)))==before

def test_original_chain_expression_and_dependency_refresh_on_source_change(tmp_path):
 source=write(tmp_path,DATA['configurations'][0]);ref=find_kernel(source,'ordinary');before=compile_kernel(ref);old=contract_from_ptx(before)
 helper=tmp_path/'library/helpers.py';helper.write_text(helper.read_text().replace('if -3<x<7:','if -3<x<9:',1));after=compile_kernel(ref);new=contract_from_ptx(after);assert before!=after
 old_record=next(iter(old['comparison_chains'].values()));new_record=next(iter(new['comparison_chains'].values()));assert old_record['specialized_expression_sha256']!=new_record['specialized_expression_sha256']
 with pytest.raises(ValueError,match='dependency changed'):validate_sources(source,old)

def test_uniform_chain_barrier_remains_admitted(tmp_path):
 source=tmp_path/'uniform.py';source.write_text(HEADER+'@kernel\ndef k(out:Array(i32)):\n    if 0<=ctaid_x()<3:\n        syncthreads()\n    out[tid_x()]=7\n');text,_=compile_file(source);assert 'bar.sync'in text

@pytest.mark.oracle
def test_normal_builder_binds_chain_expression_and_rejects_manifest_mutation(tmp_path):
 if not oracle.available('ptxas'):pytest.skip('needs CUDA toolkit')
 source=write(tmp_path,DATA['configurations'][0]);out=tmp_path/'output';assert build.compile_paths(source,out,build.Options(no_sass=True)).rc==0
 entry,=json.loads((out/'manifest.json').read_text())['entries'];binary=(out/entry['cubin']).read_bytes()
 from openassembler.native_ownership import validate
 validate(entry,binary);next(iter(entry['kernels'][0]['python_source_graph']['comparison_chains'].values()))['specialized_expression_sha256']='0'*64
 with pytest.raises(ValueError,match='inventory'):validate(entry,binary)

