"""Original-source Final scalar/tuple semantics and rejection/dependency contracts."""
import copy,json,struct
from pathlib import Path
import pytest
from openassembler import build,oracle
from openassembler.pyfront import compile_file,compile_kernel,find_kernel,NotSupported
from openassembler.pyfront.modules import contract_from_ptx,validate_sources,PREFIX
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing
DATA=json.loads((Path(__file__).parent/'data/static_value_fixtures.json').read_text())

def write(root,case):
 for name,text in case['files'].items():
  path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
 return root/case['entry']

@pytest.mark.parametrize('case',DATA['configurations'],ids=lambda c:c['name'])
@pytest.mark.parametrize('name',list(DATA['values']))
@pytest.mark.parametrize('level',['safe','opt'])
def test_original_static_declarations_exact_bits_and_per_kernel_constant_inventory(case,name,level,tmp_path):
 source=write(tmp_path,case);text,names=compile_file(source);module=parse_module(text);graph=contract_from_ptx(text);validate_sources(source,graph)
 assert {r['path']for r in graph['files']}==set(case['files'])
 gain=next(r for r in graph['static_values'].values()if r['name']=='GAIN');assert set(gain['kernels'])=={'signed','shared_helper','local_shadow'}
 single=compile_kernel(find_kernel(source,name));single_graph=contract_from_ptx(single)
 assert all(set(row['kernels'])=={name}for row in single_graph['static_values'].values())
 code,labels,words,_,regs=build.generate(module.kernels[name],'sm_80',level);assert words
 _,_,other,_,other_regs=build.generate(parse_module(single).kernels[name],'sm_80',level);assert words==other and regs==other_regs
 data=DATA['values'][name];before,initial,wanted=(bytes.fromhex(data[key])for key in('input_hex','initial_hex','expected_hex'))
 ap,out=0x100010000,0x100000000;guard=b'\xa5'*64
 for engine in('ptx','normal','poisoned'):
  memory=Memory();memory.write_bytes(ap-64,guard+before+guard);memory.write_bytes(out-64,guard+initial+guard)
  if engine=='ptx':Interp(module,name,PLaunch(block=(128,1,1),params={'a':ap,'out':out,'n':91}),memory).run()
  else:Machine(load_listing(listing(code,labels)),Launch(block=(128,1,1),params=struct.pack('<QQI',ap,out,91),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
  assert bytes(memory.read(out-64+i,1)for i in range(len(wanted)+128))==guard+wanted+guard
  assert bytes(memory.read(ap-64+i,1)for i in range(len(before)+128))==guard+before+guard

@pytest.mark.parametrize('case',DATA['negatives'],ids=lambda c:c['name'])
def test_invalid_static_declarations_operations_and_mutations_reject_with_source(case,tmp_path):
 source=write(tmp_path,DATA['configurations'][0]);file=tmp_path/case['file'];text=file.read_text();assert case['old']in text;file.write_text(text.replace(case['old'],case['new'],1))
 if case['name']=='cyclic':file.write_text(file.read_text().replace('BIAS: types.Final[int] = -(1 << 32) + 7','BIAS: types.Final[int] = GAIN'))
 with pytest.raises(NotSupported,match=case['reason'])as error:compile_file(source)
 assert error.value.path and error.value.lineno

@pytest.mark.parametrize('definition,expression,expected_value',[
 ('VALUE:Final[int]=(-17//3)*9+(-17%3)','i64(VALUE)',-53),
 ('VALUE:Final[int]=(2**100+7)>>80','i64(VALUE)',1<<20),
 ('TABLE:Final[tuple[int,...]]=(3,5,7)\nVALUE:Final[int]=len(TABLE[::-1])+TABLE[-1]','i64(VALUE)',10),
 ('FLAG:Final[bool]=(3<5)\nVALUE:Final[int]=7','i64(VALUE)',7),
 ('VALUE:Final[float]=16777217.0','VALUE',16777216.0),
])
def test_constant_only_kernel_original_identity_and_independent_values(definition,expression,expected_value,tmp_path):
 dtype='f32'if type(expected_value)is float else'i64';source=tmp_path/'static.py'
 source.write_text('from typing import Final\nfrom openassembler.pyfront import *\n'+definition+'\n@kernel\ndef k(out:Array('+dtype+')):\n    out[0]='+expression+'\n')
 text,_=compile_file(source);graph=contract_from_ptx(text);assert not graph['helpers']and graph['static_values']
 module=parse_module(text);memory=Memory();out=0x100000000;Interp(module,'k',PLaunch(params={'out':out}),memory).run()
 fmt='<f'if dtype=='f32'else'<q';actual=bytes(memory.read(out+i,1)for i in range(struct.calcsize(fmt)));assert actual==struct.pack(fmt,expected_value)

@pytest.mark.parametrize('prefix,body',[
 ('TABLE:Final=(3,5,7)','len=17\n    out[0]=i64(len(TABLE))'),
 ('TABLE:Final=(3,5,7)','TABLE=i64(17)\n    out[0]=i64(TABLE[0])'),
 ('VALUE:Final[int]=3','out[0]=i64(VALUE)\n    VALUE=i64(7)'),
 ('VALUE:Final=tuple([3,5])','out[0]=i64(VALUE[0])'),
 ('VALUE:Final[int]=1<<2048','out[0]=i64(VALUE)'),
 ('VALUE:Final[tuple[int,...]]=('+','.join('1'for _ in range(257))+')','out[0]=i64(VALUE[0])'),
])
def test_static_symbols_do_not_override_local_bindings_or_resource_limits(prefix,body,tmp_path):
 source=tmp_path/'invalid.py';source.write_text('from typing import Final\nfrom openassembler.pyfront import *\n'+prefix+'\n@kernel\ndef k(out:Array(i64)):\n    '+body+'\n')
 with pytest.raises(NotSupported):compile_file(source)


def test_changed_original_constant_invalidates_graph_and_fresh_reference(tmp_path):
 source=write(tmp_path,DATA['configurations'][0]);ref=find_kernel(source,'signed');before=compile_kernel(ref)
 file=tmp_path/'library/constants.py';file.write_text(file.read_text().replace('GAIN: Frozen[int] = 3','GAIN: Frozen[int] = 11'))
 after=compile_kernel(ref);assert before!=after
 with pytest.raises(ValueError,match='dependency changed'):validate_sources(source,contract_from_ptx(before))


@pytest.mark.parametrize('change',[
 lambda g:next(iter(g['static_values'].values())).update(path='../escape'),
 lambda g:next(iter(g['static_values'].values())).update(value={'type':'int','value':'03'}),
 lambda g:next(iter(g['static_values'].values())).update(value={'type':'float','bits':'7ff0000000000000'}),
 lambda g:next(iter(g['static_values'].values())).update(kernels={'signed':0}),
])
def test_invalid_static_metadata_rejects(change,tmp_path):
 source=write(tmp_path,DATA['configurations'][0]);text,_=compile_file(source);graph=contract_from_ptx(text);change(graph)
 with pytest.raises(ValueError):contract_from_ptx(PREFIX+json.dumps(graph)+'\n')


@pytest.mark.oracle
def test_normal_build_owns_original_final_source_and_rejects_value_tampering(tmp_path):
 if not oracle.available('ptxas'):pytest.skip('needs CUDA toolkit')
 source=write(tmp_path,DATA['configurations'][0]);out=tmp_path/'out';assert build.compile_paths(source,out,build.Options(no_sass=True)).rc==0
 entry,=json.loads((out/'manifest.json').read_text())['entries'];binary=(out/entry['cubin']).read_bytes()
 from openassembler.native_ownership import validate
 validate(entry,binary);next(iter(entry['kernels'][0]['python_source_graph']['static_values'].values()))['value']={'type':'int','value':'19'}
 with pytest.raises(ValueError,match='inventory'):validate(entry,binary)
