"""Original static tuple loop control, numerical, source and ownership contracts."""
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
DATA=json.loads((Path(__file__).parent/'data/static_loop_fixtures.json').read_text())
HEADER='from typing import Final\nfrom openassembler.pyfront import *\n'

def write(root,case):
 for name,text in case['files'].items():
  path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
 return root/case['entry']

@pytest.mark.parametrize('case',DATA['configurations'],ids=lambda c:c['name'])
@pytest.mark.parametrize('name',list(DATA['values']))
@pytest.mark.parametrize('level',['safe','opt'])
def test_complete_static_tuple_loops_targets_exits_else_and_original_identity(case,name,level,tmp_path):
 source=write(tmp_path,case);text,names=compile_file(source);graph=contract_from_ptx(text);validate_sources(source,graph)
 assert{r['path']for r in graph['files']}==set(case['files'])and graph['static_iterations']
 single=compile_kernel(find_kernel(source,name));single_graph=contract_from_ptx(single)
 assert all(set(row['kernels'])=={name}for row in single_graph['static_iterations'].values())
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
def test_static_tuple_type_control_resource_and_convergence_domains_reject(case,tmp_path):
 source=tmp_path/(case['name']+'.py');source.write_text(HEADER+case['source'])
 with pytest.raises((NotSupported,ra.NotSupported),match=case['reason']):
  text,_=compile_file(source)
  for kernel in parse_module(text).kernels.values():build.generate(kernel,'sm_80','safe')

@pytest.mark.parametrize('body,expected',[
 ('    for c in (3,5):\n        if c==3:\n            return\n        out[0]=i64(c)\n',None),
 ('    c=11\n    for c in ():\n        c=19\n    else:\n        out[0]=i64(c)\n',11),
 ('    value=i64(0)\n    for c in (1,2):\n        for d in (3,4):\n            break\n        else:\n            value+=i64(17)\n    else:\n        value+=i64(19)\n    out[0]=value\n',19),
 ('    value=i64(0)\n    for c in (1,2):\n        for d in (3,4):\n            value+=i64(d)\n        else:\n            break\n    else:\n        value+=i64(19)\n    out[0]=value\n',7),
])
def test_kernel_return_empty_else_and_nearest_loop_else_break_targets(body,expected,tmp_path):
 source=tmp_path/'loop.py';source.write_text(HEADER+'@kernel\ndef k(out:Array(i64)):\n'+body)
 text,_=compile_file(source);module=parse_module(text);out=0x100000000;memory=Memory();memory.write_bytes(out,struct.pack('<q',-991))
 Interp(module,'k',PLaunch(params={'out':out}),memory).run()
 assert bytes(memory.read(out+i,1)for i in range(8))==struct.pack('<q',-991 if expected is None else expected)


def test_constant_tuple_change_refreshes_iterated_values_and_dependency_identity(tmp_path):
 case=DATA['configurations'][0];source=write(tmp_path,case);ref=find_kernel(source,'plain');before=compile_kernel(ref)
 values=tmp_path/'library/constants.py';values.write_text(values.read_text().replace('VALUES_PLAIN:Final=(3,-5,7)','VALUES_PLAIN:Final=(11,-5,7)'))
 after=compile_kernel(ref);assert before!=after
 with pytest.raises(ValueError,match='dependency changed'):validate_sources(source,contract_from_ptx(before))


@pytest.mark.parametrize('mutate',[
 lambda row:row.update(path='../escape'),lambda row:row.update(values={'type':'int','value':'3'}),
 lambda row:row.update(kernels={'plain':0}),lambda row:row.update(line=0),
])
def test_invalid_static_iteration_metadata_rejects(tmp_path,mutate):
 source=write(tmp_path,DATA['configurations'][0]);text,_=compile_file(source);graph=contract_from_ptx(text);mutate(next(iter(graph['static_iterations'].values())))
 with pytest.raises(ValueError):contract_from_ptx(PREFIX+json.dumps(graph)+'\n')


@pytest.mark.oracle
def test_normal_builder_snapshots_static_iteration_value_and_control_identity(tmp_path):
 if not oracle.available('ptxas'):pytest.skip('needs CUDA toolkit')
 source=write(tmp_path,DATA['configurations'][0]);out=tmp_path/'output';assert build.compile_paths(source,out,build.Options(no_sass=True)).rc==0
 entry,=json.loads((out/'manifest.json').read_text())['entries'];binary=(out/entry['cubin']).read_bytes()
 from openassembler.native_ownership import validate
 validate(entry,binary);next(iter(entry['kernels'][0]['python_source_graph']['static_iterations'].values()))['values']={'type':'tuple','items':[]}
 with pytest.raises(ValueError,match='inventory'):validate(entry,binary)
