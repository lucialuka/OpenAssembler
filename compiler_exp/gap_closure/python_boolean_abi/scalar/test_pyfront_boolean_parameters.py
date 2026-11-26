"""Logical Boolean source/host roles, exact whole-byte effects and native custody."""
import json,struct
from pathlib import Path
import pytest
from openassembler import build,oracle
from openassembler.pyfront import compile_file,compile_kernel,find_kernel,NotSupported
from openassembler.pyfront.modules import validate_sources,contract_from_ptx
from openassembler.ptx.parse import parse_module
from openassembler.verify import parameter_contract_from_ptx,run_and_diff,_TYPE_METADATA
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing

DATA=json.loads((Path(__file__).parent/'data/boolean_parameter_fixtures.json').read_text())
HEADER='from openassembler.pyfront import *\n'


def write(root,case):
 for name,text in case['files'].items():
  path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
 return root/case['entry']


@pytest.mark.parametrize('configuration',DATA['configurations'],ids=lambda row:row['name'])
@pytest.mark.parametrize('case',DATA['cases'],ids=lambda row:row['name'])
@pytest.mark.parametrize('level',['safe','opt'])
def test_original_boolean_parameters_canonical_bits_lazy_effects_and_adjacent_numeric_slots(configuration,case,level,tmp_path):
 source=write(tmp_path,configuration);text,names=compile_file(source);graph=contract_from_ptx(text);validate_sources(source,graph)
 module=parse_module(text);kernel=module.kernels[case['name']];contract=parameter_contract_from_ptx(text,kernel);assert any(row['dtype']=='bool'for row in contract['parameters'])
 code,labels,words,_,regs=build.generate(kernel,'sm_80',level)
 single=compile_kernel(find_kernel(source,case['name']));_,_,single_words,_,single_regs=build.generate(parse_module(single).kernels[case['name']],'sm_80',level);assert words==single_words and regs==single_regs
 before=bytes.fromhex(case['input_hex']);initial=bytes.fromhex(case['initial_hex']);guard=b'\xa5'*64
 for row in case['rows']:
  vals=row['values'];wanted=guard+bytes.fromhex(row['expected_hex'])+guard;counter=guard+bytes.fromhex(row['counter_hex'])+guard
  for engine in('ptx','normal','poisoned'):
   memory=Memory();memory.write_bytes(vals['a']-64,guard+before+guard);memory.write_bytes(vals['out']-64,guard+initial+guard);memory.write_bytes(vals['counter']-64,guard+bytes(512)+guard)
   if engine=='ptx':Interp(module,case['name'],PLaunch(block=(128,1,1),params=vals),memory).run()
   else:Machine(load_listing(listing(code,labels)),Launch(block=(128,1,1),params=bytes.fromhex(row['packed_hex']),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
   for address,data in((vals['a'],guard+before+guard),(vals['out'],wanted),(vals['counter'],counter)):assert bytes(memory.read(address-64+i,1)for i in range(len(data)))==data


@pytest.mark.parametrize('case',DATA['negatives'],ids=lambda row:row['name'])
def test_boolean_parameter_storage_arithmetic_effect_and_annotation_boundaries_reject(case,tmp_path):
 source=tmp_path/'negative.py';source.write_text(case['source'])
 with pytest.raises(NotSupported,match=case['reason']):compile_file(source)


@pytest.mark.parametrize('mutate',[
 lambda c:c['parameters'][0].update(dtype='s32'),lambda c:c['parameters'][0].update(kind='array'),
 lambda c:c['parameters'][0].update(name='other'),lambda c:c['parameters'].pop(),
 lambda c:c.update(kernel='other'),lambda c:c.update(schema=True),
])
def test_boolean_logical_type_metadata_must_match_physical_slot(mutate,tmp_path):
 source=tmp_path/'k.py';source.write_text(HEADER+'@kernel\ndef k(flag:bool,out:Array(i32)):\n    out[0]=i32(flag)\n');text,_=compile_file(source);module=parse_module(text);contract=parameter_contract_from_ptx(text,module.kernels['k']);mutate(contract)
 raw='\n'.join(line for line in text.splitlines()if not line.startswith(_TYPE_METADATA));raw=_TYPE_METADATA+json.dumps(contract)+'\n'+raw
 with pytest.raises(ValueError):parameter_contract_from_ptx(raw,module.kernels['k'])


@pytest.mark.parametrize('seed',range(8))
def test_typed_verifier_boolean_named_dimension_stays_canonical(seed,tmp_path):
 source=tmp_path/'flag.py';source.write_text(HEADER+'@kernel\ndef k(n:bool,out:Array(i32)):\n    out[tid_x()]=i32(n)\n');text,_=compile_file(source);module=parse_module(text);kernel=module.kernels['k'];contract=parameter_contract_from_ptx(text,kernel);code,labels,_,_,_=build.generate(kernel,'sm_80','opt')
 assert run_and_diff(module,'k',code,labels,n_elems=17,seed=seed,parameter_contract=contract)==[]


def structural_plan(tmp_path,argument,logical=True,raw_change=False,contract_change=False):
 from tests.native_fixture import save_native_manifest
 from openassembler.runner import RunOptions,build_plan
 import hashlib
 text='.entry k(.param .u8 flag){ret;}'
 contract=dict(schema=1,kernel='k',parameters=[dict(name='flag',kind='scalar',dtype='bool')])
 if logical:text=_TYPE_METADATA+json.dumps(contract)+'\n'+text
 source=tmp_path/'k.ptx';source.write_text(text)
 kernel=dict(name='k',params=[['u8','flag']],compiled_by='openassembler')
 if logical:kernel['logical_parameter_contract']=contract
 manifest=save_native_manifest(tmp_path,tmp_path,'sm_80','opt',[dict(source='k.ptx',ptx='k.ptx',cubin='k.cubin',kernels=[kernel])]);path=tmp_path/'manifest.json';saved=json.loads(path.read_text());saved['entries'][0]['native_ownership']['ptx_sha256']=hashlib.sha256(text.encode()).hexdigest()
 if contract_change:
  changed=dict(schema=1,kernel='k',parameters=[dict(name='flag',kind='scalar',dtype='u8')]);saved['entries'][0]['kernels'][0]['logical_parameter_contract']=changed
 if raw_change:source.write_text(text+'\n// changed\n')
 path.write_text(json.dumps(saved))
 spec=tmp_path/'launch.json';spec.write_text(json.dumps(dict(buffers={'out':dict(count=1,init='zeros')},steps=[dict(kernel='k',args=[argument])],compare=['out'],reps=0)))
 return build_plan(RunOptions(out_dir=tmp_path,spec=spec,reps=0))[0]


@pytest.mark.parametrize('value',[True,False])
def test_explicit_boolean_host_values_pack_to_canonical_byte(tmp_path,value):
 plan=structural_plan(tmp_path,{'bool':value});assert plan.jobs[0]['steps'][0][-1]==[('u8',int(value))]


@pytest.mark.parametrize('argument',[
 {'bool':0},{'bool':1},{'bool':'true'},{'bool':None},{'bool':True,'offset':0},
 {'u8':1},{'u8':0},{'i8':1},{'u32':1},{'buf':'out'},
])
def test_boolean_launch_rejects_non_boolean_values_or_roles(tmp_path,argument):
 from openassembler.runner import RunError
 with pytest.raises(RunError):structural_plan(tmp_path,argument)


def test_boolean_host_argument_cannot_relabel_raw_ptx_byte(tmp_path):
 from openassembler.runner import RunError
 with pytest.raises(RunError,match='logical source contract'):structural_plan(tmp_path,{'bool':True},logical=False)


def test_logical_boolean_launch_requires_exact_ptx_bytes(tmp_path):
 from openassembler.runner import RunError
 with pytest.raises(RunError,match='identity differs'):structural_plan(tmp_path,{'bool':True},raw_change=True)


def test_mutated_logical_parameter_manifest_cannot_mutate_ownership_snapshot(tmp_path):
 from openassembler.runner import RunError
 with pytest.raises((RunError,ValueError)):structural_plan(tmp_path,{'bool':True},contract_change=True)


@pytest.mark.oracle
def test_normal_boolean_build_owns_logical_signature_and_rejects_removal(tmp_path):
 if not oracle.available('ptxas'):pytest.skip('needs CUDA toolkit')
 source=write(tmp_path,DATA['configurations'][0]);out=tmp_path/'output';assert build.compile_paths(source,out,build.Options(no_sass=True,verify=True)).rc==0
 entry,=json.loads((out/'manifest.json').read_text())['entries'];binary=(out/entry['cubin']).read_bytes()
 from openassembler.native_ownership import validate
 validate(entry,binary);assert all(kernel['logical_parameter_contract']and kernel['verify']=='ok'for kernel in entry['kernels']);entry['kernels'][0].pop('logical_parameter_contract')
 with pytest.raises(ValueError,match='inventory'):validate(entry,binary)
