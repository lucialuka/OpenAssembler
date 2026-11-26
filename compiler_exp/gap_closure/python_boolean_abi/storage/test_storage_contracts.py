"""Logical Boolean storage contracts, canonical host bytes and rejection boundaries."""
import hashlib,json
from pathlib import Path
import pytest
from openassembler.pyfront import compile_file,NotSupported,Array,Buffer
from openassembler.pyfront.types import Scalar,element_bytes
from openassembler.pyfront.buffers import metadata,contract_from_ptx
from openassembler.ptx.parse import parse_module
from openassembler.verify import parameter_contract_from_ptx,run_and_diff
from openassembler import build
HEADER='from openassembler.pyfront import *\n'

@pytest.mark.parametrize('shape,strides,access,storage',[
 ((4,7),None,'readwrite',28),((4,7),(11,1),'readwrite',40),((4,7),(1,5),'readwrite',34),((4,7),(0,1),'read',7),((2,3,4,5),None,'readwrite',120),
])
def test_public_boolean_layout_has_exact_byte_extent(shape,strides,access,storage):
 view=Buffer(bool,shape,strides=strides,access=access);record=metadata('a',view)
 assert record['dtype']=='bool'and record['element_bytes']==1 and record['storage_elements']==storage and record['alignment']==1
 assert Array(bool).elem==Scalar('pred') and element_bytes('pred')==1

@pytest.mark.parametrize('kwargs',[
 dict(shape=(4,7),strides=(0,1)),dict(shape=(4,7),strides=(1,1)),dict(shape=(4,7),strides=(-7,1)),dict(shape=(4,7),alignment=0),dict(shape=(4,7),alignment=3),dict(shape=(4,7),alignment=512),dict(shape=(0,7)),dict(shape=(4,7),strides=(2**31-1,1)),dict(shape=(True,)),
])
def test_boolean_buffer_shape_alignment_overlap_bounds_reject(kwargs):
 with pytest.raises(ValueError):Buffer(bool,**kwargs)

NEGATIVES=[
 ('numeric_store','@kernel\ndef k(out:Array(bool)):\n    out[0]=1\n','storing'),
 ('float_store','@kernel\ndef k(out:Array(bool)):\n    out[0]=1.0\n','storing'),
 ('boolean_to_numeric','@kernel\ndef k(out:Array(i32)):\n    out[0]=True\n','storing'),
 ('numeric_buffer_store','@kernel\ndef k(out:Buffer(bool,(4,7))):\n    out[0,0]=0\n','storing'),
 ('readonly','@kernel\ndef k(out:Buffer(bool,(4,7),access="read")):\n    out[0,0]=False\n','read-only'),
 ('boolean_index','@kernel\ndef k(a:Array(bool),out:Array(bool)):\n    out[0]=a[True]\n','int32'),
 ('boolean_buffer_index','@kernel\ndef k(a:Buffer(bool,(4,7)),out:Array(bool)):\n    out[0]=a[False,0]\n','int32'),
 ('negative_array_index','@kernel\ndef k(a:Array(bool),out:Array(bool)):\n    out[0]=a[-1]\n','negative index'),
 ('shared_bounds','@kernel\ndef k(out:Array(bool)):\n    s=shared_array(bool,1)\n    s[1]=True\n    out[0]=s[1]\n','out of range'),
 ('nested_shared','@kernel\ndef k(out:Array(bool),flag:bool):\n    if flag:\n        s=shared_array(bool,1)\n','top level'),
 ('shared_no_writer','@kernel\ndef k(out:Array(bool)):\n    s=shared_array(bool,128)\n    out[tid_x()]=s[tid_x()]\n','no possible writer'),
 ('divergent_barrier','@kernel\ndef k(a:Array(bool)):\n    if a[tid_x()]:\n        syncthreads()\n','thread-dependent'),
 ('boolean_atomic','@kernel\ndef k(a:Array(bool)):\n    atomic_add(a,0,1)\n','atomic'),
 ('boolean_arithmetic','@kernel\ndef k(a:Array(bool),out:Array(bool)):\n    out[0]=a[0]+True\n','arithmetic'),
 ('mixed_lazy','@kernel\ndef k(a:Array(bool),out:Array(bool)):\n    out[0]=a[0] or 1\n','fixed scalar kind'),
 ('bool_comparison','@kernel\ndef k(a:Array(bool),out:Array(bool)):\n    out[0]=a[0]==True\n','compar'),
 ('bool_shadow','bool=i32\n@kernel\ndef k(out:Array(bool)):\n    out[0]=False\n','annotation'),
 ('helper_array','@device\ndef f(a:Array(bool))->bool:\n    return a[0]\n@kernel\ndef k(a:Array(bool),out:Array(bool)):\n    out[0]=f(a)\n','scalar'),
]
@pytest.mark.parametrize('name,source,reason',NEGATIVES,ids=lambda v:v if isinstance(v,str)and len(v)<30 else None)
def test_original_type_effect_convergence_negatives(name,source,reason,tmp_path):
 path=tmp_path/(name+'.py');path.write_text(HEADER+source)
 with pytest.raises(NotSupported,match=reason):compile_file(path)

@pytest.mark.parametrize('mutation',[
 lambda r:r.update(element_bytes=4),lambda r:r.update(dtype='pred'),lambda r:r.update(storage_elements=29),lambda r:r.update(alignment=0),lambda r:r.update(strides=[0,1]),lambda r:r.update(name='other'),lambda r:r.update(shape=[4,True]),lambda r:r.update(element_bytes=True),
])
def test_boolean_shaped_metadata_cannot_change_storage_role(mutation,tmp_path):
 path=tmp_path/'k.py';path.write_text(HEADER+'@kernel\ndef k(a:Buffer(bool,(4,7))):\n    a[0,0]=True\n');text,_=compile_file(path);module=parse_module(text);contract=contract_from_ptx(text,module.kernels['k']);mutation(contract['buffers'][0]);marker='// openassembler-pyfront-buffer-v1 '
 changed='\n'.join(marker+json.dumps(contract)if line.startswith(marker)else line for line in text.splitlines())
 with pytest.raises(ValueError):contract_from_ptx(changed,module.kernels['k'])

@pytest.mark.parametrize('seed',range(8))
def test_typed_boolean_arrays_use_canonical_bytes(seed,tmp_path):
 path=tmp_path/'k.py';path.write_text(HEADER+'@kernel\ndef k(a:Array(bool),out:Array(bool),n:i32):\n    i=tid_x()\n    if i<n:\n        out[i]=not a[i]\n');text,_=compile_file(path);module=parse_module(text);kernel=module.kernels['k'];code,labels,_,_,_=build.generate(kernel,'sm_80','opt')
 assert run_and_diff(module,'k',code,labels,n_elems=37,seed=seed,parameter_contract=parameter_contract_from_ptx(text,kernel))==[]

def structural_plan(tmp_path,dtype='bool',init=None,logical=True,argument=None):
 from tests.native_fixture import save_native_manifest
 from openassembler.runner import RunOptions,build_plan
 contract=dict(schema=1,kernel='k',parameters=[dict(name='a',kind='array',dtype='bool')]);text='.entry k(.param .u64 a){ret;}'
 if logical:text='// openassembler-pyfront-types-v1 '+json.dumps(contract)+'\n'+text
 (tmp_path/'k.ptx').write_text(text);kernel=dict(name='k',params=[['u64','a']],compiled_by='openassembler')
 if logical:kernel['logical_parameter_contract']=contract
 save_native_manifest(tmp_path,tmp_path,'sm_80','opt',[dict(source='k.ptx',ptx='k.ptx',cubin='k.cubin',kernels=[kernel])]);path=tmp_path/'manifest.json';saved=json.loads(path.read_text());saved['entries'][0]['native_ownership']['ptx_sha256']=hashlib.sha256(text.encode()).hexdigest();path.write_text(json.dumps(saved))
 spec=tmp_path/'launch.json';spec.write_text(json.dumps(dict(buffers={'a':dict(dtype=dtype,count=8,init=init if init is not None else [True,False,True])},steps=[dict(kernel='k',args=[argument or {'buf':'a'}])],compare=['a'],reps=0)))
 return build_plan(RunOptions(out_dir=tmp_path,spec=spec,reps=0))[0]

def test_public_boolean_buffer_has_byte_extent_and_canonical_initial_values(tmp_path):
 plan=structural_plan(tmp_path);job=plan.jobs[0];assert job['buffers']=={'a':8}and job['init']['a']==bytes([1,0,1])

@pytest.mark.parametrize('how',['zeros','random'])
def test_public_boolean_buffer_zero_and_random_initialization_is_canonical(tmp_path,how):
 job=structural_plan(tmp_path,init=how).jobs[0];assert job['buffers']['a']==8
 assert not job['init'].get('a')or set(job['init']['a'])<={0,1}

@pytest.mark.parametrize('init',[[1],[0],[2],['true'],[None],[True,1]])
def test_public_boolean_buffer_rejects_non_boolean_initial_values(tmp_path,init):
 from openassembler.runner import RunError
 with pytest.raises(RunError,match='true or false'):structural_plan(tmp_path,init=init)

@pytest.mark.parametrize('dtype',['f32','i32','u32','i64','u64'])
def test_boolean_array_cannot_bind_numeric_buffer(tmp_path,dtype):
 from openassembler.runner import RunError
 with pytest.raises(RunError,match='Boolean array'):structural_plan(tmp_path,dtype=dtype,init='zeros')

def test_boolean_buffer_cannot_relabel_raw_ptx_pointer(tmp_path):
 from openassembler.runner import RunError
 with pytest.raises(RunError,match='logical source contract'):structural_plan(tmp_path,logical=False)

def test_boolean_array_requires_buffer_role(tmp_path):
 from openassembler.runner import RunError
 with pytest.raises(RunError):structural_plan(tmp_path,argument={'u64':0})

@pytest.mark.parametrize('count,passed',[(49152,True),(49153,False)])
def test_shared_boolean_byte_capacity_uses_existing_static_limit(tmp_path,count,passed):
 path=tmp_path/'k.py';path.write_text(HEADER+f'@kernel\ndef k(out:Array(bool)):\n    s=shared_array(bool,{count})\n    s[0]=True\n    out[0]=s[0]\n')
 if passed:
  text,_=compile_file(path);assert '.shared .align 1 .b8 __smem_s['+str(count)+'];'in text
 else:
  with pytest.raises(NotSupported,match='49152-byte'):compile_file(path)

@pytest.mark.parametrize('a_count,out_count,a_dtype,out_dtype,a_offset,out_offset,alias,passed',[
 (40,34,'bool','bool',0,0,False,True),
 (39,34,'bool','bool',0,0,False,False),
 (40,33,'bool','bool',0,0,False,False),
 (40,34,'u32','bool',0,0,False,False),
 (40,34,'bool','u32',0,0,False,False),
 (64,64,'bool','bool',0,40,True,False),
 (80,80,'bool','bool',0,40,True,True),
 (40,34,'bool','bool',1,0,False,False),
])
def test_boolean_checked_buffer_host_span_dtype_and_writable_alias(tmp_path,a_count,out_count,a_dtype,out_dtype,a_offset,out_offset,alias,passed):
 from tests.native_fixture import save_native_manifest
 from openassembler.runner import RunOptions,build_plan,RunError
 source=tmp_path/'original.py';source.write_text(HEADER+'@kernel\ndef k(a:Buffer(bool,(4,7),strides=(11,1),access="read"),out:Buffer(bool,(4,7),strides=(9,1))):\n    out[0,0]=a[0,0]\n')
 text,_=compile_file(source);module=parse_module(text);kernel=dict(name='k',params=module.kernels['k'].params,compiled_by='openassembler',logical_parameter_contract=parameter_contract_from_ptx(text,module.kernels['k'],allow_buffers=True),buffer_contract=contract_from_ptx(text,module.kernels['k']));(tmp_path/'k.ptx').write_text(text)
 save_native_manifest(tmp_path,tmp_path,'sm_80','opt',[dict(source='k.ptx',ptx='k.ptx',cubin='k.cubin',kernels=[kernel])]);path=tmp_path/'manifest.json';saved=json.loads(path.read_text());saved['entries'][0]['native_ownership']['ptx_sha256']=hashlib.sha256(text.encode()).hexdigest();path.write_text(json.dumps(saved))
 buffers={'a':dict(dtype=a_dtype,count=a_count,init='zeros')}
 if not alias:buffers['out']=dict(dtype=out_dtype,count=out_count,init='zeros')
 spec=tmp_path/'launch.json';spec.write_text(json.dumps(dict(buffers=buffers,steps=[dict(kernel='k',args=[dict(buf='a',offset=a_offset),dict(buf='a'if alias else'out',offset=out_offset)])],compare=['a'],reps=0)))
 if passed:assert build_plan(RunOptions(out_dir=tmp_path,spec=spec,reps=0))[0].jobs
 else:
  with pytest.raises(RunError):build_plan(RunOptions(out_dir=tmp_path,spec=spec,reps=0))

@pytest.mark.parametrize('tail,passed',[(49136,True),(49137,False)])
def test_shared_boolean_mixed_width_padding_counts_against_limit(tmp_path,tail,passed):
 path=tmp_path/'k.py';path.write_text(HEADER+f'@kernel\ndef k(out:Array(bool)):\n    prefix=shared_array(bool,3)\n    wide=shared_array(i64,1)\n    tail=shared_array(bool,{tail})\n    tail[0]=True\n    out[0]=tail[0]\n')
 if passed:compile_file(path)
 else:
  with pytest.raises(NotSupported,match='49152-byte'):compile_file(path)
