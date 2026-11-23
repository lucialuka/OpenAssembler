"""Complete checkpoint format/identity/failure atomicity, independently of GPUs."""
from contextlib import nullcontext
import copy
import io
import json
from pathlib import Path
import random
from types import SimpleNamespace,MethodType
import zipfile
import pytest

torch=pytest.importorskip('torch')
from openassembler.modelfront import checkpoint
from openassembler.modelfront.optimizer import optimizer_options
from openassembler.modelfront.runtime import Runtime

class CPUTorch:
    """Only the boundary's CUDA-stream metadata is stubbed for format tests."""
    cuda=SimpleNamespace(current_stream=lambda device:SimpleNamespace(cuda_stream=17))
    def __getattr__(self,name):return getattr(torch,name)


def runtime(policy=None):
    policy=optimizer_options(policy or {'name':'adamw','amsgrad':True})
    specs={'p':{'name':'p','shape':[3],'dtype':'f32'},'b':{'name':'b','shape':[1],'dtype':'f32','requires_grad':False}}
    state={'p':{'m':torch.tensor([.1,-.2,0.]),'v':torch.tensor([.2,.3,0.]),'max_v':torch.tensor([.3,.3,.1])}}
    if policy['name']=='sgd':state={'p':{'momentum':torch.tensor([.1,-.2,0.])}}
    r=SimpleNamespace(torch=CPUTorch(),plan={'schema':1,'arch':'sm_80','parameters':list(specs.values()),'optimizer':policy,'backward':{'grad_parameters':{'p':'gp'}}},
       specs=specs,ledger={'source_sha256':'a'*64,'cubin_sha256':'b'*64},weights={'p':torch.tensor([1.,-2.,3.]),'b':torch.tensor([.25])},
       _closed=False,_device='cpu',_stream_id=17,_tape=None,_gradients=None,_backward_ready=False,
       _optimizer_step=12,_optimizer_state=state,_allocations=[],_guards=[],_last=None,_saved_versions={},_packed_ready=set(),_forward_epoch=3)
    r.synchronize=lambda:None;r._validate_optimizer_state_dict=MethodType(Runtime._validate_optimizer_state_dict,r)
    r.optimizer_state_dict=MethodType(Runtime.optimizer_state_dict,r)
    return r


def entries(path):
    with zipfile.ZipFile(path) as z:return {n:z.read(n) for n in z.namelist()}


def write(path,files):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_STORED) as z:
        for n,data in files.items():z.writestr(n,data)


def fingerprint(r):
    return ({n:v.clone() for n,v in r.weights.items()},copy.deepcopy(r._optimizer_state),r._optimizer_step,random.getstate(),torch.get_rng_state().clone())


def unchanged(r,before):
    w,s,step,py,tr=before
    assert r._optimizer_step==step and py==random.getstate() and torch.equal(tr,torch.get_rng_state())
    for n,v in w.items():assert torch.equal(r.weights[n],v)
    for n,pair in s.items():
        for k,v in pair.items():assert torch.equal(r._optimizer_state[n][k],v)

@pytest.mark.parametrize('policy',[{'name':'sgd','momentum':.9},{'name':'adamw','amsgrad':True}])
def test_complete_format_owns_all_model_buffer_optimizer_and_host_state(tmp_path,policy):
    r=runtime(policy);path=tmp_path/'checkpoint.oa';random.seed(35052);torch.manual_seed(35052)
    application={'cursor':12,'scheduler':{'last_epoch':12,'rate':.001},'scaler':None}
    before=fingerprint(r);checkpoint.save_checkpoint(r,path,application=application)
    actual=checkpoint.read_checkpoint(r,path)
    assert actual['application']==application and actual['optimizer']['step']==12
    assert set(actual['weights'])=={'p','b'}
    for n,v in r.weights.items():assert torch.equal(actual['weights'][n],v)
    for n,pair in r._optimizer_state.items():
        for k,v in pair.items():assert torch.equal(actual['optimizer']['moments'][n][k],v)
    first=path.read_bytes();checkpoint.save_checkpoint(r,path,application=application)
    assert path.read_bytes()==first
    application['cursor']=999;actual['weights']['p'][0]=999
    unchanged(r,before)

@pytest.mark.parametrize('change',[
 'version','program','missing_buffer','bad_hash','missing_state','negative_variance','maximum_below_v',
 'unknown_file','unknown_metadata','path_escape','wrong_dtype','wrong_shape','bad_rng','duplicate_json',
])
def test_invalid_checkpoints_leave_all_owned_state_and_rng_unchanged(tmp_path,change):
    r=runtime();path=tmp_path/'good';checkpoint.save_checkpoint(r,path,application={'cursor':12});files=entries(path)
    manifest=json.loads(files['manifest.json']);before=fingerprint(r)
    if change=='version':manifest['version']=2
    elif change=='program':manifest['program']['cubin_sha256']='c'*64
    elif change=='missing_buffer':del manifest['model']['b']
    elif change=='bad_hash':manifest['model']['p']['sha256']='0'*64
    elif change=='missing_state':del manifest['optimizer']['moments']['p']['max_v']
    elif change in ('negative_variance','maximum_below_v'):
        entry=manifest['optimizer']['moments']['p']['v' if change=='negative_variance' else 'max_v']
        import array,hashlib
        raw=array.array('f',[-1,1,1] if change=='negative_variance' else [0,0,0]).tobytes()
        files[entry['path']]=raw;entry['sha256']=hashlib.sha256(raw).hexdigest()
    elif change=='unknown_file':files['host.py']=b'raise RuntimeError("must never execute")'
    elif change=='unknown_metadata':manifest['callback']='arbitrary host code'
    elif change=='path_escape':manifest['model']['p']['path']='../outside'
    elif change=='wrong_dtype':manifest['model']['p']['dtype']='f64'
    elif change=='wrong_shape':manifest['model']['p']['shape']=[True,3]
    elif change=='bad_rng':manifest['rng']['python'][1][-1]=999
    files['manifest.json']=json.dumps(manifest).encode()
    if change=='duplicate_json':files['manifest.json']=files['manifest.json'].replace(b'"version": 1',b'"version": 1, "version": 1')
    bad=tmp_path/'bad';write(bad,files)
    with pytest.raises(ValueError):checkpoint.read_checkpoint(r,bad)
    unchanged(r,before)


def test_invalid_save_and_failed_atomic_replace_preserve_existing_checkpoint(tmp_path,monkeypatch):
    r=runtime();path=tmp_path/'state';checkpoint.save_checkpoint(r,path);old=path.read_bytes()
    with pytest.raises(ValueError):checkpoint.save_checkpoint(r,path,application={'rate':float('nan')})
    assert path.read_bytes()==old
    def fail(*args):raise OSError('injected publication failure')
    monkeypatch.setattr(checkpoint.os,'replace',fail)
    with pytest.raises(OSError,match='injected'):checkpoint.save_checkpoint(r,path,application={'cursor':13})
    assert path.read_bytes()==old and not list(tmp_path.glob('*.tmp'))

@pytest.mark.parametrize('flag',['_tape','_gradients','_backward_ready','_checkpoint_replay_owner','_closed'])
def test_checkpoint_rejects_partial_training_or_replay_boundary(tmp_path,flag):
    r=runtime();setattr(r,flag,True)
    with pytest.raises(ValueError,match='boundary'):checkpoint.save_checkpoint(r,tmp_path/'checkpoint')


def test_rng_omission_requires_explicit_nonrestoring_load(tmp_path):
    r=runtime();path=tmp_path/'checkpoint';checkpoint.save_checkpoint(r,path,include_rng=False)
    with pytest.raises(ValueError,match='no CPU RNG'):checkpoint.read_checkpoint(r,path)
    assert checkpoint.read_checkpoint(r,path,restore_rng=False)['rng'] is None


def test_commit_rng_failure_rolls_back_before_any_bindings_change(monkeypatch):
    r=runtime();before=fingerprint(r)
    staged=checkpoint.StagedCheckpoint({'p':torch.ones(3),'b':torch.ones(1)}, {},99,[],[],{'cursor':99},(random.Random(1).getstate(),torch.Generator().manual_seed(1).get_state()))
    original=torch.set_rng_state;calls=0
    def fail_once(state):
        nonlocal calls
        calls+=1
        if calls==1:raise RuntimeError('injected RNG commit failure')
        return original(state)
    monkeypatch.setattr(torch,'set_rng_state',fail_once)
    with pytest.raises(RuntimeError,match='injected'):checkpoint.commit_checkpoint(r,staged)
    unchanged(r,before)


def test_complete_checkpoint_preserves_owned_ieee_bits_including_nan_payloads(tmp_path):
    r=runtime()
    raw=torch.tensor([0x7fc00123,-0x3ffedd, -2147483648],dtype=torch.int32).view(torch.float32)
    r.weights['p']=raw.clone()
    r._optimizer_state['p']['m']=raw.clone()
    r._optimizer_state['p']['v']=torch.tensor([float('nan'),float('inf'),-0.])
    r._optimizer_state['p']['max_v']=torch.tensor([float('nan'),float('inf'),0.])
    path=tmp_path/'ieee-checkpoint';checkpoint.save_checkpoint(r,path)
    restored=checkpoint.read_checkpoint(r,path)
    assert torch.equal(restored['weights']['p'].view(torch.int32),r.weights['p'].view(torch.int32))
    for key,value in r._optimizer_state['p'].items():
        assert torch.equal(restored['optimizer']['moments']['p'][key].view(torch.int32),value.view(torch.int32))


def test_module_checkpoint_refreshes_replaced_registered_parameters_and_buffers(tmp_path):
    from openassembler.modelfront.autograd import CompiledModel
    r=runtime();r.plan['inputs']=[]
    owner=CompiledModel(r.plan,r.weights,tmp_path);r.weights=owner.native_parameters()
    r.save_checkpoint=MethodType(Runtime.save_checkpoint,r);owner._native_runtime=r
    r.close=lambda:None
    state=owner.state_dict();state['_oa_state_0']=torch.full((3,),99.);state['_oa_state_1']=torch.full((1,),.5)
    owner.load_state_dict(state,assign=True)
    path=tmp_path/'current-bindings';owner.save_native_checkpoint(path,include_rng=False)
    saved=checkpoint.read_checkpoint(r,path,restore_rng=False)
    for name,value in owner.native_parameters().items():assert torch.equal(saved['weights'][name],value)
    assert r.weights==owner.native_parameters()
    owner.close()


def test_repeated_checkpoint_commit_retires_only_superseded_owned_backings():
    r=runtime();persistent=torch.empty(9);old=torch.empty(7);old_guard=('old','old-end');persistent_guard=('work','work-end')
    r._allocations=[persistent,old];r._guards=[persistent_guard,old_guard];r._checkpoint_backings=[old]
    first=torch.empty(7);staged=checkpoint.StagedCheckpoint({'p':first[4:].view(3),'b':torch.ones(1)}, {},12,[first],[('new','new-end')],{},None)
    checkpoint.commit_checkpoint(r,staged)
    assert len(r._allocations)==2 and r._allocations[0] is persistent and r._allocations[1] is first
    assert r._guards==[persistent_guard,('new','new-end')]
    second=torch.empty(7);staged=checkpoint.StagedCheckpoint({'p':second[4:].view(3),'b':torch.ones(1)}, {},12,[second],[('new2','new2-end')],{},None)
    checkpoint.commit_checkpoint(r,staged)
    assert len(r._allocations)==2 and r._allocations[0] is persistent and r._allocations[1] is second
    assert r._guards==[persistent_guard,('new2','new2-end')]
    assert old.numel()==7 and first.numel()==7 # caller-held storage stays alive
