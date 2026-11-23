"""Extended optimizer policies, exact emitted contracts and snapshot rejection."""
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import pytest

torch=pytest.importorskip('torch')
from openassembler.modelfront.optimizer import optimizer_options,optimizer_state_keys,torch_optimizer_kwargs
from openassembler.modelfront.graph import capture_model,extract_weights
from openassembler.modelfront.kernels import prepare,emit_kernel
from openassembler.modelfront.reference import CPUReference

POLICIES=[{'name':'sgd','weight_decay':.03,'maximize':True},
 {'name':'sgd','momentum':.8,'dampening':.25,'weight_decay':.03},
 {'name':'sgd','momentum':.9,'nesterov':True,'maximize':True,'weight_decay':.02},
 {'name':'adam','amsgrad':True,'maximize':True,'beta1':.8,'beta2':.95,'weight_decay':.03},
 {'name':'adamw','amsgrad':True,'beta1':.8,'beta2':.95,'weight_decay':.03},
 {'name':'adamw','maximize':True}]

class Point(torch.nn.Module):
    def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.linspace(-.5,.5,7))
    def forward(self,x):return self.weight+x

@pytest.mark.parametrize('policy',POLICIES)
def test_complete_reference_optimizer_progression(policy):
    torch.manual_seed(34052);model=Point();plan=capture_model(model,(torch.zeros(7),));plan['optimizer']=optimizer_options(policy)
    prepare(plan);reference=CPUReference(plan,extract_weights(model,plan));p=optimizer_options(policy)
    optimizer={'sgd':torch.optim.SGD,'adam':torch.optim.Adam,'adamw':torch.optim.AdamW}[p['name']](model.parameters(),lr=.001,foreach=False,**torch_optimizer_kwargs(p))
    for step in range(12):
        x=torch.randn(7)*.1;target=torch.randn(7)*.1
        optimizer.zero_grad();loss=((model(x)-target)**2).mean();loss.backward();optimizer.step()
        actual=reference.train_step({'x':x},target,lr=.001)
        torch.testing.assert_close(actual['loss'],loss)
        torch.testing.assert_close(reference.weights['parameter:weight'],model.weight)
        expected_keys=({'momentum_buffer'} if p.get('momentum',0) else set()) if p['name']=='sgd' else (
            {'step','exp_avg','exp_avg_sq'} | ({'max_exp_avg_sq'} if p.get('amsgrad',False) else set()))
        assert set(optimizer.state[model.weight])==expected_keys

@pytest.mark.parametrize('policy',[
 {'name':'sgd','momentum':-1}, {'name':'sgd','dampening':float('nan')},
 {'name':'sgd','weight_decay':-1}, {'name':'sgd','nesterov':True},
 {'name':'sgd','momentum':.9,'nesterov':True,'dampening':.1},
 {'name':'sgd','nesterov':1}, {'name':'adam','amsgrad':1},
 {'name':'adamw','maximize':'yes'}, {'name':'sgd','amsgrad':True},
 {'name':'adam','momentum':.9}, {'name':'adamw','capturable':True},
])
def test_invalid_optimizer_policies_reject(policy):
    with pytest.raises(ValueError):optimizer_options(policy)


def test_default_normalization_and_plain_sgd_source_identity():
    assert optimizer_options()==optimizer_options({'name':'sgd','momentum':0.,'dampening':0.,'weight_decay':0.,'nesterov':False,'maximize':False})=={'name':'sgd'}
    for kind in ('adam','adamw'):
        assert optimizer_options({'name':kind})==optimizer_options({'name':kind,'amsgrad':False,'maximize':False})
    records=json.loads((Path(__file__).resolve().parents[1]/'compiler_exp/gap_closure/optimizers/plain-sgd-before.json').read_text())
    for shape in ([],[3],[129],[2,7]):
        kernel=emit_kernel('sgd',[{'shape':shape}]*2,{'shape':shape})
        assert records[str(shape)]=={'name':kernel.name,'ptx_sha256':hashlib.sha256(kernel.ptx.encode()).hexdigest()}

@pytest.mark.parametrize('policy',POLICIES)
def test_extended_policies_reject_cuda_graph_before_runtime(policy,tmp_path):
    from openassembler.modelfront.frontend import load_config,ModelError
    from openassembler.modelfront.replay import TrainingReplay
    (tmp_path/'model.py').write_text('def create_model():pass\n')
    def toml(v):return str(v).lower() if type(v)is bool else '"'+v+'"' if isinstance(v,str) else str(v)
    text='[model]\nentrypoint="model.py"\ninput_shapes=[[7]]\nmode="training"\nexecution="cuda_graph"\n'
    text+=''.join(('optimizer' if k=='name' else k)+'='+toml(v)+'\n' for k,v in policy.items())
    (tmp_path/'openassembler.toml').write_text(text)
    with pytest.raises(ModelError,match='require direct'):load_config(tmp_path)
    with pytest.raises(ValueError,match='require direct'):TrainingReplay({'optimizer':policy},{},tmp_path,{},None)


@pytest.mark.parametrize('policy',POLICIES)
def test_explicit_cli_optimizer_options_are_preserved(policy,tmp_path):
    from openassembler.modelfront.frontend import load_config
    (tmp_path/'model.py').write_text('def create_model():pass\n')
    def toml(v):return str(v).lower() if type(v)is bool else '"'+v+'"' if isinstance(v,str) else str(v)
    text='[model]\nentrypoint="model.py"\ninput_shapes=[[7]]\nmode="training"\n'
    text+=''.join(('optimizer' if k=='name' else k)+'='+toml(v)+'\n' for k,v in policy.items())
    (tmp_path/'openassembler.toml').write_text(text)
    assert load_config(tmp_path)['optimizer_policy']==optimizer_options(policy)


def test_module_compile_api_preserves_extended_sgd_policy(monkeypatch,tmp_path):
    from openassembler.modelfront.autograd import compile_model
    import openassembler.modelfront.runtime as runtime
    captured=[]
    def build(plan,weights,*args,**kwargs):captured.append(copy.deepcopy(plan));prepare(plan)
    monkeypatch.setattr(runtime,'build_bundle',build)
    policy={'name':'sgd','momentum':.8,'dampening':.25,'weight_decay':.03,'maximize':True}
    compiled=compile_model(Point(),(torch.zeros(7),),tmp_path,optimizer=policy)
    assert compiled.plan['optimizer']==captured[0]['optimizer']==optimizer_options(policy)
    compiled.close()


def test_amsgrad_fusion_rejects_explicitly():
    plan=capture_model(Point(),(torch.zeros(7),));plan.update(optimizer={'name':'adamw','amsgrad':True},optimizer_fusion=True)
    with pytest.raises(ValueError,match='AMSGrad requires split'):prepare(plan)

@pytest.mark.parametrize('policy',[
 {'name':'sgd','momentum':.9}, {'name':'adamw','amsgrad':True},
])
def test_complete_snapshot_validation_precedes_allocation(policy):
    from openassembler.modelfront.runtime import Runtime
    runtime=Runtime.__new__(Runtime);runtime._device='cpu';runtime._tape=None;runtime._gradients=None;runtime.torch=torch
    runtime.plan={'optimizer':policy,'backward':{'grad_parameters':{'p':'gp'}}};runtime.specs={'p':{'shape':[3],'dtype':'f32'}}
    runtime._optimizer_step=9;runtime._optimizer_state={'sentinel':object()};original=runtime._optimizer_state
    p=optimizer_options(policy)
    state={'schema':1,'step':2,'options':p,'moments':{'p':{k:torch.ones(3) for k in optimizer_state_keys(p)}}}
    runtime._allocate=lambda *a,**k:pytest.fail('invalid snapshot allocated state')
    for variant in ('missing','extra','nan','wrong_shape','non_tensor','step','options','incomplete_max'):
        bad=copy.deepcopy(state)
        key=next(iter(bad['moments']['p']))
        if variant=='missing':del bad['moments']['p'][key]
        elif variant=='extra':bad['moments']['p']['extra']=torch.ones(3)
        elif variant=='nan':bad['moments']['p'][key][0]=float('nan')
        elif variant=='wrong_shape':bad['moments']['p'][key]=torch.ones(2)
        elif variant=='non_tensor':bad['moments']['p'][key]=[1,2,3]
        elif variant=='step':bad['step']=-1
        elif variant=='options':bad['options']={'name':'sgd'}
        else:
            if 'max_v' not in bad['moments']['p']:continue
            bad['moments']['p']['max_v'][0]=.5
        with pytest.raises(ValueError):runtime.load_optimizer_state_dict(bad)
        assert runtime._optimizer_state is original and runtime._optimizer_step==9


def execute(kernel,arrays,scalars):
    from backend_harness import compile_ptx,listing
    from openassembler.emu.machine import Memory,Machine,Launch
    from openassembler.emu.program import load_listing
    code,stats,words=compile_ptx(kernel.ptx,kernel.name);assert words
    program=load_listing(listing(code,stats['labels']));outputs=[]
    for poison in (None,0xdeadbeef):
        memory=Memory();pointers=[]
        for i,values in enumerate(arrays):
            pointer=0x100000000+i*4096;pointers.append(pointer)
            memory.write_bytes(pointer,struct.pack('<'+'f'*len(values),*values))
        output=0x100100000;pointers.append(output)
        params=struct.pack('<'+'Q'*len(pointers),*pointers)+b''.join(struct.pack('<f',v) for v in scalars)
        for cx in range(kernel.grid[0]):Machine(program,Launch(grid=kernel.grid,block=(128,1,1),ctaid=(cx,0,0),params=params,poison=poison),memory).run()
        outputs.append(torch.tensor([struct.unpack('<f',struct.pack('<I',memory.read(output+i*4,4)))[0] for i in range(math.prod(kernel.output))]))
    assert torch.equal(outputs[0].view(torch.int32),outputs[1].view(torch.int32));return outputs[0]

@pytest.mark.parametrize('policy',POLICIES[:3])
def test_sgd_native_kernels_encode_and_match_independent_recurrence(policy):
    from importlib.util import spec_from_file_location,module_from_spec
    path=Path(__file__).resolve().parents[1]/'compiler_exp/gap_closure/optimizers/candidate.py'
    spec=spec_from_file_location('optimizer_candidate',path);candidate=module_from_spec(spec);spec.loader.exec_module(candidate)
    p=optimizer_options(policy);shape={'shape':[7]};w=torch.linspace(-.5,.5,7);state={}
    for step in range(1,5):
        gradient=torch.tensor([-.3,.2,0.,1e-6,-1e-6,.1,-.2])*(-1)**step;lr=.001
        expected,state=candidate.recurrence(w,gradient,state,p,step,lr)
        args=[w,gradient]
        if p.get('momentum',0):
            if step==1:buffer=execute(emit_kernel('sgd_buffer_init',[shape]*2,shape,p),[gradient.tolist(),w.tolist()],[])
            else:buffer=execute(emit_kernel('sgd_buffer',[shape]*3,shape,p),[buffer.tolist(),gradient.tolist(),w.tolist()],[])
            torch.testing.assert_close(buffer,state['momentum'],rtol=4e-5,atol=5e-6);args.append(buffer)
        w=execute(emit_kernel('sgd_update',[shape]*len(args),shape,p),[x.tolist() for x in args],[lr])
        torch.testing.assert_close(w,expected,rtol=4e-5,atol=5e-6)


def test_amsgrad_maximum_nan_classification_and_signed_zero_ties():
    p={'name':'adamw','amsgrad':True};shape={'shape':[8]}
    left=[float('nan'),1.,float('nan'),0.,-0.,-0.,.3,.2]
    right=[1.,float('nan'),float('nan'),-0.,0.,-0.,.2,.3]
    actual=execute(emit_kernel('adam_max',[shape]*2,shape,p),[left,right],[])
    expected=torch.maximum(torch.tensor(left),torch.tensor(right))
    assert torch.equal(torch.isnan(actual),torch.isnan(expected))
    ordinary=~torch.isnan(expected)&(expected!=0)
    assert torch.equal(actual[ordinary].view(torch.int32),expected[ordinary].view(torch.int32))
    # Torch signed-zero ties vary by CPU scalar/vector implementation; require
    # the native contract's deterministic left operand on equality explicitly.
    zeros=torch.tensor(left)[expected==0]
    assert torch.equal(actual[expected==0].view(torch.int32),zeros.view(torch.int32))
