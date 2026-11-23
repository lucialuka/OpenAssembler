"""Independent CPU autograd controls; exact full tensors, no sampling."""
import copy
import pytest
torch = pytest.importorskip("torch")
from openassembler.modelfront.graph import capture_model,extract_weights
from openassembler.modelfront.kernels import prepare,emit_kernel
from openassembler.modelfront.reference import CPUReference
from openassembler.modelfront.optimization import optimize_plan

class Broadcast(torch.nn.Module):
    def forward(self,x,y):return (x+y)*y

class Attention(torch.nn.Module):
    def __init__(self,causal=False):super().__init__();self.causal=causal
    def forward(self,q,k,v,mask):
        return torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=None if self.causal else mask,is_causal=self.causal)

class TokenModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.emb=torch.nn.Embedding(11,8);self.out=torch.nn.Linear(8,11)
    def forward(self,tokens):
        x=self.emb(tokens)
        x=torch.nn.functional.scaled_dot_product_attention(x,x,x,is_causal=True)
        return self.out(x).reshape(-1,11)

@pytest.mark.parametrize('a,b',[((),(2,3)),((3,),(2,3)),((2,1,3),(1,4,1)),((1,3,1,5),(2,1,7,1))])
@pytest.mark.parametrize('views,reuse',[(False,False),(False,True),(True,False),(True,True)])
def test_broadcast(a,b,views,reuse):
    compare(Broadcast(),(torch.randn(a),torch.randn(b)),views,reuse)

@pytest.mark.parametrize('causal,width,qlen',[(False,1,1),(False,33,3),(False,513,2),(True,7,4)])
@pytest.mark.parametrize('views,reuse',[(False,False),(False,True),(True,False),(True,True)])
def test_attention(causal,width,qlen,views,reuse):
    mask=torch.rand(qlen,width)>.4;mask[0]=False
    compare(Attention(causal),(torch.randn(2,2,qlen,5),torch.randn(2,2,width,5),torch.randn(2,2,width,3),mask),views,reuse)

def compare(model,args,views=True,reuse=True):
    args=tuple(x.detach().clone().requires_grad_(x.dtype==torch.float32) for x in args)
    plan=capture_model(model,args);frozen=copy.deepcopy(plan)
    plan=optimize_plan(plan,storage_views=views,reuse_storage=reuse)
    prepare(plan);assert frozen==capture_model(model,args)
    ref=CPUReference(plan,extract_weights(model,plan));actual=ref.forward(args)
    expected=model(*args);name=plan['outputs'][0]
    torch.testing.assert_close(actual[name],expected.detach(),rtol=2e-5,atol=2e-6)
    seed=torch.randn_like(expected);expected.backward(seed)
    grads=ref.backward({name:seed})
    for spec,x in zip(plan['inputs'],args):
        if x.grad is not None:torch.testing.assert_close(grads['inputs'][spec['name']],x.grad,rtol=3e-5,atol=3e-6)
        else:assert spec['name'] not in grads['inputs']
    weights=dict(model.named_parameters())
    for spec in plan['parameters']:
        if weights[spec['target']].grad is not None:
            torch.testing.assert_close(grads['parameters'][spec['name']],weights[spec['target']].grad,rtol=3e-5,atol=3e-6)

@pytest.mark.parametrize('views,reuse',[(False,False),(False,True),(True,False),(True,True)])
def test_embedding_transformer(views,reuse):
    compare(TokenModel(),(torch.tensor([[0,1,1,10],[3,3,0,7]]),),views,reuse)

@pytest.mark.parametrize('op,inputs,out,attrs',[
 ('embedding',[{'shape':[2],'dtype':'f32'},{'shape':[4,3]}],{'shape':[2,3]},{}),
 ('embedding',[{'shape':[2],'dtype':'i64'},{'shape':[4,3]}],{'shape':[2,4]},{}),
 ('broadcast_add',[{'shape':[2,3]},{'shape':[2]}],{'shape':[2,3]},{}),
 ('sum_to',[{'shape':[2,3]}],{'shape':[3,2]},{}),
 ('masked_softmax',[{'shape':[2,3]},{'shape':[2,3]}],{'shape':[2,3]},{'causal':False}),
 ('ce_rows',[{'shape':[2,3]},{'shape':[2],'dtype':'f32'}],{'shape':[2]},{}),
 ('ce_grad',[{'shape':[2,1025]},{'shape':[2],'dtype':'i64'}],{'shape':[2,1025]},{}),
 ('adam_fused',[{'shape':[2]}]*4,{'shape':[2]},{'name':'adamw','amsgrad':True}),
])
def test_rejections(op,inputs,out,attrs):
    with pytest.raises(ValueError):emit_kernel(op,inputs,out,attrs)

@pytest.mark.parametrize('policy',[
    {'name':'adam','epsilon':0}, {'name':'adamw','beta1':1},
    {'name':'adam','beta2':-1}, {'name':'adamw','weight_decay':-1},
    {'name':'adam','epsilon':float('nan')}, {'name':'sgd','beta1':.9},
    {'name':'adam','amsgrad':1}, {'name':'adamw','maximize':'yes'},
])
def test_optimizer_policy_rejects_unsupported_settings(policy):
    from openassembler.modelfront.optimizer import optimizer_options
    with pytest.raises(ValueError):optimizer_options(policy)


def test_embedding_domain_cannot_be_removed_or_forged():
    plan=capture_model(TokenModel(),(torch.tensor([[0,1,1,10],[3,3,0,7]]),))
    assert plan['index_domains']==[{'name':'tokens','lower':0,'upper':11}]
    for mutation in ('remove','negative','upper'):
        bad=copy.deepcopy(plan)
        if mutation=='remove':del bad['index_domains']
        elif mutation=='negative':bad['index_domains'][0]['lower']=-1
        else:bad['index_domains'][0]['upper']=12
        with pytest.raises(ValueError,match='index domains'):prepare(bad)


def test_integer_buffer_metadata_cannot_disagree_with_input():
    plan=capture_model(TokenModel(),(torch.tensor([[0,1,1,10],[3,3,0,7]]),))
    plan['buffers']['tokens']['dtype']='f32'
    with pytest.raises(ValueError,match='metadata disagrees'):prepare(plan)


def test_checkpoint_validation_is_atomic_before_allocation():
    from openassembler.modelfront.runtime import Runtime
    from openassembler.modelfront.optimizer import optimizer_options
    runtime=Runtime.__new__(Runtime)
    runtime._device='cpu';runtime._tape=None;runtime._gradients=None;runtime.torch=torch
    runtime.plan={'optimizer':{'name':'adamw'},'backward':{'grad_parameters':{'p':'gp'}}}
    runtime.specs={'p':{'shape':[3],'dtype':'f32'}}
    runtime._optimizer_step=9;runtime._optimizer_state={'sentinel':object()}
    original=runtime._optimizer_state
    valid={'schema':1,'step':2,'options':optimizer_options({'name':'adamw'}),
           'moments':{'p':{'m':torch.zeros(3),'v':torch.ones(3)}}}
    def forbidden(*args,**kwargs):pytest.fail('invalid checkpoint allocated state')
    runtime._allocate=forbidden
    for kind in ('negative_v','nan','shape','missing','options','step'):
        bad=copy.deepcopy(valid)
        if kind=='negative_v':bad['moments']['p']['v'][0]=-1
        elif kind=='nan':bad['moments']['p']['m'][0]=float('nan')
        elif kind=='shape':bad['moments']['p']['m']=torch.zeros(2)
        elif kind=='missing':del bad['moments']['p']['v']
        elif kind=='options':bad['options']['epsilon']=1e-4
        else:bad['step']=-1
        with pytest.raises(ValueError):runtime.load_optimizer_state_dict(bad)
        assert runtime._optimizer_state is original and runtime._optimizer_step==9


def test_repo_integer_cross_entropy_optimizer_config(tmp_path):
    from openassembler.modelfront.frontend import load_config,ModelError
    (tmp_path/'model.py').write_text('def create_model(): pass\n')
    config='[model]\nentrypoint="model.py"\ninput_shapes=[[2,4]]\ninput_dtypes=["i64"]\ninput_ranges=[[0,11]]\nmode="training"\nloss="cross_entropy"\noptimizer="adamw"\noptimizer_fusion=true\n'
    (tmp_path/'openassembler.toml').write_text(config)
    options=load_config(tmp_path)
    assert options['optimizer_policy']['epsilon']==1e-8 and options['optimizer_fusion']
    assert options['input_dtypes']==['i64'] and options['loss']=='cross_entropy'
    for extra in ('execution="cuda_graph"\n','amsgrad=true\n','input_ranges=[[0,0]]\n'):
        candidate=config if not extra.startswith('input_ranges') else config.replace('input_ranges=[[0,11]]\n','')
        (tmp_path/'openassembler.toml').write_text(candidate+extra)
        with pytest.raises(ModelError):load_config(tmp_path)


def test_decode_session_owns_cache_and_rejects_stale_producer_or_weights():
    from types import SimpleNamespace
    from openassembler.modelfront.decode import DecodeSession
    class FakeRuntime:
        def __init__(self,length,prefill=False):
            self.plan={'parameters':[{'name':'parameter:w','target':'w','shape':[1]}],
                       'outputs':['prediction','keys','values'],'inputs':[] if prefill else [
                           {'name':n,'shape':[1,length,2],'dtype':'f32'} for n in ('past_k','past_v')]}
            self.specs={'keys':{'shape':[1,length+1,2]},'values':{'shape':[1,length+1,2]}}
            self.weights={'parameter:w':torch.ones(1)};self._forward_epoch=0;self.closed=False
            self.length=length;self.is_prefill=prefill
        def forward(self,inputs):
            self._forward_epoch+=1
            n=self.length if self.is_prefill else self.length+1
            return {'prediction':torch.zeros(1,1,2),'keys':torch.zeros(1,n,2),'values':torch.zeros(1,n,2)}
        def close(self):self.closed=True
    prefill=FakeRuntime(2,True);bucket=FakeRuntime(2);session=DecodeSession(prefill,{2:bucket})
    assert bucket.weights['parameter:w'] is prefill.weights['parameter:w']
    with pytest.raises(ValueError,match='prefill'):session.step({'x':torch.zeros(1)})
    session.prefill({});cache=session.cache
    with pytest.raises(ValueError,match='owns cache'):session.step({'past_k':cache[0]})
    assert session.cache is cache
    prefill.forward({})
    with pytest.raises(ValueError,match='producer'):session.step({})
    session.reset();session.prefill({});prefill.weights['parameter:w'].add_(1)
    with pytest.raises(ValueError,match='weights changed'):session.step({})
    session.reset();session.prefill({});session.cache[0].add_(1)
    with pytest.raises(ValueError,match='cache was mutated'):session.step({})
    session.reset();session.prefill({});session.step({});cache=session.cache
    with pytest.raises(ValueError,match='past length 3'):session.step({})
    assert session.cache is cache
    session.close();assert prefill.closed and bucket.closed and session.cache is None
