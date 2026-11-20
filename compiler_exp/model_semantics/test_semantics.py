"""Independent CPU autograd controls; exact full tensors, no sampling."""
import copy
import pytest
import torch
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
 ('adam_update',[{'shape':[2]}]*4,{'shape':[2]},{'name':'adamw','amsgrad':True}),
])
def test_rejections(op,inputs,out,attrs):
    with pytest.raises(ValueError):emit_kernel(op,inputs,out,attrs)
