"""FP32 reductions and sigmoid against eager autograd and native instruction paths."""
import copy
import math
import struct
import pytest

torch = pytest.importorskip('torch')
from openassembler.modelfront.graph import capture_model,extract_weights,ModelGraphError
from openassembler.modelfront.kernels import prepare,emit_kernel
from openassembler.modelfront.optimization import optimize_plan
from openassembler.modelfront.reference import CPUReference

CASES=[((),None,False),((3,),None,True),((2,3),0,False),((2,3),-1,True),
       ((2,3,5),(0,2),False),((2,3,5),(-1,0),True),((2,1,3,5),(),False),
       ((2,1,3,5),(1,3),False),((2,3,5),[2,0],True),
       ((),0,False),((),-1,True),((2,3),None,False)]

class Reduction(torch.nn.Module):
    def __init__(self,dim,keepdim,kind,method):
        super().__init__();self.dim=dim;self.keepdim=keepdim;self.kind=kind;self.method=method
    def forward(self,x):
        if self.method:return getattr(x,self.kind)(dim=self.dim,keepdim=self.keepdim)
        return getattr(torch,self.kind)(x,dim=self.dim,keepdim=self.keepdim,dtype=torch.float32)


def compare(model,args,views,reuse):
    args=tuple(x.detach().clone().requires_grad_() for x in args)
    plan=optimize_plan(capture_model(model,args),storage_views=views,reuse_storage=reuse)
    prepare(plan);ref=CPUReference(plan,extract_weights(model,plan))
    actual=ref.forward(args);expected=model(*args);name=plan['outputs'][0]
    torch.testing.assert_close(actual[name],expected.detach(),rtol=2e-5,atol=3e-6)
    seed=torch.randn_like(expected);expected.backward(seed)
    grads=ref.backward({name:seed})
    for spec,x in zip(plan['inputs'],args):
        torch.testing.assert_close(grads['inputs'][spec['name']],x.grad,rtol=2e-5,atol=3e-6)
    for spec in plan['parameters']:
        param=dict(model.named_parameters())[spec['target']]
        torch.testing.assert_close(grads['parameters'][spec['name']],param.grad,rtol=2e-5,atol=3e-6)
    return plan

@pytest.mark.parametrize('shape,dim,keepdim',CASES)
@pytest.mark.parametrize('kind',('sum','mean'))
@pytest.mark.parametrize('method',(False,True))
@pytest.mark.parametrize('views,reuse',((False,False),(False,True),(True,False),(True,True)))
def test_reduction_axes_and_backward(shape,dim,keepdim,kind,method,views,reuse):
    torch.manual_seed(21052)
    compare(Reduction(dim,keepdim,kind,method),(torch.randn(shape),),views,reuse)

@pytest.mark.parametrize('route',('torch','module','tensor','functional'))
@pytest.mark.parametrize('views,reuse',((False,False),(False,True),(True,False),(True,True)))
def test_sigmoid_entrypoints_and_extreme_inputs(route,views,reuse):
    class Activation(torch.nn.Module):
        def __init__(self):super().__init__();self.activation=torch.nn.Sigmoid()
        def forward(self,x):
            if route=='module':return self.activation(x)
            if route=='torch':return torch.sigmoid(x)
            if route=='functional':return torch.nn.functional.sigmoid(x)
            return x.sigmoid()
    x=torch.tensor([-1e30,-1000,-100,-88,-20,-1.,-0.,0.,1.,20.,88.,100.,1000.,1e30])
    compare(Activation(),(x,),views,reuse)

@pytest.mark.parametrize('views,reuse',((False,False),(False,True),(True,False),(True,True)))
def test_sigmoid_reduction_shared_parameters_and_multioutput(views,reuse):
    class Network(torch.nn.Module):
        def __init__(self):super().__init__();self.bias=torch.nn.Parameter(torch.randn(5)*.25)
        def forward(self,x):
            y=torch.sigmoid(x+self.bias)
            return y.sum(dim=(0,2)),y.mean(dim=1,keepdim=True)
    torch.manual_seed(21052);model=Network();x=torch.randn(2,3,5,requires_grad=True)
    plan=optimize_plan(capture_model(model,(x,)),storage_views=views,reuse_storage=reuse)
    prepare(plan);reference=CPUReference(plan,extract_weights(model,plan));actual=reference.forward((x,))
    expected=model(x);seeds=tuple(torch.randn_like(y) for y in expected)
    torch.autograd.backward(expected,seeds)
    gradients=reference.backward(dict(zip(plan['outputs'],seeds)))
    for name,value in zip(plan['outputs'],expected):torch.testing.assert_close(actual[name],value)
    torch.testing.assert_close(gradients['inputs']['x'],x.grad)
    torch.testing.assert_close(gradients['parameters']['parameter:bias'],model.bias.grad)

@pytest.mark.parametrize('expression',[
    'x.sum(dtype=torch.float64)', 'x.mean(dtype=torch.float64)',
    'torch.sum(x,dim=0,out=out)', 'torch.mean(x,dim=0,out=out)',
    'torch.sigmoid(x,out=x)', 'x.sigmoid_()',
])
def test_dtype_and_mutation_rejections(expression):
    class Bad(torch.nn.Module):
        def forward(self,x,out):return eval(expression,{'torch':torch},{'x':x,'out':out})
    with pytest.raises(ModelGraphError):
        capture_model(Bad(),(torch.ones(2,3),torch.zeros(3)))

@pytest.mark.parametrize('kind',('sum','mean','sigmoid'))
def test_integer_input_does_not_silently_convert(kind):
    class Bad(torch.nn.Module):
        def forward(self,x):return getattr(torch,kind)(x)
    with pytest.raises(ModelGraphError):capture_model(Bad(),(torch.ones(3,dtype=torch.int64),))


def test_rank_five_and_empty_reductions_reject():
    with pytest.raises(ModelGraphError,match='rank zero through four'):
        capture_model(Reduction(None,False,'sum',True),(torch.ones(1,2,1,2,3),))
    with pytest.raises(ModelGraphError,match='positive-shape'):
        capture_model(Reduction(None,False,'sum',True),(torch.ones(0,3),))


@pytest.mark.parametrize('kind',('sigmoid','sigmoid_backward'))
def test_sigmoid_emitted_ptx_encodes_and_emulates_against_independent_math(kind):
    from backend_harness import compile_ptx,listing
    from openassembler.emu.program import load_listing
    from openassembler.emu.machine import Machine,Memory,Launch
    x=[-1e30,-1000,-100,-88,-20,-1.,-0.,0.,1.,20.,88.,100.,1000.,1e30]+[i*.125-8 for i in range(129)]
    expected=[(1/(1+math.exp(-v)) if v>=0 else math.exp(v)/(1+math.exp(v))) for v in x]
    values=x
    if kind=='sigmoid_backward':
        values=[float(i%7-3)*.25 for i in range(len(x))]
        expected=[dy*y*(1-y) for dy,y in zip(values,expected)]
    inputs=[{'shape':[len(x)]}]*(2 if kind.endswith('backward') else 1)
    kernel=emit_kernel(kind,inputs,{'shape':[len(x)]})
    code,stats,words=compile_ptx(kernel.ptx,kernel.name);assert words
    program=load_listing(listing(code,stats['labels']))
    for poison in (None,0xdeadbeef):
        memory=Memory();base=0x100000000;output=base+4096;second=base+8192
        memory.write_bytes(base,struct.pack('<'+'f'*len(x),*values))
        if len(inputs)==2:
            sigmoid=[(1/(1+math.exp(-v)) if v>=0 else math.exp(v)/(1+math.exp(v))) for v in x]
            memory.write_bytes(second,struct.pack('<'+'f'*len(x),*sigmoid))
        pointers=(base,second,output) if len(inputs)==2 else (base,output)
        params=struct.pack('<'+'Q'*len(pointers),*pointers)
        for cx in range(kernel.grid[0]):
            Machine(program,Launch(block=(128,1,1),grid=kernel.grid,ctaid=(cx,0,0),params=params,poison=poison),memory).run()
        actual=[struct.unpack('<f',struct.pack('<I',memory.read(output+i*4,4)))[0] for i in range(len(x))]
        for a,b in zip(actual,expected):assert math.isclose(a,b,rel_tol=3e-6,abs_tol=2e-7),(a,b)


def test_sigmoid_bad_kernel_contracts_reject():
    for inputs,output,attrs in [([{'shape':[3],'dtype':'i64'}],{'shape':[3]},{}),
                                ([{'shape':[3]}],{'shape':[2]},{}),
                                ([{'shape':[3]}],{'shape':[3]},{'out':True})]:
        with pytest.raises(ValueError):emit_kernel('sigmoid',inputs,output,attrs)


@pytest.mark.parametrize('source,target',[((2,3),(1,3)),((2,3),(2,1)),((2,3,5),(1,3,1)),
                                        ((2,1,3,5),(2,1,1,5)),((2,3),()),((),())])
def test_reduction_kernel_encodes_and_matches_ptx_with_poisoned_registers(source,target):
    from backend_harness import compile_and_diff,pack_f32
    kernel=emit_kernel('sum_to',[{'shape':list(source)}],{'shape':list(target)})
    values=[(i%13-6)*.125 for i in range(math.prod(source))]
    _,_,bad=compile_and_diff(kernel.ptx,kernel.name,
                            {'a0':pack_f32(values),'output':bytes(math.prod(target)*4)}, {},
                            block=(kernel.threads,1,1),grid=kernel.grid)
    assert not bad
