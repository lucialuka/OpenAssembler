"""G25 normalization against eager autograd and native PTX/SASS contracts."""
import math
import struct
import pytest

torch = pytest.importorskip('torch')
from openassembler.modelfront.graph import capture_model, extract_weights, ModelGraphError
from openassembler.modelfront.kernels import prepare, emit_kernel
from openassembler.modelfront.optimization import optimize_plan
from openassembler.modelfront.reference import CPUReference

CASES = [
    ((2,3),(3,5)), ((2,3),(4,3,5)), ((4,2,3),(3,5)),
    ((1,2,3),(4,3,5)), ((4,2,3),(1,3,5)),
    ((2,1,2,3),(1,4,3,5)), ((2,4,2,3),(4,3,5)),
    ((2,3),(2,4,3,5)),
]

class Product(torch.nn.Module):
    def forward(self,a,b):return a @ b

@pytest.mark.parametrize('ashape,bshape',CASES)
@pytest.mark.parametrize('views,reuse',[(False,False),(False,True),(True,False),(True,True)])
def test_matmul_batch_broadcast_matches_full_eager_derivatives(ashape,bshape,views,reuse):
    generator=torch.Generator().manual_seed(25051)
    a=torch.randn(ashape,generator=generator).requires_grad_()
    b=torch.randn(bshape,generator=generator).requires_grad_()
    plan=optimize_plan(capture_model(Product(),(a,b)),storage_views=views,reuse_storage=reuse)
    prepare(plan)
    ref=CPUReference(plan,{})
    name=plan['outputs'][0]
    actual=ref.forward((a,b))[name]
    expected=a@b
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
    seed=torch.randn(expected.shape,generator=generator)
    expected.backward(seed)
    gradients=ref.backward({name:seed})['inputs']
    torch.testing.assert_close(gradients['a'],a.grad,rtol=2e-5,atol=3e-6)
    torch.testing.assert_close(gradients['b'],b.grad,rtol=2e-5,atol=3e-6)
    if ashape!=bshape and len(ashape+bshape)>4:
        assert any(step['kernel']=='broadcast_to' for step in plan['steps']) == (
            ashape[:-2]!=bshape[:-2])


def test_shared_broadcast_weight_accumulates_both_uses():
    class Shared(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.randn(3,5))
        def forward(self,x):return (x@self.weight)+(x@self.weight)*.25
    torch.manual_seed(25051);model=Shared();x=torch.randn(2,4,2,3,requires_grad=True)
    plan=optimize_plan(capture_model(model,(x,)),reuse_storage=True)
    prepare(plan);reference=CPUReference(plan,extract_weights(model,plan))
    actual=reference.forward((x,));expected=model(x)
    seed=torch.randn_like(expected);expected.backward(seed)
    gradients=reference.backward({plan['outputs'][0]:seed})
    torch.testing.assert_close(actual[plan['outputs'][0]],expected,rtol=2e-5,atol=3e-6)
    torch.testing.assert_close(gradients['inputs']['x'],x.grad,rtol=2e-5,atol=3e-6)
    torch.testing.assert_close(gradients['parameters']['parameter:weight'],model.weight.grad,rtol=2e-5,atol=3e-6)


def test_keyword_matmul_capture_and_out_mutation_rejection():
    class Keywords(torch.nn.Module):
        def forward(self,a,b):return torch.matmul(input=a,other=b)
    plan=capture_model(Keywords(),(torch.ones(2,3),torch.ones(4,3,5)))
    prepare(plan)
    assert any(step['kernel']=='broadcast_to' for step in plan['steps'])
    class Mutating(torch.nn.Module):
        def forward(self,a,b,out):return torch.matmul(a,b,out=out)
    with pytest.raises(ModelGraphError,match='out= mutation'):
        capture_model(Mutating(),(torch.ones(2,3),torch.ones(4,3,5),torch.zeros(4,2,5)))


@pytest.mark.parametrize('source,target',[((),(2,3)),((3,),(2,3)),((1,3,1),(2,3,5)),((2,1,2,3),(2,4,2,3))])
def test_broadcast_copy_encodes_and_preserves_bits(source,target):
    from backend_harness import compile_and_diff
    # Signed zero and ordinary finite FP32 values must pass through unmodified.
    values=[0x80000000,0x3f800001,0xbf800001,0x00000000]
    source_bytes=b''.join(struct.pack('<I',values[i%4]) for i in range(math.prod(source)))
    kernel=emit_kernel('broadcast_to',[{'shape':list(source)}],{'shape':list(target)})
    _,_,bad=compile_and_diff(kernel.ptx,kernel.name,
                            {'a0':source_bytes,'output':bytes(math.prod(target)*4)}, {},
                            block=(kernel.threads,1,1),grid=kernel.grid)
    assert not bad


@pytest.mark.parametrize('source,target,kind,attrs',[
    ((2,3),(3,2),'f32',{}), ((2,3),(3,),'f32',{}),
    ((3,),(2,3),'i64',{}), ((3,),(2,3),'bool',{}),
    ((3,),(1,1,1,2,3),'f32',{}), ((3,),(2,3),'f32',{'axis':0}),
])
def test_broadcast_copy_rejects_invalid_contracts(source,target,kind,attrs):
    with pytest.raises(ValueError):
        emit_kernel('broadcast_to',[{'shape':list(source),'dtype':kind}],{'shape':list(target)},attrs)


def test_matmul_rank_five_rejects_and_bmm_does_not_inherit_broadcasting():
    with pytest.raises(ModelGraphError,match='rank two through four'):
        capture_model(Product(),(torch.ones(2,1,1,2,3),torch.ones(1,1,1,3,5)))
    class BMM(torch.nn.Module):
        def forward(self,a,b):return torch.bmm(a,b)
    with pytest.raises(ModelGraphError):
        capture_model(BMM(),(torch.ones(2,2,3),torch.ones(1,3,5)))


def test_existing_equal_batch_rank_five_matmul_remains_available():
    a=torch.ones(2,1,3,2,3);b=torch.ones(2,1,3,3,5)
    plan=capture_model(Product(),(a,b));prepare(plan)
    result=CPUReference(plan,{}).forward((a,b))[plan['outputs'][0]]
    torch.testing.assert_close(result,a@b)


def test_half_batch_broadcast_uses_declared_existing_product_recipe_bounds():
    plan=capture_model(Product(),(torch.ones(2,1,2,3),torch.ones(1,4,3,5)),math_mode='f16_rne_f32_accum')
    _,steps,_=prepare(plan)
    assert all(step.get('tensor') for step in steps if step['kernel'].startswith('bmm'))
    plan=capture_model(Product(),(torch.ones(1,2,3),torch.ones(33,3,5)),math_mode='f16_rne_f32_accum')
    with pytest.raises(ValueError):prepare(plan)


def test_input_dtype_diagnostic_names_index_and_mask_contracts():
    class Identity(torch.nn.Module):
        def forward(self,x):return x
    with pytest.raises(ModelGraphError,match='FP32, int64 index or Boolean mask'):
        capture_model(Identity(),(torch.ones(2,3,dtype=torch.float64),))
