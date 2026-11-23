"""Independent numerical candidate before native graph/operator promotion."""
import hashlib
import json
import math
from pathlib import Path
import platform

CASES=[((),None,False),((2,3),None,False),((2,3),0,False),((2,3),-1,True),
       ((2,3,5),(0,2),False),((2,3,5),(-1,0),True),((2,1,3,5),(),False),
       ((2,1,3,5),(1,3),False),((2,3,5),[2,0],True)]

def axes_for(shape,dim):
    rank=len(shape)
    dims=tuple(range(rank)) if dim is None or dim==() or dim==[] else (dim,) if type(dim)is int else tuple(dim)
    return tuple(sorted(d%rank for d in dims)) if rank else ()

def reduction(x,dim,keepdim,mean):
    shape=tuple(x.shape);axes=axes_for(shape,dim)
    kept=tuple(1 if i in axes else n for i,n in enumerate(shape))
    y=x.sum_to_size(kept)
    if mean:y=y*(1/math.prod(shape[i] for i in axes))
    result=kept if keepdim else tuple(n for i,n in enumerate(shape) if i not in axes)
    return y.reshape(result)

def run():
    import torch
    torch.set_num_threads(1);g=torch.Generator().manual_seed(21052);records=[]
    for shape,dim,keepdim in CASES:
        for mean in (False,True):
            x=torch.randn(shape,generator=g).requires_grad_();kind='mean' if mean else 'sum'
            expected=getattr(torch,kind)(x,dim=dim,keepdim=keepdim)
            actual=reduction(x,dim,keepdim,mean)
            seed=torch.randn(expected.shape,generator=g)
            eg,=torch.autograd.grad(expected,(x,),seed,retain_graph=True)
            ag,=torch.autograd.grad(actual,(x,),seed)
            torch.testing.assert_close(actual,expected,rtol=2e-6,atol=1e-6)
            torch.testing.assert_close(ag,eg,rtol=2e-6,atol=1e-6)
            records.append({'kind':kind,'shape':list(shape),'dim':dim,'keepdim':keepdim,'passed':True})
    x=torch.tensor([-1e30,-1000,-100,-88,-20,-1.,-0.,0.,1.,20.,88.,100.,1000.,1e30],requires_grad=True)
    z=torch.exp(-torch.clamp(torch.abs(x),max=104))
    actual=torch.where(x>=0,1/(1+z),z/(1+z))
    expected=torch.sigmoid(x)
    torch.testing.assert_close(actual,expected,rtol=2e-6,atol=1e-7)
    # Production differentiates the saved output, matching torch sigmoid's rule.
    expected_grad,=torch.autograd.grad(expected,(x,),torch.ones_like(x))
    torch.testing.assert_close(actual*(1-actual),expected_grad,rtol=2e-6,atol=1e-7)
    records.append({'kind':'stable sigmoid','values':x.detach().tolist(),'passed':True})
    return {'schema':1,'gap':'G21','seed':21052,'python':platform.python_version(),'torch':torch.__version__,
            'candidate_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'passed':True,'records':records,'hardware':'not run by candidate algebra'}
if __name__=='__main__':
    result=run();Path(__file__).with_name('candidate-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
