"""A100 full-output/derivative gates for native sigmoid and reductions."""
import copy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import torch
from openassembler.modelfront.graph import capture_model,extract_weights,flatten_outputs
from openassembler.modelfront.kernels import build_bundle
from openassembler.modelfront.optimization import optimize_plan
from openassembler.modelfront.runtime import Runtime
from openassembler.modelfront.profiling import audit_trace

class Reduction(torch.nn.Module):
    def __init__(self,dim,keepdim,kind,method):
        super().__init__();self.dim=dim;self.keepdim=keepdim;self.kind=kind;self.method=method
    def forward(self,x):
        if self.method:return getattr(x,self.kind)(dim=self.dim,keepdim=self.keepdim)
        return getattr(torch,self.kind)(x,dim=self.dim,keepdim=self.keepdim)
class Activation(torch.nn.Module):
    def __init__(self,route):super().__init__();self.route=route;self.activation=torch.nn.Sigmoid()
    def forward(self,x):
        if self.route=='module':return self.activation(x)
        if self.route=='torch':return torch.sigmoid(x)
        if self.route=='functional':return torch.nn.functional.sigmoid(x)
        return x.sigmoid()
class MultiOutput(torch.nn.Module):
    def __init__(self):super().__init__();self.bias=torch.nn.Parameter(torch.randn(5)*.25)
    def forward(self,x):
        y=torch.sigmoid(x+self.bias)
        return y.sum(dim=(0,2)),y.mean(dim=1,keepdim=True)

def run(root):
    root=Path(root);root.mkdir(exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(21052);torch.backends.cuda.matmul.allow_tf32=False
    records=[];raw={};runs=[]
    def check(actual,expected,label):
        a=actual.detach().cpu().clone();b=expected.detach().cpu().clone()
        raw[label]={'actual':a,'expected':b}
        close=torch.isclose(a,b,rtol=4e-5,atol=5e-6)
        row={'name':label,'elements':a.numel(),'passed':bool(close.all()),
             'max_abs':float((a-b).abs().max()),'mismatches':int((~close).sum())}
        records.append(row);torch.save(raw,root/'full-tensors.pt');assert row['passed'],row
    def test(model,args,label,views,reuse):
        model=copy.deepcopy(model)
        plan=optimize_plan(capture_model(model,args),storage_views=views,reuse_storage=reuse)
        bundle=root/label;build_bundle(plan,extract_weights(model,plan),bundle,sass_listings=False)
        rt=Runtime.load(bundle)
        try:
            for iteration in range(2):
                xs=tuple((x+.03125*iteration).detach().clone().requires_grad_() for x in args)
                gpu={s['name']:x.detach().cuda() for s,x in zip(plan['inputs'],xs)}
                before={n:t.cpu().clone() for n,t in gpu.items()}
                weights={n:t.cpu().clone() for n,t in rt.weights.items()}
                if iteration:rt.poison_workspace()
                model.zero_grad(set_to_none=True)
                expected=flatten_outputs(plan,model(*xs))
                seeds={n:torch.randn_like(y) for n,y in expected.items()}
                torch.autograd.backward(tuple(expected.values()),tuple(seeds.values()))
                actual=rt.forward(gpu,require_tape=True)
                for n,y in expected.items():check(actual[n],y,f'{label}/{iteration}/output/{n}')
                gradients=rt.backward({n:v.cuda() for n,v in seeds.items()})
                for s,x in zip(plan['inputs'],xs):check(gradients['inputs'][s['name']],x.grad,f'{label}/{iteration}/dx/{s["name"]}')
                for s in plan['parameters']:
                    value=dict(model.named_parameters())[s['target']]
                    check(gradients['parameters'][s['name']],value.grad,f'{label}/{iteration}/dw/{s["name"]}')
                for n,t in gpu.items():assert torch.equal(t.cpu(),before[n])
                for n,t in rt.weights.items():assert torch.equal(t.cpu(),weights[n])
                runs.append({'name':label,'iteration':iteration,'guards':rt.check_canaries(),'launches':list(rt.launch_ledger),'inputs_weights_immutable':True})
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
                rt.forward(gpu);rt.synchronize()
            trace=root/(label+'-trace.json');profiler.export_chrome_trace(str(trace))
            runs.append({'name':label,'profiler':audit_trace(json.loads(trace.read_text()),rt.launch_ledger)})
        finally:rt.close()
    cases=[((),None,False),((2,3),0,False),((2,3),-1,True),((2,3,5),(0,2),False),
           ((2,3,5),(-1,0),True),((2,1,3,5),(),False),((2,1,3,5),(1,3),False),
           ((2,3,5),[2,0],True),((3,33),1,False),((2,513),1,True)]
    for views,reuse in ((False,False),(False,True),(True,False),(True,True)):
        suffix=f'{int(views)}{int(reuse)}'
        for i,(shape,dim,keepdim) in enumerate(cases):
            for kind in ('sum','mean'):
                test(Reduction(dim,keepdim,kind,bool(i&1)),(torch.randn(shape)*.25,),f'{kind}-{i}-{suffix}',views,reuse)
        extreme=torch.tensor([-1e30,-1000,-100,-88,-20,-1.,-0.,0.,1.,20.,88.,100.,1000.,1e30]+[i*.125-8 for i in range(129)])
        for route in ('torch','module','tensor','functional'):
            test(Activation(route),(extreme,),f'sigmoid-{route}-{suffix}',views,reuse)
        test(MultiOutput(),(torch.randn(2,3,5)*.25,),f'combined-{suffix}',views,reuse)
    cli=[]
    for args in [['compile','repo','--backend','local','--verify','--no-sass','--out-dir','cli-output'],
                 ['run','cli-output','--backend','local','--verify']]:
        done=subprocess.run([sys.executable,'-m','openassembler',*args],capture_output=True,text=True,timeout=600)
        cli.append({'args':args,'returncode':done.returncode,'stdout':done.stdout,'stderr':done.stderr})
        (root/'cli-results.json').write_text(json.dumps(cli,indent=2)+'\n')
        assert done.returncode==0,cli[-1]
    reports={p.name:json.loads(p.read_text()) for p in Path('cli-output').glob('*report.json')}
    package=Path(__import__('openassembler').__file__).parent
    result={'schema':1,'gap':'G21','passed':all(r['passed'] for r in records),
            'python':platform.python_version(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
            'device':str(torch.cuda.get_device_properties(0)),'seed':21052,'rtol':4e-5,'atol':5e-6,
            'comparisons':records,'runs':runs,'cli':cli,'cli_reports':reports,
            'source_hashes':{p.relative_to(package).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in package.rglob('*.py')},
            'performance':'not measured by this correctness gate'}
    (root/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
