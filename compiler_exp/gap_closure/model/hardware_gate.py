"""A100 full-state gate for exact-shape batch-broadcast products."""
import copy
import hashlib
import json
import platform
from pathlib import Path
import torch
from openassembler.modelfront.graph import capture_model,extract_weights
from openassembler.modelfront.kernels import build_bundle
from openassembler.modelfront.optimization import optimize_plan
from openassembler.modelfront.reference import CPUReference
from openassembler.modelfront.runtime import Runtime

CASES = [
    ((2,3),(4,3,5)), ((4,2,3),(3,5)), ((1,2,3),(4,3,5)),
    ((4,2,3),(1,3,5)), ((2,1,2,3),(1,4,3,5)),
    ((2,4,2,3),(4,3,5)), ((2,3),(2,4,3,5)),
]
class Product(torch.nn.Module):
    def forward(self,a,b):return a@b
class Shared(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.randn(3,5)*.25)
    def forward(self,x):return (x@self.weight)+(x@self.weight)*.25


def run(root):
    root=Path(root);root.mkdir(exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(25051)
    torch.backends.cuda.matmul.allow_tf32=False
    records=[];raw={};runtimes=[]
    def check(actual,expected,label):
        a=actual.detach().cpu().clone();b=expected.detach().cpu().clone()
        raw[label]={'actual':a,'expected':b}
        close=torch.isclose(a,b,rtol=4e-5,atol=5e-6)
        record={'name':label,'elements':a.numel(),'passed':bool(close.all()),
                'max_abs':float((a-b).abs().max()),'mismatches':int((~close).sum())}
        records.append(record)
        torch.save(raw,root/'full-tensors.pt')
        assert record['passed'],record
    def test(model,args,label,views,reuse,math_mode='fp32'):
        model=copy.deepcopy(model)
        plan=optimize_plan(capture_model(model,args,math_mode=math_mode),storage_views=views,reuse_storage=reuse)
        bundle=root/label;build_bundle(plan,extract_weights(model,plan),bundle,sass_listings=False)
        rt=Runtime.load(bundle)
        try:
            for iteration in range(2):
                xs=tuple((x+.03125*iteration).detach().clone().requires_grad_() for x in args)
                inputs={s['name']:x.detach().cuda() for s,x in zip(plan['inputs'],xs)}
                before={n:t.cpu().clone() for n,t in inputs.items()}
                weights_before={n:t.cpu().clone() for n,t in rt.weights.items()}
                if iteration:rt.poison_workspace()
                model.zero_grad(set_to_none=True)
                if math_mode=='fp32':
                    expected=model(*xs);seed=torch.randn_like(expected);expected.backward(seed)
                    expected_grads={'inputs':{s['name']:x.grad for s,x in zip(plan['inputs'],xs)},
                                    'parameters':{s['name']:dict(model.named_parameters())[s['target']].grad for s in plan['parameters']}}
                else:
                    ref=CPUReference(plan,extract_weights(model,plan));expected=ref.forward(xs)[plan['outputs'][0]]
                    seed=torch.randn_like(expected);expected_grads=ref.backward({plan['outputs'][0]:seed})
                got=rt.forward(inputs,require_tape=True);name=plan['outputs'][0]
                check(got[name],expected,f'{label}/{iteration}/output')
                gradients=rt.backward({name:seed.cuda()})
                for category in ('inputs','parameters'):
                    for n,value in expected_grads[category].items():
                        check(gradients[category][n],value,f'{label}/{iteration}/{category}/{n}')
                for n,value in inputs.items():assert torch.equal(value.cpu(),before[n])
                for n,value in rt.weights.items():assert torch.equal(value.cpu(),weights_before[n])
                runtimes.append({'name':label,'iteration':iteration,'guards':rt.check_canaries(),
                                 'launches':list(rt.launch_ledger),'inputs_and_weights_immutable':True})
            # An independent activity audit is distinct from profiler-free correctness.
            from openassembler.modelfront.profiling import audit_trace
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
                rt.forward(inputs)
                rt.synchronize()
            trace=root/(label+'-trace.json');profiler.export_chrome_trace(str(trace))
            audit=audit_trace(json.loads(trace.read_text()),rt.launch_ledger)
            runtimes.append({'name':label,'profiler':audit})
        finally:rt.close()
    for i,(a,b) in enumerate(CASES):
        test(Product(),(torch.randn(a)*.25,torch.randn(b)*.25),f'fp32-{i}',bool(i&1),bool(i&2))
    for views,reuse in ((False,False),(False,True),(True,False),(True,True)):
        test(Shared(),(torch.randn(2,4,2,3)*.25,),f'shared-{int(views)}{int(reuse)}',views,reuse)
    test(Product(),(torch.randn(2,1,2,3)*.25,torch.randn(1,4,3,5)*.25),'half-outer',True,True,'f16_rne_f32_accum')
    identity={}
    package=Path(__import__('openassembler').__file__).parent
    for p in package.rglob('*.py'):identity[p.relative_to(package).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    result={'schema':1,'gap':'G25','passed':all(r['passed'] for r in records),
            'python':platform.python_version(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
            'device':str(torch.cuda.get_device_properties(0)), 'seed':25051,
            'rtol':4e-5,'atol':5e-6,'comparisons':records,'runs':runtimes,
            'source_hashes':identity,'performance':'not measured'}
    (root/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
