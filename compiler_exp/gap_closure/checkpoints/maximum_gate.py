"""AMSGrad NaN/Inf full-state classification plus exact checkpoint IEEE custody."""
import hashlib
import json
from pathlib import Path
import platform
import torch
from openassembler.modelfront.graph import capture_model,extract_weights
from openassembler.modelfront.kernels import build_bundle
from openassembler.modelfront.optimizer import optimizer_options,torch_optimizer_kwargs
from openassembler.modelfront.runtime import Runtime
from openassembler.modelfront.profiling import audit_trace

class Point(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.linspace(-.3,.3,17));self.bias=torch.nn.Parameter(torch.zeros(17))
    def forward(self,x):return self.weight*x+self.bias

def run(root):
    root=Path(root);root.mkdir(exist_ok=True);torch.set_num_threads(1);torch.manual_seed(35053)
    records=[];tensors={};runs=[]
    def check(a,b,label,exact=False):
        a=a.detach().cpu().clone();b=b.detach().cpu().clone();tensors[label]={'actual':a,'expected':b}
        if exact:passed=torch.equal(a.view(torch.int32),b.view(torch.int32))
        else:
            finite=torch.isfinite(b)
            passed=torch.equal(torch.isnan(a),torch.isnan(b)) and torch.equal(torch.isposinf(a),torch.isposinf(b)) and torch.equal(torch.isneginf(a),torch.isneginf(b)) and bool(torch.isclose(a[finite],b[finite],rtol=4e-5,atol=5e-6).all())
        row={'name':label,'elements':a.numel(),'passed':passed,'nan_elements':int(torch.isnan(a).sum()),'inf_elements':int(torch.isinf(a).sum()),'exact_bits':exact};records.append(row)
        assert passed,row
    for kind in ('adam','adamw'):
        p=optimizer_options({'name':kind,'amsgrad':True,'maximize':True,'beta1':.8,'beta2':.95,'weight_decay':.03})
        model=Point();plan=capture_model(model,(torch.zeros(17),));plan['optimizer']=p;out=root/kind
        build_bundle(plan,extract_weights(model,plan),out,sass_listings=False);rt=Runtime.load(out);other=None
        opt={'adam':torch.optim.Adam,'adamw':torch.optim.AdamW}[kind](model.parameters(),lr=.001,foreach=False,**torch_optimizer_kwargs(p))
        try:
            for step in range(1,5):
                x=torch.linspace(-.2,.2,17);seed=torch.linspace(.1,-.1,17)
                if step==4:seed[:3]=torch.tensor([float('nan'),float('inf'),-float('inf')])
                opt.zero_grad();model(x).backward(seed)
                rt.forward({'x':x.cuda()},require_tape=True);rt.backward({plan['outputs'][0]:seed.cuda()})
                if step==4:
                    # Audit the targeted state update region after the forward/
                    # backward has synchronized, avoiding missing initial GPU
                    # activities during the first profiler startup in this process.
                    rt.synchronize();rt.launches=[]
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
                        rt.update(.001);rt.synchronize()
                    trace=root/(kind+'-trace.json');profile.export_chrome_trace(str(trace));audit=audit_trace(json.loads(trace.read_text()),rt.launch_ledger)
                else:rt.update(.001);rt.synchronize()
                opt.step();state=rt.optimizer_state_dict();assert state['step']==step
                for spec in plan['parameters']:
                    parameter=model.get_parameter(spec['target']);n=spec['name']
                    check(rt.weights[n],parameter,f'{kind}/{step}/weight/{n}')
                    for key,ref in [('m','exp_avg'),('v','exp_avg_sq'),('max_v','max_exp_avg_sq')]:check(state['moments'][n][key],opt.state[parameter][ref],f'{kind}/{step}/{key}/{n}')
                assert rt.check_canaries()['passed']
            checkpoint=root/(kind+'-ieee.oa-checkpoint');rt.save_checkpoint(checkpoint,application={'cursor':4})
            saved_weights={n:w.cpu().clone() for n,w in rt.weights.items()};saved_state=rt.optimizer_state_dict()
            other=Runtime.load(out);result=other.load_checkpoint(checkpoint,device='cuda');assert result['step']==4 and result['application']=={'cursor':4}
            restored=other.optimizer_state_dict()
            for n,w in saved_weights.items():check(other.weights[n],w,f'{kind}/restored/weight/{n}',True)
            for n,pair in saved_state['moments'].items():
                for key,v in pair.items():check(restored['moments'][n][key],v,f'{kind}/restored/{key}/{n}',True)
            again=root/(kind+'-resaved.oa-checkpoint');other.save_checkpoint(again,application={'cursor':4});assert again.read_bytes()==checkpoint.read_bytes()
            runs.append({'optimizer':kind,'updates':4,'full_ieee_checkpoint_bytes_equal':True,'profiler':audit,'guards':other.check_canaries()})
        finally:
            if other:other.close()
            rt.close()
    torch.save(tensors,root/'full-tensors.pt')
    package=Path(__import__('openassembler').__file__).parent
    result={'schema':1,'gap':'G34 maximum / G35 IEEE state','passed':all(r['passed'] for r in records),'comparisons':records,'runs':runs,
            'python':platform.python_version(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'device':str(torch.cuda.get_device_properties(0)),
            'source_hashes':{p.relative_to(package).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in package.rglob('*.py')},
            'scope':'NaN/Inf classifications of every weight/moment/max against CPU Torch; no NaN-payload equivalence claim across implementations; exact raw-bit checkpoint self-roundtrip'}
    (root/'report.json').write_text(json.dumps(result,indent=2)+'\n');return result
