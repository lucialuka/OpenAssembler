"""Native extended optimizers: changing gradients, complete state, resume, CLI."""
import copy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import torch
from openassembler.modelfront.graph import capture_model,extract_weights
from openassembler.modelfront.kernels import build_bundle
from openassembler.modelfront.optimization import optimize_plan
from openassembler.modelfront.optimizer import optimizer_options,optimizer_state_keys,torch_optimizer_kwargs
from openassembler.modelfront.runtime import Runtime
from openassembler.modelfront.profiling import audit_trace

POLICIES=[
 {'name':'sgd'}, {'name':'sgd','weight_decay':.03,'maximize':True},
 {'name':'sgd','momentum':.8,'dampening':.25,'weight_decay':.03},
 {'name':'sgd','momentum':.9,'nesterov':True,'maximize':True,'weight_decay':.02},
 {'name':'adam','beta1':.8,'beta2':.95,'epsilon':1e-8,'weight_decay':.03,'amsgrad':True,'maximize':True},
 {'name':'adamw','beta1':.8,'beta2':.95,'epsilon':1e-8,'weight_decay':.03,'amsgrad':True},
 {'name':'adam','beta1':0.,'beta2':0.,'epsilon':1e-4,'weight_decay':.03,'maximize':True},
 {'name':'adamw','beta1':.9,'beta2':.999,'epsilon':1e-8,'weight_decay':0.,'maximize':True},
]
class Point(torch.nn.Module):
    def __init__(self,n):
        super().__init__();self.weight=torch.nn.Parameter(torch.linspace(-.5,.5,n));self.bias=torch.nn.Parameter(torch.linspace(.1,-.1,n))
    def forward(self,x):return self.weight*x+self.bias


def run(root):
    root=Path(root);root.mkdir(exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(34052);torch.backends.cuda.matmul.allow_tf32=False
    comparisons=[];raw={};runs=[]
    def check(actual,expected,label):
        a=actual.detach().cpu().clone();b=expected.detach().cpu().clone();raw[label]={'actual':a,'expected':b}
        close=torch.isclose(a,b,rtol=4e-5,atol=5e-6)
        row={'name':label,'elements':a.numel(),'passed':bool(close.all()),'max_abs':float((a-b).abs().max()),'mismatches':int((~close).sum())}
        comparisons.append(row)
        if not row['passed']:torch.save(raw,root/'failure-tensors.pt');raise AssertionError(row)
    mapping={'momentum':'momentum_buffer','m':'exp_avg','v':'exp_avg_sq','max_v':'max_exp_avg_sq'}
    for i,policy in enumerate(POLICIES):
        for fused in ([False,True] if policy['name']!='sgd' and not policy.get('amsgrad',False) else [False]):
            model=Point(129);p=optimizer_options(policy);plan=capture_model(model,(torch.zeros(129),));plan['optimizer']=p
            if fused:plan['optimizer_fusion']=True
            plan=optimize_plan(plan,storage_views=bool(i&1),reuse_storage=bool(i&2))
            label=f'policy-{i}-fused-{int(fused)}';out=root/label
            build_bundle(plan,extract_weights(model,plan),out,sass_listings=False);rt=Runtime.load(out);resumed=None
            optimizer={'sgd':torch.optim.SGD,'adam':torch.optim.Adam,'adamw':torch.optim.AdamW}[p['name']](model.parameters(),lr=.001,foreach=False,**torch_optimizer_kwargs(p))
            try:
                for step in range(1,25):
                    x=torch.randn(129)*.1;seed=torch.randn(129)*(.1 if step%3 else 1e-7)
                    if step%5==0:seed.zero_()
                    lr=0. if step==7 else .001+.0001*(step%4)
                    optimizer.param_groups[0]['lr']=lr;optimizer.zero_grad();prediction=model(x);prediction.backward(seed)
                    gpu={'x':x.cuda()};before=gpu['x'].cpu().clone()
                    if step>1:rt.poison_workspace()
                    actual=rt.forward(gpu,require_tape=True);name=plan['outputs'][0]
                    check(actual[name],prediction,f'{label}/{step}/output')
                    grads=rt.backward({name:seed.cuda()})
                    for s in plan['parameters']:
                        original=dict(model.named_parameters())[s['target']]
                        check(grads['parameters'][s['name']],original.grad,f'{label}/{step}/gradient/{s["name"]}')
                    rt.update(lr);optimizer.step();state=rt.optimizer_state_dict()
                    assert state['step']==step and state['options']==p
                    assert set(state['moments'])==(set(plan['backward']['grad_parameters']) if optimizer_state_keys(p) else set())
                    for s in plan['parameters']:
                        original=dict(model.named_parameters())[s['target']]
                        check(rt.weights[s['name']],original,f'{label}/{step}/weight/{s["name"]}')
                        if optimizer_state_keys(p):
                            assert set(state['moments'][s['name']])==set(optimizer_state_keys(p))
                            for key,value in state['moments'][s['name']].items():check(value,optimizer.state[original][mapping[key]],f'{label}/{step}/{key}/{s["name"]}')
                    if step==12:
                        snapshot=copy.deepcopy(state)
                        resumed=Runtime(plan,{n:w.detach().clone() for n,w in rt.weights.items()},out)
                        resumed.forward(gpu);resumed.discard_tape();resumed.load_optimizer_state_dict(snapshot)
                        for pair in snapshot['moments'].values():
                            for value in pair.values():value.fill_(123) # caller mutation must not affect loaded device state
                    elif step>12:
                        resumed.forward(gpu,require_tape=True);resumed.backward({name:seed.cuda()});resumed.update(lr)
                        resume_state=resumed.optimizer_state_dict();assert resume_state['step']==step
                        for n,w in rt.weights.items():check(resumed.weights[n],w,f'{label}/{step}/resume/weight/{n}')
                        for n,pair in state['moments'].items():
                            for key,value in pair.items():check(resume_state['moments'][n][key],value,f'{label}/{step}/resume/{key}/{n}')
                        assert resumed.check_canaries()['passed']
                    assert torch.equal(gpu['x'].cpu(),before)
                    runs.append({'name':label,'step':step,'options':p,'guards':rt.check_canaries(),'launches':list(rt.launch_ledger),'input_immutable':True})
                # Profile one additional prescribed gradient update; CPU control advances too.
                x=torch.randn(129)*.1;seed=torch.randn(129)*.1;optimizer.zero_grad();model(x).backward(seed)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
                    rt.forward({'x':x.cuda()},require_tape=True);rt.backward({name:seed.cuda()});rt.update(.001);rt.synchronize()
                optimizer.param_groups[0]['lr']=.001;optimizer.step()
                trace=root/(label+'-trace.json');profile.export_chrome_trace(str(trace))
                for s in plan['parameters']:check(rt.weights[s['name']],dict(model.named_parameters())[s['target']],f'{label}/profile/weight/{s["name"]}')
                runs.append({'name':label,'profile':audit_trace(json.loads(trace.read_text()),rt.launch_ledger)})
            finally:
                if resumed:resumed.close()
                rt.close()
    torch.save(raw,root/'full-tensors.pt')
    cli=[];cli_reports={}
    for label,policy in [('sgd-momentum',POLICIES[3]),('adam-amsgrad',POLICIES[4]),('adamw-amsgrad',POLICIES[5]),('adamw-maximize-fused',POLICIES[7])]:
        def toml(v):return str(v).lower() if type(v)is bool else '"'+v+'"' if isinstance(v,str) else str(v)
        config='[model]\nentrypoint="model.py"\ninput_shapes=[[3,7]]\nmode="training"\nsteps=6\nprofile_iterations=2\nlearning_rate=0.001\nseed=34052\nstorage_views=true\nreuse_storage=true\nsass_listings=false\nrtol=0.00004\natol=0.000005\n'
        config+=''.join(('optimizer' if k=='name' else k)+'='+toml(v)+'\n' for k,v in policy.items())
        if label.endswith('fused'):config+='optimizer_fusion=true\n'
        Path('repo/openassembler.toml').write_text(config)
        output='cli-'+label
        for args in [['compile','repo','--backend','local','--verify','--no-sass','--out-dir',output],['run',output,'--backend','local','--verify']]:
            done=subprocess.run([sys.executable,'-m','openassembler',*args],capture_output=True,text=True,timeout=600)
            cli.append({'args':args,'returncode':done.returncode,'stdout':done.stdout,'stderr':done.stderr})
            (root/'cli-results.json').write_text(json.dumps(cli,indent=2)+'\n');assert done.returncode==0,cli[-1]
        cli_reports[label]={p.name:json.loads(p.read_text()) for p in Path(output).glob('*report.json')}
        cli_reports[label]['model-run.json']=json.loads(Path(output,'model-run.json').read_text())
    package=Path(__import__('openassembler').__file__).parent
    result={'schema':1,'gap':'G34','passed':all(r['passed'] for r in comparisons),'seed':34052,'rtol':4e-5,'atol':5e-6,
            'python':platform.python_version(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'device':str(torch.cuda.get_device_properties(0)),
            'comparisons':comparisons,'runs':runs,'cli':cli,'cli_reports':cli_reports,
            'source_hashes':{p.relative_to(package).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in package.rglob('*.py')},
            'performance':'not measured by this correctness/state gate'}
    (root/'report.json').write_text(json.dumps(result,indent=2)+'\n');return result
