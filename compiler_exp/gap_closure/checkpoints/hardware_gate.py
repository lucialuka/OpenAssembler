"""Complete native checkpoints: interrupted/resumed all-state/RNG and rollback gates."""
import copy
import hashlib
import json
from pathlib import Path
import platform
import random
import torch
import zipfile
from openassembler.modelfront.graph import capture_model,extract_weights
from openassembler.modelfront.kernels import build_bundle,emit_kernel
from openassembler.modelfront.optimization import optimize_plan
from openassembler.modelfront.optimizer import optimizer_options,torch_optimizer_kwargs
from openassembler.modelfront.runtime import Runtime
from openassembler.modelfront.autograd import CompiledModel
from openassembler.modelfront.profiling import audit_trace

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.linspace(-.5,.5,17));self.bias=torch.nn.Parameter(torch.zeros(17));self.register_buffer('offset',torch.linspace(0,.1,17))
    def forward(self,x):return self.weight*x+self.bias+self.offset
POLICIES=[{'name':'sgd','momentum':.9,'nesterov':True,'weight_decay':.02,'maximize':True},
          {'name':'adam','beta1':.8,'beta2':.95,'amsgrad':True,'maximize':True,'weight_decay':.03},
          {'name':'adamw','beta1':.8,'beta2':.95,'amsgrad':True,'weight_decay':.03}]

def run(root):
    root=Path(root);root.mkdir(exist_ok=True)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    records=[];raw={};cases=[];negatives=[]
    def check(a,b,label,exact=False):
        a=a.detach().cpu().clone();b=b.detach().cpu().clone();raw[label]={'actual':a,'expected':b}
        passed=torch.equal(a.view(torch.int32),b.view(torch.int32)) if exact else bool(torch.isclose(a,b,rtol=4e-5,atol=5e-6,equal_nan=True).all())
        records.append({'name':label,'passed':passed,'elements':a.numel(),'exact_bits':exact})
        if not passed:torch.save(raw,root/'failure-tensors.pt');raise AssertionError(records[-1])
    def app(step):return {'cursor':step,'scheduler':{'kind':'declared_step_table','last_epoch':step,'next_rate':rate(step+1)},'scaler':None,'data_cursor':{'epoch':step//7,'offset':step%7}}
    def rate(step):return 0. if step==7 else .001+.0001*(step%4)
    def data():return torch.randn(17)*.1,torch.randn(17)*.1+random.random()*.01
    def snapshot(rt):return ({n:w.detach().cpu().clone() for n,w in rt.weights.items()},rt.optimizer_state_dict())
    def compare_state(rt,expected,label,exact=False):
        weights,state=expected;now=rt.optimizer_state_dict();assert now['step']==state['step'] and now['options']==state['options']
        assert set(now['moments'])==set(state['moments'])
        for n,w in weights.items():check(rt.weights[n],w,label+'/weight/'+n,exact)
        for n,pair in state['moments'].items():
            assert set(now['moments'][n])==set(pair)
            for k,v in pair.items():check(now['moments'][n][k],v,label+'/'+k+'/'+n,exact)
    def assert_unchanged(rt,before,label):
        weights,state,identities,py,tr,guards=before
        compare_state(rt,(weights,state),label,True)
        assert {n:v.data_ptr() for n,v in rt.weights.items()}==identities
        assert random.getstate()==py and torch.equal(torch.get_rng_state(),tr)
        assert len(rt._guards)==guards
    def before(rt):
        weights,state=snapshot(rt)
        return weights,state,{n:v.data_ptr() for n,v in rt.weights.items()},random.getstate(),torch.get_rng_state().clone(),len(rt._guards)
    mapping={'momentum':'momentum_buffer','m':'exp_avg','v':'exp_avg_sq','max_v':'max_exp_avg_sq'}
    for index,policy in enumerate(POLICIES):
        for views,reuse in ((False,False),(False,True),(True,False),(True,True)):
            p=optimizer_options(policy);model=Net();plan=capture_model(model,(torch.zeros(17),));plan['optimizer']=p
            plan=optimize_plan(plan,storage_views=views,reuse_storage=reuse)
            label=f'checkpoint-{index}-{int(views)}{int(reuse)}';bundle=root/label
            build_bundle(plan,extract_weights(model,plan),bundle,sass_listings=False);rt=Runtime.load(bundle);resume=None;owner=None
            opt={'sgd':torch.optim.SGD,'adam':torch.optim.Adam,'adamw':torch.optim.AdamW}[p['name']](model.parameters(),lr=.001,foreach=False,**torch_optimizer_kwargs(p))
            torch.manual_seed(35052);random.seed(35052);expected={};inputs={}
            saved=root/(label+'-step12.oa-checkpoint');final=root/(label+'-final.oa-checkpoint')
            try:
                for step in range(1,31):
                    x,target=data();inputs[step]=(x.clone(),target.clone());x.requires_grad_();opt.zero_grad()
                    prediction=model(x);loss=((prediction-target)**2).mean();loss.backward();opt.param_groups[0]['lr']=rate(step)
                    if step>1:rt.poison_workspace()
                    actual=rt.train_step({'x':x.detach().cuda()},target.cuda(),rate(step))
                    check(actual['prediction'],prediction,f'{label}/{step}/output');check(actual['loss'].reshape(()),loss,f'{label}/{step}/loss')
                    check(actual['gradients']['inputs']['x'],x.grad,f'{label}/{step}/dx')
                    for spec in plan['parameters']:
                        if not spec.get('requires_grad',True):continue
                        value=dict(model.named_parameters())[spec['target']]
                        check(actual['gradients']['parameters'][spec['name']],value.grad,f'{label}/{step}/gradient/{spec["name"]}')
                    opt.step();state=rt.optimizer_state_dict();assert state['step']==step
                    for spec in plan['parameters']:
                        value=model.get_parameter(spec['target']) if spec.get('requires_grad',True) else model.get_buffer(spec['target'])
                        check(rt.weights[spec['name']],value,f'{label}/{step}/state/{spec["name"]}')
                        if spec['name'] in state['moments']:
                            for k,v in state['moments'][spec['name']].items():check(v,opt.state[value][mapping[k]],f'{label}/{step}/{k}/{spec["name"]}')
                    expected[step]={'state':snapshot(rt),'prediction':actual['prediction'].cpu().clone(),'loss':actual['loss'].cpu().clone()}
                    assert rt.check_canaries()['passed']
                    if step==12:rt.save_checkpoint(saved,application=app(12))
                rt.save_checkpoint(final,application=app(30))
                # All malformed-data controls fail before even initializing a fresh runtime.
                resume=Runtime.load(bundle);valid_bytes=saved.read_bytes()
                with zipfile.ZipFile(saved) as z:files={n:z.read(n) for n in z.namelist()}
                manifest=json.loads(files['manifest.json']);manifest['program']['plan_sha256']='0'*64
                files['manifest.json']=json.dumps(manifest).encode();bad=root/(label+'-wrong-program.oa-checkpoint')
                with zipfile.ZipFile(bad,'w',zipfile.ZIP_STORED) as z:
                    for n,v in files.items():z.writestr(n,v)
                identities={n:t.data_ptr() for n,t in resume.weights.items()};rng=(random.getstate(),torch.get_rng_state().clone())
                try:resume.load_checkpoint(bad)
                except ValueError:pass
                else:raise AssertionError('wrong program checkpoint admitted')
                assert resume._module is None and identities=={n:t.data_ptr() for n,t in resume.weights.items()}
                assert rng[0]==random.getstate() and torch.equal(rng[1],torch.get_rng_state())
                negatives.append({'case':label,'wrong_program_before_initialization':True})
                restored=resume.load_checkpoint(saved,device='cuda');assert restored['application']==app(12) and restored['step']==12
                compare_state(resume,expected[12]['state'],label+'/loaded',True)
                if views and reuse:
                    allocations=len(resume._allocations);guards=len(resume._guards)
                    borrowed=resume.weights[plan['parameters'][0]['name']]
                    borrowed_copy=borrowed.cpu().clone()
                    for _ in range(4):
                        resume.load_checkpoint(saved)
                        assert len(resume._allocations)==allocations and len(resume._guards)==guards
                    check(borrowed,borrowed_copy,label+'/borrowed-old-storage',True)
                    # Legacy state-loader strided CPU inputs must compact on CPU before DMA.
                    state=resume.optimizer_state_dict()
                    for pair in state['moments'].values():
                        for k,v in list(pair.items()):
                            backing=torch.empty(v.numel()*2);backing[::2].copy_(v.reshape(-1));pair[k]=backing[::2].reshape(v.shape)
                    resume.load_optimizer_state_dict(state);compare_state(resume,expected[12]['state'],label+'/strided-load',True)
                    assert len(resume._allocations)==allocations and len(resume._guards)==guards
                    # Allocation failure after one staged buffer cannot replace any live binding.
                    initial=before(resume);original=Runtime._allocate;count=0
                    def injected(self,*a,**k):
                        nonlocal count
                        count+=1
                        if count==2:raise RuntimeError('injected staged allocation failure')
                        return original(self,*a,**k)
                    Runtime._allocate=injected
                    try:
                        try:resume.load_checkpoint(saved)
                        except RuntimeError as error:assert 'injected' in str(error)
                        else:raise AssertionError('staging failure not observed')
                    finally:Runtime._allocate=original
                    assert_unchanged(resume,initial,label+'/allocation-rollback')
                    # RNG failure after a partial CPU RNG write also rolls back completely.
                    initial=before(resume);original_rng=torch.set_rng_state;count=0
                    def rng_failure(value):
                        nonlocal count
                        count+=1;original_rng(value)
                        if count==1:raise RuntimeError('injected RNG commit failure')
                    torch.set_rng_state=rng_failure
                    try:
                        try:resume.load_checkpoint(saved)
                        except RuntimeError as error:assert 'injected' in str(error)
                        else:raise AssertionError('RNG failure not observed')
                    finally:torch.set_rng_state=original_rng
                    assert_unchanged(resume,initial,label+'/rng-rollback')
                    # No checkpoint may cross an outstanding tape, gradients or replay owner.
                    for flag,value in (('_tape',{}),('_gradients',{}),('_checkpoint_replay_owner',True)):
                        old=getattr(resume,flag,None);setattr(resume,flag,value);initial=snapshot(resume)
                        try:
                            try:resume.load_checkpoint(saved)
                            except ValueError:pass
                            else:raise AssertionError('unsafe checkpoint boundary admitted')
                            compare_state(resume,initial,label+'/boundary-'+flag,True)
                        finally:
                            if old is None and flag=='_checkpoint_replay_owner':delattr(resume,flag)
                            else:setattr(resume,flag,old)
                    negatives.append({'case':label,'allocation_rollback':True,'rng_rollback_after_partial_write':True,'tape_gradient_replay_boundaries':True})
                for step in range(13,31):
                    x,target=data();check(x,inputs[step][0],f'{label}/{step}/rng/input',True);check(target,inputs[step][1],f'{label}/{step}/rng/target',True)
                    if step>13:resume.poison_workspace()
                    got=resume.train_step({'x':x.cuda()},target.cuda(),rate(step))
                    check(got['prediction'],expected[step]['prediction'],f'{label}/{step}/resume/output',True)
                    check(got['loss'],expected[step]['loss'],f'{label}/{step}/resume/loss',True)
                    compare_state(resume,expected[step]['state'],f'{label}/{step}/resume',True)
                    assert resume.check_canaries()['passed']
                resumed_file=root/(label+'-resumed-final.oa-checkpoint');resume.save_checkpoint(resumed_file,application=app(30))
                assert resumed_file.read_bytes()==final.read_bytes()
                if views and reuse:
                    owner=CompiledModel(plan,extract_weights(model,plan),bundle)
                    result=owner.load_native_checkpoint(saved,device='cuda');assert result['application']==app(12)
                    for step in range(13,31):
                        x,target=data();check(x,inputs[step][0],f'{label}/{step}/module-rng',True)
                        got=owner.native_train_step((x.cuda(),),target.cuda(),learning_rate=rate(step))
                        check(got['prediction'],expected[step]['prediction'],f'{label}/{step}/module-output',True)
                        compare_state(owner._native_runtime,expected[step]['state'],f'{label}/{step}/module-state',True)
                    owned_file=root/(label+'-module-final.oa-checkpoint');owner.save_native_checkpoint(owned_file,application=app(30))
                    assert owned_file.read_bytes()==final.read_bytes()
                    # Registered Module state can be replaced without an intervening
                    # native call. Save must refresh the persistent binding cache.
                    replacement={n:v.detach().cpu().clone() for n,v in owner.state_dict().items()}
                    first=owner._state_attributes[0];replacement[first]=replacement[first]+.0625
                    owner.load_state_dict({n:v.cuda() for n,v in replacement.items()},assign=True)
                    replacement_file=root/(label+'-registered-replacement.oa-checkpoint')
                    owner.save_native_checkpoint(replacement_file,application=app(30))
                    from openassembler.modelfront.checkpoint import read_checkpoint
                    replaced=read_checkpoint(owner._native_runtime,replacement_file)
                    for n,v in owner.native_parameters().items():check(replaced['weights'][n],v,label+'/module-replaced/'+n,True)
                cases.append({'name':label,'policy':p,'updates':30,'resume_cursor':12,'all_checkpoint_bytes_equal':True,
                              'all_model_parameters_buffers_optimizer_rng_application_equal':True,'compiled_module_resume':views and reuse,
                              'guards':resume.check_canaries()})
                # AMSGrad's corrected maximum: NaN classification and signed-zero tie controls.
                if index and views and reuse:
                    spec={'shape':[17],'dtype':'f32'};left=torch.tensor([float('nan'),1.,float('nan'),0.,-0.,-0.,.3,.2]+[0.]*9)
                    right=torch.tensor([1.,float('nan'),float('nan'),-0.,0.,-0.,.2,.3]+[0.]*9)
                    maximum=emit_kernel('adam_max',[spec]*2,spec,p);out=resume._allocate([17]);resume.launches=[]
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
                        resume._launch(maximum,[left.cuda(),right.cuda()],out);resume.synchronize()
                    trace=root/(label+'-maximum-trace.json');profile.export_chrome_trace(str(trace));audit=audit_trace(json.loads(trace.read_text()),resume.launch_ledger)
                    expected_max=torch.maximum(left,right);actual_max=out.cpu();assert torch.equal(torch.isnan(actual_max),torch.isnan(expected_max))
                    ordinary=(~torch.isnan(expected_max))&(expected_max!=0);assert torch.equal(actual_max[ordinary].view(torch.int32),expected_max[ordinary].view(torch.int32))
                    assert torch.equal(actual_max[expected_max==0].view(torch.int32),left[expected_max==0].view(torch.int32))
                    raw[label+'/maximum']={'actual':actual_max,'expected_torch':expected_max,'left':left,'right':right}
                    # Every live IEEE model/state bit, including payloads, survives checkpoint restore.
                    pattern=torch.tensor([0x7fc00123,-0x3ffedd,-2147483648,0x7f800000,-0x800000]+list(range(12)),dtype=torch.int32).view(torch.float32)
                    for value in resume.weights.values():value.copy_(pattern)
                    for pair in resume._optimizer_state.values():
                        pair['m'].copy_(pattern);pair['v'].copy_(torch.tensor([float('nan'),float('inf'),-0.]+[.1]*14));pair['max_v'].copy_(torch.tensor([float('nan'),float('inf'),0.]+[.2]*14))
                    ieee_before=snapshot(resume);ieee=root/(label+'-ieee.oa-checkpoint');resume.save_checkpoint(ieee,application=app(30))
                    resume.load_checkpoint(ieee);compare_state(resume,ieee_before,label+'/ieee-restored',True)
                    cases[-1]['maximum_nan_zero_control']=True;cases[-1]['maximum_profiler']=audit;cases[-1]['ieee_payload_roundtrip']=True
            finally:
                if owner:owner.close()
                if resume:resume.close()
                rt.close()
    # Repeat full G34 finite trajectories and ordinary CLI with the final maximum and checkpoint hosts.
    from optimizer_gate import run as optimizer_run
    optimizer_result=optimizer_run(root/'optimizer-options')
    torch.save(raw,root/'full-tensors.pt')
    package=Path(__import__('openassembler').__file__).parent
    result={'schema':1,'gap':'G35','passed':all(r['passed'] for r in records) and optimizer_result['passed'],
            'seed':35052,'python':platform.python_version(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'device':str(torch.cuda.get_device_properties(0)),
            'comparisons':records,'cases':cases,'negative_controls':negatives,'optimizer_gate':optimizer_result,
            'source_hashes':{p.relative_to(package).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in package.rglob('*.py')},
            'performance':'not measured by this full-state correctness/rollback gate'}
    (root/'report.json').write_text(json.dumps(result,indent=2)+'\n');return result
