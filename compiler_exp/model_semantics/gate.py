"""A100 gate: full outputs/derivatives/state, poison, guards and OA provenance."""
import copy
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import torch
from test_semantics import Broadcast,Attention,TokenModel
from openassembler.modelfront.graph import capture_model,extract_weights
from openassembler.modelfront.kernels import build_bundle,prepare
from openassembler.modelfront.optimization import optimize_plan
from openassembler.modelfront.runtime import Runtime


def run(root):
    torch.set_num_threads(1);torch.manual_seed(73);torch.backends.cuda.matmul.allow_tf32=False
    records=[];raw={};plans={};archive=io.BytesIO()
    rtol,atol=1e-4,1e-5
    def check(got,expected,label):
        a=got.detach().cpu();b=expected.detach().cpu()
        close=torch.isclose(a,b,rtol=rtol,atol=atol)
        row={'name':label,'elements':a.numel(),'passed':bool(close.all()),'mismatches':int((~close).sum()),'max_abs':float((a-b).abs().max())}
        records.append(row);raw[label]={'actual':a,'expected':b}
        if not row['passed']:
            torch.save(raw,root/'failure-full-tensors.pt')
            raise AssertionError(row)
    def make(model,args,label,views=True,reuse=True,**policy):
        plan=capture_model(model,args);plan.update(policy)
        plan=optimize_plan(plan,storage_views=views,reuse_storage=reuse)
        out=root/label;build_bundle(plan,extract_weights(model,plan),out)
        runtime=Runtime.load(out);plans[label]={'plan':plan,'ledger':json.loads((out/'native-ledger.json').read_text())} if (out/'native-ledger.json').exists() else {'plan':plan}
        return plan,runtime
    runtime_records=[]
    def test(model,args,label,views,reuse):
        model=copy.deepcopy(model)
        plan,rt=make(model,args,label,views,reuse)
        try:
            for iteration in range(2):
                xs=tuple(x.detach().clone().requires_grad_(x.dtype==torch.float32) for x in args)
                if iteration:
                    rt.poison_workspace()
                    xs=tuple((x+.03125).detach().requires_grad_(True) if x.dtype==torch.float32 else x for x in xs)
                gpu={s['name']:x.detach().cuda() for s,x in zip(plan['inputs'],xs)}
                before={n:v.cpu().clone() for n,v in gpu.items()}
                model.zero_grad(set_to_none=True);expected=model(*xs)
                got=rt.forward(gpu,require_tape=True);name=plan['outputs'][0]
                check(got[name],expected,f'{label}/{iteration}/output')
                seed=torch.randn_like(expected);expected.backward(seed)
                grad=rt.backward({name:seed.cuda()})
                for s,x in zip(plan['inputs'],xs):
                    if x.grad is not None:check(grad['inputs'][s['name']],x.grad,f'{label}/{iteration}/dx/{s["name"]}')
                for s in plan['parameters']:
                    value=dict(model.named_parameters())[s['target']]
                    if value.grad is not None:check(grad['parameters'][s['name']],value.grad,f'{label}/{iteration}/dw/{s["name"]}')
                for n,v in gpu.items():assert torch.equal(v.cpu(),before[n])
                runtime_records.append({'name':label,'iteration':iteration,'guards':rt.check_canaries(),'launches':list(rt.launch_ledger)})
        finally:rt.close()
    for views,reuse in ((False,False),(False,True),(True,False),(True,True)):
        suffix=f'{int(views)}{int(reuse)}'
        for a,b in [((),(2,3)),((3,),(2,3)),((2,1,3),(1,4,1)),((1,3,1,5),(2,1,7,1))]:
            test(Broadcast(),(torch.randn(a),torch.randn(b)),f'broadcast-{a}-{b}-{suffix}',views,reuse)
        for causal,width,qlen in [(False,1,1),(False,33,3),(False,513,2),(True,7,4)]:
            mask=torch.rand(qlen,width)>.4;mask[0]=False
            test(Attention(causal),(torch.randn(2,2,qlen,5)*.25,torch.randn(2,2,width,5)*.25,torch.randn(2,2,width,3)*.25,mask),f'attention-{causal}-{width}-{suffix}',views,reuse)
        test(TokenModel(),(torch.tensor([[0,1,1,10],[3,3,0,7]]),),f'language-{suffix}',views,reuse)
    training=[]
    for kind in ('sgd','adam','adamw'):
        model=TokenModel();args=(torch.tensor([[0,1,1,10],[3,3,0,7]]),)
        options={'name':kind}
        if kind!='sgd':options.update(beta1=.8,beta2=.95,epsilon=1e-8,weight_decay=.03)
        plan,rt=make(model,args,'training-'+kind,loss='cross_entropy',optimizer=options,optimizer_fusion=True)
        cls={'sgd':torch.optim.SGD,'adam':torch.optim.Adam,'adamw':torch.optim.AdamW}[kind]
        kwargs={} if kind=='sgd' else dict(betas=(.8,.95),eps=1e-8,weight_decay=.03,foreach=False)
        opt=cls(model.parameters(),lr=.001,**kwargs)
        resume=None;resume_dir=root/('training-'+kind)
        try:
            for iteration in range(8):
                tokens=torch.randint(0,11,(2,4));tokens[0,1]=tokens[0,0];labels=torch.randint(0,11,(8,))
                opt.zero_grad(set_to_none=True);logits=model(tokens);loss=torch.nn.functional.cross_entropy(logits,labels);loss.backward()
                if iteration:rt.poison_workspace()
                gpu={'tokens':tokens.cuda()};target=labels.cuda()
                got=rt.train_step(gpu,target,.001)
                check(got['prediction'],logits,f'{kind}/{iteration}/logits');check(got['loss'].reshape(()),loss,f'{kind}/{iteration}/loss')
                for p in plan['parameters']:
                    param=dict(model.named_parameters())[p['target']]
                    check(got['gradients']['parameters'][p['name']],param.grad,f'{kind}/{iteration}/grad/{p["name"]}')
                opt.step();state=rt.optimizer_state_dict()
                assert state['step']==iteration+1
                for p in plan['parameters']:
                    param=dict(model.named_parameters())[p['target']]
                    check(rt.weights[p['name']],param,f'{kind}/{iteration}/weight/{p["name"]}')
                    if kind!='sgd':
                        for key,refkey in [('m','exp_avg'),('v','exp_avg_sq')]:check(state['moments'][p['name']][key],opt.state[param][refkey],f'{kind}/{iteration}/{key}/{p["name"]}')
                if iteration==3:
                    snap={n:w.detach().cpu().clone() for n,w in rt.weights.items()};checkpoint=state
                    resume=Runtime(plan,{n:w.cuda() for n,w in snap.items()},resume_dir)
                    resume.forward(gpu);resume.discard_tape();resume.load_optimizer_state_dict(checkpoint)
                elif iteration>3:
                    other=resume.train_step(gpu,target,.001)
                    check(other['loss'],got['loss'],f'{kind}/{iteration}/resume/loss')
                    for n,w in rt.weights.items():check(resume.weights[n],w,f'{kind}/{iteration}/resume/weight/{n}')
                    rs=resume.optimizer_state_dict()
                    for n,pair in state['moments'].items():
                        for key,value in pair.items():check(rs['moments'][n][key],value,f'{kind}/{iteration}/resume/{n}/{key}')
                training.append({'optimizer':kind,'step':iteration+1,'guards':rt.check_canaries(),'launches':list(rt.launch_ledger)})
            before={n:w.cpu().clone() for n,w in rt.weights.items()};step=rt._optimizer_step
            for bad in (-1,11):
                invalid=torch.full((2,4),bad,dtype=torch.int64,device='cuda')
                try:rt.train_step({'tokens':invalid},target,.001)
                except ValueError:pass
                else:raise AssertionError('bad embedding index admitted')
                invalid_target=torch.full((8,),bad,dtype=torch.int64,device='cuda')
                try:rt.train_step(gpu,invalid_target,.001)
                except ValueError:pass
                else:raise AssertionError('bad class label admitted')
            assert rt._optimizer_step==step and all(torch.equal(rt.weights[n].cpu(),w) for n,w in before.items())
        finally:
            rt.close()
            if resume:resume.close()
    from decode_gate import check_decode
    decode=check_decode(root,make,check)
    from optimizer_bench import benchmark
    timing=benchmark(root,make,check)
    from loss_gate import run as loss_boundaries
    loss_checks=loss_boundaries(make,check)
    # Timing records are explicitly host-inclusive diagnostic observations, not speedup admission.
    report={'schema':1,'passed':True,'device':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,
            'gate':{'rtol':rtol,'atol':atol,'frozen_before_execution':True},'optimizer_timing':timing,'loss_boundary_guards':loss_checks,'checks':records,'runtime':runtime_records,'training':training,'decode':decode}
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        f=io.BytesIO();torch.save(raw,f);z.writestr('full-tensors.pt',f.getvalue());z.writestr('plans.json',json.dumps(plans))
        for out in root.iterdir():
            if out.is_dir():
                for p in out.rglob('*'):
                    if p.is_file() and p.suffix in ('.json','.ptx','.cubin'):z.write(p,p.relative_to(root))
    return report,archive.getvalue()
