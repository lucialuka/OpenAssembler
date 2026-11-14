"""Full FP16 BMM/attention training with independent CPU operand-rounding checks."""
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import json,sys,subprocess,runpy
import torch
from openassembler.modelfront import kernels,runtime
from openassembler.modelfront.graph import capture_model,extract_weights
from openassembler.modelfront.optimization import optimize_plan
from openassembler.modelfront.reference import CPUReference
from openassembler.modelfront.profiling import audit_profile
from exp.attention_half.candidate import recipe, _tensor_recipe


class Reference(CPUReference):
    def _steps(self,steps):
        for step in steps:
            if step['kernel'] in ('bmm','bmm_dx','bmm_dw'):
                names=step['inputs'];old={n:self.state[n] for n in names}
                for n in names:self.state[n]=self.state[n].half().float()
                super()._steps([step]);self.state.update(old)
            else:super()._steps([step])

class Attention(torch.nn.Module):
    def __init__(self,b,s,h,d):
        super().__init__();self.b=b;self.s=s;self.h=h;self.d=d;w=h*d
        self.q=torch.nn.Linear(w,w);self.k=torch.nn.Linear(w,w);self.v=torch.nn.Linear(w,w);self.o=torch.nn.Linear(w,w)
    def forward(self,x):
        q=self.q(x).reshape(self.b,self.s,self.h,self.d).permute(0,2,1,3)
        k=self.k(x).reshape(self.b,self.s,self.h,self.d).permute(0,2,1,3)
        v=self.v(x).reshape(self.b,self.s,self.h,self.d).permute(0,2,1,3)
        y=torch.softmax((q@k.transpose(-2,-1))*(self.d**-.5),dim=-1)@v
        return self.o(y.permute(0,2,1,3).reshape(self.b,self.s,self.h*self.d))+x


def case(name,production=False,all_tensor=False):
    torch.manual_seed(818);torch.set_num_threads(1)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());torch.cuda.set_stream(stream)
    config={'tiny':(2,7,2,4),'tail':(1,17,3,5),'medium':(2,33,2,16)}
    if name=='block':model=runpy.run_path('examples/native_attention_model/model.py')['create_model']();shape=(2,7,8)
    else:
        b,s,h,d=config[name];model=Attention(b,s,h,d);shape=(b,s,h*d)
    plan=optimize_plan(capture_model(model,[torch.randn(shape)*.15],math_mode='f16_rne_f32_accum' if production else 'fp32'),reuse_storage=True)
    plan['math']='f16_rne_f32_accum';plan['tensor_packing']='dual'
    weights=extract_weights(model,plan);root=Path('results')/(name+('-tensor' if all_tensor else ''));root.mkdir(parents=True,exist_ok=True)
    stack=ExitStack();old_recipes=kernels.tensor_recipes;old_prepare=kernels.prepare
    def recipes(p,specs=None):
        result=old_recipes(p,specs);specs=specs or kernels.tensor_specs(p)
        for st in p['steps']+p['backward']['steps']:
            if st['kernel'].startswith('bmm'):
                rec=(_tensor_recipe if all_tensor else recipe)(st['kernel'],[specs[n] for n in st['inputs']],specs[st['output']]);result[rec.name]=rec
        return result
    def prepare(p):
        specs,steps,ks=old_prepare(p)
        for st in steps:
            if st['kernel'].startswith('bmm'):
                rec=(_tensor_recipe if all_tensor else recipe)(st['kernel'],[specs[n] for n in st['inputs']],specs[st['output']]);ks.pop(st['entry'],None);st['entry']=rec.name;st['tensor']=True
        return specs,steps,ks
    if not production:
        stack.enter_context(patch.object(kernels,'tensor_recipes',recipes));stack.enter_context(patch.object(runtime,'tensor_recipes',recipes))
        stack.enter_context(patch.object(kernels,'prepare',prepare));stack.enter_context(patch.object(runtime,'prepare',prepare))
    kernels.build_bundle(plan,weights,root/'bundle',sass_listings=False,cache=False);run=runtime.Runtime.load(root/'bundle')
    input_name=plan['inputs'][0]['name'];x=torch.empty(shape,device='cuda');target=torch.empty(shape,device='cuda')
    ref=(CPUReference if production else Reference)(plan,weights);checks=[]
    def check(result,want,it):
        fields={'prediction':result['prediction'],'loss':result['loss'].reshape(()),
                **{'dx/'+n:v for n,v in result['gradients']['inputs'].items()},**{'dw/'+n:v for n,v in result['gradients']['parameters'].items()},**{'w/'+n:v for n,v in run.weights.items()}}
        expected={'prediction':want['prediction'],'loss':want['loss'].reshape(()),
                  **{'dx/'+n:v for n,v in want['gradients']['inputs'].items()},**{'dw/'+n:v for n,v in want['gradients']['parameters'].items()},**{'w/'+n:v for n,v in ref.weights.items()}}
        for n,v in fields.items():
            a=v.detach().cpu();e=expected[n];good=torch.isclose(a,e,rtol=.002,atol=2e-4 if name=='block' else 2e-5)
            assert bool(good.all()),(name,it,n,int((~good).sum()),float((a-e).abs().max()))
            checks.append({'iteration':it,'field':n,'max_abs':float((a-e).abs().max()),'strict_2e5_failures':int((~torch.isclose(a,e,rtol=.002,atol=2e-5)).sum()),'passed':True})
        run.check_canaries()
    for it in range(4):
        cx=torch.randn(shape)*.15;cy=torch.randn(shape)*.15;x.copy_(cx);target.copy_(cy)
        if it:run.poison_workspace()
        result=run.train_step({input_name:x},target,.01);run.synchronize();check(result,ref.train_step({input_name:cx},cy,.01),it)
        assert torch.equal(x.cpu(),cx) and torch.equal(target.cpu(),cy)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        run.train_step({input_name:x},target,.01);run.synchronize()
    audit=audit_profile(prof,run.launch_ledger)
    with torch.no_grad():
        for n,w in run.weights.items():w.copy_(weights[n].cuda())
    ref=(CPUReference if production else Reference)(plan,weights)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=stream):result=run.train_step({input_name:x},target,.01)
    for it in range(3):
        cx=torch.randn(shape)*.15;cy=torch.randn(shape)*.15;x.copy_(cx);target.copy_(cy);run.poison_workspace();g.replay();run.synchronize()
        check(result,ref.train_step({input_name:cx},cy,.01),it+4)
    rows=[]
    for _ in range(15):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
        for __ in range(4):g.replay()
        b.record();b.synchronize();rows.append(a.elapsed_time(b)/4)
    report={'passed':True,'case':name,'variant':'tensor' if all_tensor else 'hybrid','production':production,'device':torch.cuda.get_device_name(),'checks':checks,'audit':audit,'launches':len(run.launch_ledger),'samples_ms':rows,'tolerance':{'rtol':.002,'atol':2e-4 if name=='block' else 2e-5},'scope':'bounded composed unmasked attention; FP16 RNE BMM forward/dX/dW, FP32 softmax/master state; complete SGD replay'}
    (root/'result.json').write_text(json.dumps(report,indent=2));run.close();stack.close()


def bmm_cases(production=False):
    import statistics
    class BMM(torch.nn.Module):
        def forward(self,a,b): return torch.bmm(a,b)
    records=[]
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());torch.cuda.set_stream(stream)
    for batch,m,n,k in [(1,1,1,1),(3,7,9,5),(2,17,33,65),(2,128,128,128),(4,129,67,31),(1,256,64,128),(32,2,5,3),(2,31,33,512),(1,511,3,17)]:
        torch.manual_seed(719);torch.set_num_threads(1)
        a,b=torch.randn(batch,m,k)*.1,torch.randn(batch,k,n)*.1
        plan=capture_model(BMM(),[a,b],math_mode='f16_rne_f32_accum' if production else 'fp32')
        old_recipes=kernels.tensor_recipes;old_prepare=kernels.prepare
        def recipes(p,specs=None):
            specs=specs or kernels.tensor_specs(p)
            return {r.name:r for st in p['steps']+p['backward']['steps'] for r in [selected_recipe(st['kernel'],[specs[x] for x in st['inputs']],specs[st['output']])]}
        def prepare(p):
            specs,steps,ks=old_prepare(p)
            for st in steps:
                if st['kernel'].startswith('bmm'):
                    r=selected_recipe(st['kernel'],[specs[x] for x in st['inputs']],specs[st['output']]);ks.pop(st['entry'],None);st.update(entry=r.name,tensor=True)
            return specs,steps,ks
        root=Path('results/bmm')/f'{batch}-{m}-{n}-{k}';root.mkdir(parents=True,exist_ok=True)
        runs=[];times={};errors=[]
        for mode in ['fp32','tensor','half']:
            selected_recipe=_tensor_recipe if mode=='tensor' else recipe
            plan['math']='fp32' if mode=='fp32' else 'f16_rne_f32_accum'
            plan['tensor_packing']='separate' if mode=='fp32' or production and mode=='half' else 'dual'
            with ExitStack() as stack:
                if mode!='fp32' and (not production or mode=='tensor'):
                    for module in [kernels,runtime]:
                        stack.enter_context(patch.object(module,'tensor_recipes',recipes));stack.enter_context(patch.object(module,'prepare',prepare))
                kernels.build_bundle(plan,{},root/mode,sass_listings=False,cache=False);r=runtime.Runtime.load(root/mode)
                ca,cb=a.cuda(),b.cuda();dy=torch.randn(batch,m,n)*.1;cd=dy.cuda();output=plan['outputs'][0]
                def step():
                    y=r.forward([ca,cb]);gr=r.backward({output:cd});return y,gr
                result=step();r.synchronize()
                ref=(CPUReference if production or mode=='fp32' else Reference)(plan,{})
                want=ref.forward([a,b]);grad=ref.backward({output:dy})
                for got,expected in [(result[0][output],want[output])]+[(v,grad['inputs'][q]) for q,v in result[1]['inputs'].items()]:
                    torch.testing.assert_close(got.cpu(),expected,rtol=.001,atol=2e-6);errors.append(float((got.cpu()-expected).abs().max()))
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream): step()
                for _ in range(3):
                    ca.copy_(a);cb.copy_(b);r.poison_workspace();graph.replay();r.synchronize();r.check_canaries()
                    torch.testing.assert_close(result[0][output].cpu(),want[output],rtol=.001,atol=2e-6)
                samples=[]
                for _ in range(15):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
                    for __ in range(10):graph.replay()
                    end.record();end.synchronize();samples.append(start.elapsed_time(end)/10)
                times[mode]=samples;r.close()
        records.append({'shape':[batch,m,n,k],'max_abs':max(errors),'samples_ms':times,'half_over_fp32_speedup':statistics.median(times['fp32'])/statistics.median(times['half']),'hybrid_over_tensor_speedup':statistics.median(times['tensor'])/statistics.median(times['half'])})
    Path('results/bmm.json').write_text(json.dumps({'passed':True,'records':records},indent=2))


def main(phase):
    Path('results').mkdir(exist_ok=True)
    if phase=='cpu-tests':
        p=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_model_graph.py','tests/test_model_batched_tensor.py','tests/test_torchfront_numerics.py','tests/test_native_ownership.py'],capture_output=True,text=True,timeout=240)
        Path('results/pytest.log').write_text(p.stdout+p.stderr)
        Path('results/result.json').write_text(json.dumps({'passed':p.returncode==0,'returncode':p.returncode,'output':p.stdout+p.stderr}));return
    if phase in ('bmm','production-bmm'):bmm_cases(phase=='production-bmm');Path('results/result.json').write_text(Path('results/bmm.json').read_text());return
    if phase=='paired':
        import statistics
        records=[]
        for turn in range(2):
            for name in ['tiny','medium','block']:
                timings={}
                for mode in (['tensor','production'] if turn==0 else ['production','tensor']):
                    p=subprocess.run([sys.executable,'-m','exp.attention_half.campaign',mode+':'+name],capture_output=True,text=True,timeout=300)
                    Path('results',f'{turn}-{mode}-{name}.log').write_text(p.stdout+p.stderr)
                    if p.returncode:raise RuntimeError(p.stdout+p.stderr)
                    data=json.loads(Path('results',name+('-tensor' if mode=='tensor' else ''),'result.json').read_text())
                    Path('results',f'{turn}-{mode}-{name}.json').write_text(json.dumps(data,indent=2));timings[mode]=data
                records.append({'round':turn,'case':name,'speedup':statistics.median(timings['tensor']['samples_ms'])/statistics.median(timings['production']['samples_ms']),'timings':timings})
        Path('results/result.json').write_text(json.dumps({'passed':True,'records':records},indent=2));return
    if ':' in phase:kind,name=phase.split(':');case(name,kind=='production',kind=='tensor');return
    records=[]
    for name in ['tiny','tail','medium','block']:
        p=subprocess.run([sys.executable,'-m','exp.attention_half.campaign',('production:' if phase=='production' else 'case:')+name],capture_output=True,text=True,timeout=300)
        Path('results',name+'.log').write_text(p.stdout+p.stderr)
        if p.returncode:raise RuntimeError(p.stdout+p.stderr)
        records.append(json.loads(Path('results',name,'result.json').read_text()))
    Path('results/result.json').write_text(json.dumps({'passed':True,'records':records},indent=2))
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'experiment')
