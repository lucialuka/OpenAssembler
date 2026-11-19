"""OA-only full-state admission and shuffled paired full/region intervals."""
import json,os,sys,subprocess,time,statistics
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import torch
from openassembler.modelfront import kernels_tensor,kernels,runtime
from openassembler.modelfront.graph import capture_model,extract_weights
from openassembler.modelfront.optimization import optimize_plan
from openassembler.modelfront.reference import CPUReference
from openassembler.modelfront.profiling import audit_profile
from exp.ml_suite.tasks import TASKS,create
from exp.ml_suite.common import paired
from exp.model_perf.candidate import recipe,bias_reduce,warp_mse,frozen_module


def fields(run,result):
    return {'prediction':result['prediction'],'loss':result['loss'].reshape(()),
      **{'dx/'+n:v for n,v in result['gradients']['inputs'].items()},
      **{'dw/'+n:v for n,v in result['gradients']['parameters'].items()},
      **{'w/'+n:v for n,v in run.weights.items()}}


def case(task,production=False,integrated=False):
    torch.set_num_threads(1);torch.manual_seed(818);torch.backends.cuda.matmul.allow_tf32=False
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());torch.cuda.set_stream(stream)
    shape=TASKS[task]['shape'];training=TASKS[task]['training'];model=create(task)
    plan=optimize_plan(capture_model(model,[torch.randn(shape)*.25],math_mode=TASKS[task]['math']),reuse_storage=True)
    weights=extract_weights(model,plan);root=Path('results')/task;root.mkdir(parents=True,exist_ok=True)
    original=kernels_tensor.emit_tensor_recipe;active=['baseline'];runs={};stack=ExitStack()
    current_emit=kernels.emit_kernel;current_recipe=kernels_tensor.emit_tensor_recipe
    old_emit=kernels.emit_kernel if production and not integrated else frozen_module('kernels').emit_kernel
    if not production or integrated:original=frozen_module('tensor').emit_tensor_recipe
    def emit(op,*a,**k):
        if integrated and active[0]=='production':return current_emit(op,*a,**k)
        base=old_emit(op,*a,**k)
        if active[0] in ('reductions','all'):
            if op=='bias_grad' and base.inputs[0] in ((128,128),(128,256)):return bias_reduce(base)
            if op=='mse' and base.inputs[0]==(128,128):return warp_mse(base)
        return base
    if not production or integrated:
        stack.enter_context(patch.object(kernels,'emit_kernel',emit));stack.enter_context(patch.object(runtime,'emit_kernel',emit))
    if not production or integrated:stack.enter_context(patch.object(kernels_tensor,'emit_tensor_recipe',lambda *a,**k:current_recipe(*a,**k) if integrated and active[0]=='production' else recipe(original,'tile' if active[0]=='all' else active[0],*a,**k)))
    variants=(['baseline','tile','reductions','all'] if os.environ.get('OA_PERF_REDUCTIONS') else ['baseline','pack','tile','combined']) if not production else (['baseline','production'] if integrated else ['production'])
    for v in variants:
        active[0]=v;t=time.perf_counter();kernels.build_bundle(plan,weights,root/v,sass_listings=False,cache=False)
        runs[v]=runtime.Runtime.load(root/v)
    input_name=plan['inputs'][0]['name'];output=plan['outputs'][0];outshape=plan['buffers'][output]['shape']
    x=torch.empty(shape,device='cuda');target=torch.empty(outshape,device='cuda');reference=CPUReference(plan,weights);checks=[];audits={};graphs={};results={}
    def step(v):
        active[0]=v;r=runs[v]
        return r.train_step({input_name:x},target,.01) if training else r.forward({input_name:x})
    def check(v,result,want,it):
        actual=fields(runs[v],result) if training else {'prediction':result[output]}
        expected=({'prediction':want['prediction'],'loss':want['loss'].reshape(()),
          **{'dx/'+n:t for n,t in want['gradients']['inputs'].items()},**{'dw/'+n:t for n,t in want['gradients']['parameters'].items()},
          **{'w/'+n:t for n,t in reference.weights.items()}} if training else {'prediction':want[output]})
        for field,a in actual.items():
            e=expected[field];a=a.detach().cpu();good=torch.isclose(a,e,rtol=.002 if plan['math']!='fp32' else 1e-4,atol=2e-5 if plan['math']!='fp32' else 1e-5)
            checks.append({'variant':v,'iteration':it,'field':field,'passed':bool(good.all()),'max_abs':float((a-e).abs().max())})
            assert bool(good.all()),checks[-1]
        runs[v].check_canaries()
    for it in range(4):
        cx=torch.randn(shape)*.25;cy=torch.randn(outshape)*.25;x.copy_(cx);target.copy_(cy)
        want=reference.train_step({input_name:cx},cy,.01) if training else reference.forward({input_name:cx})
        for v,r in runs.items():
            if it:r.poison_workspace()
            result=step(v);r.synchronize();check(v,result,want,it)
            assert torch.equal(x.cpu(),cx) and torch.equal(target.cpu(),cy)
    for v,r in runs.items():
        active[0]=v
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            step(v);r.synchronize()
        audits[v]=audit_profile(prof,r.launch_ledger)
        with torch.no_grad():
            for n,w in r.weights.items():w.copy_(weights[n].cuda())
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=stream):results[v]=step(v)
        graphs[v]=g
    reference=CPUReference(plan,weights)
    for it in range(3):
        cx=torch.randn(shape)*.25;cy=torch.randn(outshape)*.25;x.copy_(cx);target.copy_(cy)
        want=reference.train_step({input_name:cx},cy,.01) if training else reference.forward({input_name:cx})
        for v,r in runs.items():
            r.poison_workspace();graphs[v].replay();r.synchronize();check(v,results[v],want,it+4)
            assert torch.equal(x.cpu(),cx) and torch.equal(target.cpu(),cy)
    counts={v:len(r.launch_ledger) for v,r in runs.items()}
    total=paired({v:lambda v=v:step(v) for v in runs},repeats=4,samples=15,rounds=3)
    parts={}
    for phase in ['forward','backward'] if training else ['forward']:
        funcs={}
        for v,r in runs.items():
            values=dict(r._last)
            if training:values[plan['backward']['seed_grads'][output]]=r._training[output]['seed']
            def fn(v=v,r=r,values=values,phase=phase):
                active[0]=v;r._packed_ready.clear();r._run(phase,values)
            funcs[v]=fn
        parts[phase]=paired(funcs,repeats=4,samples=15,rounds=3)
    if training:
        functions={}
        for v,r in runs.items():
            def loss(v=v,r=r):
                active[0]=v;s=r.specs[output];prediction=r._last[output]
                r._launch(kernels.emit_kernel('mse',[s]*2,{'shape':[]}),[prediction,target],r._training[output]['loss'])
                r._launch(kernels.emit_kernel('mse_grad',[s]*2,s),[prediction,target],r._training[output]['seed'])
            functions[v]=loss
        parts['loss_and_seed']=paired(functions,repeats=4,samples=15,rounds=3)
    # Inventory emitted shapes, launch counts and per-kernel diagnostics.
    inventory={v:[{'op':s.op,'name':s.name,'signature':s.signature,'launch':s.launch} for rec in r.recipes.values() for s in rec.stages] for v,r in runs.items()}
    record={'passed':True,'task':task,'production':production,'device':torch.cuda.get_device_name(),'torch':torch.__version__,
      'environment':environment(),'checks':checks,'audits':audits,'launches':counts,'total':total,'parts':parts,'inventory':inventory,
      'scope':'resident CUDA graph stream time; compilation/preparation/transfers excluded; region intervals independent'}
    (root/'result.json').write_text(json.dumps(record,indent=2))
    for r in runs.values():r.close()
    stack.close();return record


def environment():
    from openassembler import oracle
    r=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],capture_output=True,text=True)
    return {'gpu':r.stdout,'cuda':torch.version.cuda,'tools':{n:oracle.run_tool(n,'--version').stdout for n in ('ptxas','nvdisasm')}}


def main(phase):
    Path('results').mkdir(exist_ok=True)
    if phase=='cache':
        from exp.model_perf.cache_gate import main as cache_gate
        cache_gate();return
    if phase=='tests':
        p=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_model_kernels.py','tests/test_model_graph.py','tests/test_model_runtime.py','tests/test_model_training_kernels.py','tests/test_model_tensor.py','tests/test_model_tensor_schedule.py','tests/test_model_cache_tools.py'],capture_output=True,text=True,timeout=300)
        Path('results/result.json').write_text(json.dumps({'passed':p.returncode==0,'returncode':p.returncode,'output':p.stdout+p.stderr},indent=2));return
    if phase=='comparisons':
        from exp.ml_suite.bench_campaign import worker
        records=[]
        for task in ['dense_regression_fp16','projection_inference','dense_regression_fp32']:
            p=subprocess.run([sys.executable,'-m','exp.model_perf.campaign','comparison:'+task],capture_output=True,text=True,timeout=450)
            Path('results',task+'-comparisons.log').write_text(p.stdout+p.stderr)
            if p.returncode:raise RuntimeError(p.stdout+p.stderr)
            records.append(json.loads(Path('results/bench',task,'result.json').read_text()))
        Path('results/result.json').write_text(json.dumps({'passed':True,'records':records},indent=2));return
    if phase.startswith('comparison:'):
        from exp.ml_suite.bench_campaign import worker
        worker(phase.split(':')[1]);return
    if phase=='gemm':
        from exp.model_perf.gemm_gate import main as gate
        gate();return
    if phase=='operators':
        from exp.model_perf.operators import main as operators
        operators();return
    if ':' in phase:
        kind,task=phase.split(':');case(task,kind in ('production','integrated'),kind=='integrated');return
    tasks=['dense_regression_fp16','projection_inference','dense_regression_fp32'] if phase in ('confirm','production','integrated') else ['dense_regression_fp16','projection_inference'];records=[]
    for task in tasks:
        p=subprocess.run([sys.executable,'-m','exp.model_perf.campaign',('production:' if phase=='production' else 'integrated:' if phase=='integrated' else 'case:')+task],capture_output=True,text=True,timeout=600)
        Path('results',task+'.log').write_text(p.stdout+p.stderr)
        if p.returncode:raise RuntimeError(p.stdout+p.stderr)
        records.append(json.loads(Path('results',task,'result.json').read_text()))
    Path('results/result.json').write_text(json.dumps({'passed':True,'records':records},indent=2))
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'experiment')
