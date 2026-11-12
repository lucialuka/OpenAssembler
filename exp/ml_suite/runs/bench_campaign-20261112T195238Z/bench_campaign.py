"""Four independent compiler/library arms, complete tasks and parts on A100."""
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time


def worker(name):
    import torch
    import triton
    import inspect
    import triton.language as tl
    from openassembler.modelfront.graph import capture_model,extract_weights
    from openassembler.modelfront.optimization import optimize_plan
    from openassembler.modelfront.kernels import build_bundle,emit_kernel
    from openassembler.modelfront.runtime import Runtime
    from openassembler.modelfront.reference import CPUReference
    from openassembler.modelfront.profiling import audit_profile
    from exp.ml_suite.tasks import TASKS,create
    from exp.ml_suite.engines import Engine
    from exp.ml_suite.torch_eager import TorchEager
    from exp.ml_suite.common import paired
    torch.set_num_threads(1);torch.manual_seed(818)
    benchmark_stream=torch.cuda.Stream();benchmark_stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(benchmark_stream)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    spec=TASKS[name];shape=spec['shape'];half=spec['math']!='fp32'
    root=Path('results/bench')/name;root.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();model=create(name)
    plan=optimize_plan(capture_model(model,[torch.randn(shape)*.25],math_mode=spec['math']),reuse_storage=True)
    weights=extract_weights(model,plan);capture_ms=(time.perf_counter()-started)*1000
    started=time.perf_counter();build_bundle(plan,weights,root/'oa',sass_listings=False)
    build_ms=(time.perf_counter()-started)*1000
    initial=time.perf_counter()
    native=Runtime.load(root/'oa');input_name=plan['inputs'][0]['name']
    sample=torch.randn(shape)*.25;target=torch.randn(shape)*.25
    oa_inputs={input_name:sample.cuda()};oa_target=target.cuda()
    oa_outputs=native.forward(oa_inputs)
    setup={'openassembler':{'bundle_build_ms':build_ms,'initial_runtime_ms':(time.perf_counter()-initial)*1000}}
    engines={}
    for kind in ['triton','nvidia_same_ptx','nvidia_libraries_eager']:
        before=time.perf_counter()
        engine=TorchEager(plan,weights,kind) if kind=='nvidia_libraries_eager' else Engine(plan,weights,root/'oa',kind)
        with torch.no_grad():engine.inputs[input_name].copy_(sample);engine.target.copy_(target)
        with torch.no_grad():
            if spec['training']:engine.train()
            else:engine.forward()
        torch.cuda.synchronize()
        if hasattr(engine,'reset'):engine.reset(weights)
        else:
            with torch.no_grad():
                for n,v in engine.weights.items():v.copy_(weights[n].cuda())
        setup[kind]={'load_compile_warm_ms':(time.perf_counter()-before)*1000}
        engines[kind]=engine
    native.discard_tape()
    reference=CPUReference(plan,weights)
    checks=[];audits={};profiler_parts={}

    def native_train():
        result=native.train_step(oa_inputs,oa_target,.01)
        nonlocal last_native_result
        last_native_result=result
        return result
    last_native_result=None

    def native_fields(result):
        return {'prediction':result['prediction'],'loss':result['loss'].reshape(()),
                **{'dx/'+n:v for n,v in result['gradients']['inputs'].items()},
                **{'dw/'+n:v for n,v in result['gradients']['parameters'].items()},
                **{'w/'+n:v for n,v in native.weights.items()}}

    def check(method,actual,expected,iteration):
        for field,value in actual.items():
            cpu=value.detach().cpu();truth=expected[field].detach().cpu()
            close=torch.isclose(cpu,truth,rtol=.002 if half else 1e-4,atol=2e-5 if half else 1e-5)
            record={'iteration':iteration,'backend':method,'field':field,'passed':bool(close.all()),
                    'mismatches':int((~close).sum()),'max_abs':float((cpu-truth).abs().max())}
            checks.append(record);assert record['passed'],record

    for iteration in range(4):
        x,y=torch.randn(shape)*.25,torch.randn(shape)*.25
        oa_inputs[input_name].copy_(x);oa_target.copy_(y)
        if spec['training']:
            step=reference.train_step({input_name:x},y,.01)
            expected={'prediction':step['outputs'][plan['outputs'][0]],'loss':step['loss'].reshape(()),
                **{'dx/'+n:v for n,v in step['gradients']['inputs'].items()},
                **{'dw/'+n:v for n,v in step['gradients']['parameters'].items()},
                **{'w/'+n:v for n,v in reference.weights.items()}}
        else:expected={'prediction':reference.forward({input_name:x})[plan['outputs'][0]]}
        if iteration:native.poison_workspace()
        with torch.no_grad():
            result=native_train() if spec['training'] else native.forward(oa_inputs)
        native.synchronize()
        check('openassembler',native_fields(result) if spec['training'] else {'prediction':result[plan['outputs'][0]]},expected,iteration)
        native.check_canaries()
        assert torch.equal(oa_inputs[input_name].cpu(),x) and torch.equal(oa_target.cpu(),y)
        for kind,engine in engines.items():
            with torch.no_grad():engine.inputs[input_name].copy_(x);engine.target.copy_(y)
            if kind=='triton' and name=='attention_training':engine.capture_snapshots=(iteration==0);engine.snapshots=[]
            if kind=='nvidia_libraries_eager' and name=='attention_training':engine.capture_snapshots=(iteration==0)
            with torch.no_grad():
                if spec['training']:engine.train()
                else:engine.forward()
            torch.cuda.synchronize();engine.check_guards()
            output=(engine.output_value if kind=='nvidia_libraries_eager' else engine.state[engine.output])
            try:check(kind,engine.fields() if spec['training'] else {'prediction':output},expected,iteration)
            except AssertionError as exc:
                if kind=='triton' and name=='attention_training':
                    diffs=[]
                    for key,actual in engine.snapshots:
                        truth=reference.state.get(key)
                        if truth is not None and tuple(actual.shape)==tuple(truth.shape):
                            diffs.append((float((actual-truth).abs().max()),key,tuple(actual.shape)))
                    exc.args=(f'{exc}; largest intermediate deltas: {sorted(diffs,reverse=True)[:20]}',)
                    if not any(k=='_oa_grad_copy_25' for _,k,_ in diffs): pass
                    exc.args=(exc.args[0]+f'; inputs to first bad bmm dX: '+str([(k, next((d for d in diffs if d[1]==k),None)) for k in ('_oa_grad_copy_25','_oa_bmm_right_11')]),)
                if kind=='nvidia_libraries_eager' and name=='attention_training':
                    diffs=[]
                    for key,actual in engine.snapshots:
                        truth=reference.state.get(key)
                        if truth is not None and tuple(actual.shape)==tuple(truth.shape):diffs.append((float((actual-truth).abs().max()),key,tuple(actual.shape)))
                    exc.args=(f'{exc}; eager intermediate deltas: {sorted(diffs,reverse=True)[:20]}',)
                raise
            assert torch.equal(engine.inputs[input_name].cpu(),x) and torch.equal(engine.target.cpu(),y)
    # Independent GPU traces describe one complete invocation, separate from
    # the distributions and CUDA-graph region times below.
    functions={'openassembler':native_train if spec['training'] else lambda:native.forward(oa_inputs),
               **{k:(e.train if spec['training'] else e.forward) for k,e in engines.items()}}
    for kind,fn in functions.items():
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
            with torch.no_grad():fn()
            torch.cuda.synchronize()
        trace=root/(kind+'-trace.json');profile.export_chrome_trace(str(trace))
        events=json.loads(trace.read_text())['traceEvents']
        kernels=[e for e in events if e.get('cat')=='kernel']
        assert kernels,'empty GPU trace'
        if kind=='openassembler':
            from openassembler.modelfront.profiling import audit_trace
            audits[kind]=audit_trace(events,native.launch_ledger)
        profiler_parts[kind]=[{'name':e['name'],'duration_us':e.get('dur'),'stream':e.get('args',{}).get('stream')} for e in kernels]
    overall=paired(functions,repeats=4,samples=15,rounds=3,bound_runtimes={'openassembler':native})
    # Uncaptured resident host path is independently timed; compilation and DMA
    # are excluded. Explicit host distributions are never summed with GPU times.
    host_samples={k:[] for k in functions};rng=random.Random(993)
    for _ in range(15):
        order=list(functions);rng.shuffle(order)
        for kind in order:
            torch.cuda.synchronize();before=time.perf_counter()
            with torch.no_grad():functions[kind]()
            torch.cuda.synchronize();host_samples[kind].append((time.perf_counter()-before)*1000)
    # Native parts use the exact admitted functions and seed bindings, without
    # adding public tape checks to each captured component repetition.
    values=dict(native._last)
    if spec['training']:
        values[plan['backward']['seed_grads'][plan['outputs'][0]]]=native._training[plan['outputs'][0]]['seed']
    def native_loss():
        s=native.specs[plan['outputs'][0]];prediction=values[plan['outputs'][0]]
        native._launch(emit_kernel('mse',[s]*2,{'shape':[]}),[prediction,oa_target],native._training[plan['outputs'][0]]['loss'])
        native._launch(emit_kernel('mse_grad',[s]*2,s),[prediction,oa_target],native._training[plan['outputs'][0]]['seed'])
    measured_gradients=last_native_result['gradients']['parameters'] if spec['training'] else {}
    def native_update():
        for n,g in measured_gradients.items():
            native._launch(emit_kernel('sgd',[native.specs[n]]*2,native.specs[n]),[native.weights[n],g],native.weights[n],lr=.01)
    parts={'forward':{'openassembler':lambda:native._run('forward',values),**{k:e.forward for k,e in engines.items()}}}
    if spec['training']:
        parts.update(loss_and_seed={'openassembler':native_loss,**{k:e.loss for k,e in engines.items()}},
                     backward={'openassembler':lambda:native._run('backward',values),**{k:e.backward for k,e in engines.items()}},
                     update={'openassembler':native_update,**{k:e.update for k,e in engines.items()}})
    component={part:paired(fns,repeats=4,samples=10,rounds=3,bound_runtimes={'openassembler':native}) for part,fns in parts.items()}
    graph_one={};stream=torch.cuda.current_stream()
    for kind,fn in functions.items():
        g=torch.cuda.CUDAGraph();before=time.perf_counter()
        with torch.cuda.graph(g,stream=stream):fn()
        graph_one[kind]=g;setup[kind]['one_step_capture_ms']=(time.perf_counter()-before)*1000
    delivery={k:[] for k in functions};graph_host={k:[] for k in functions};dma={k:[] for k in functions}
    cpu_x, cpu_y=torch.randn(shape)*.25,torch.randn(shape)*.25
    for _ in range(15):
        order=list(functions);rng.shuffle(order)
        for kind in order:
            torch.cuda.synchronize();before=time.perf_counter()
            graph_one[kind].replay();torch.cuda.synchronize()
            graph_host[kind].append((time.perf_counter()-before)*1000)
            tensors=oa_inputs if kind=='openassembler' else engines[kind].inputs
            truth_target=oa_target if kind=='openassembler' else engines[kind].target
            before=time.perf_counter()
            with torch.no_grad():
                tensors[input_name].copy_(cpu_x)
                if spec['training']:truth_target.copy_(cpu_y)
            torch.cuda.synchronize();dma[kind].append((time.perf_counter()-before)*1000)
            before=time.perf_counter()
            with torch.no_grad():
                tensors[input_name].copy_(cpu_x)
                if spec['training']:truth_target.copy_(cpu_y)
            graph_one[kind].replay()
            prediction=native._last[plan['outputs'][0]] if kind=='openassembler' else (
                engines[kind].output_value if kind=='nvidia_libraries_eager' else engines[kind].state[engines[kind].output])
            prediction.cpu()
            if spec['training']:
                loss_value=native._training[plan['outputs'][0]]['loss'] if kind=='openassembler' else engines[kind].loss_value
                loss_value.cpu()
            torch.cuda.synchronize();delivery[kind].append((time.perf_counter()-before)*1000)
    for engine in engines.values():
        engine.check_guards()
        assert all(torch.isfinite(v.cpu()).all() for v in engine.weights.values())
    native.check_canaries()
    result={'schema':1,'passed':True,'task':name,'spec':spec,'device':torch.cuda.get_device_name(),
        'torch':torch.__version__,'cuda':torch.version.cuda,'triton':triton.__version__,
        'primary_api':{'tl_dot_signature':str(inspect.signature(tl.dot)),'tl_dot_documentation':tl.dot.__doc__},
        'capture_ms':capture_ms,'setup':setup,'checks':checks,'oa_audit':audits,'profiler_kernels':profiler_parts,
        'overall_gpu_graph':overall,'resident_host_ms':host_samples,'components_gpu_graph':component,
        'resident_graph_host_ms':graph_host,'cpu_delivery_graph_ms':delivery,'h2d_host_ms':dma,
        'plan':plan,'native_ledger':native.ledger,
        'scope':'five bounded synthetic ML tasks; resident input step, 4-invocation CUDA graphs, 3 shuffled rounds; compilation/setup separate; direct Triton tiled kernels, same-PTX ptxas, and a manual PyTorch CUDA operator schedule using installed NVIDIA libraries; not stock autograd, full nvcc model compilation, Inductor or direct cuBLAS algorithm-search'}
    (root/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    for e in engines.values():
        if hasattr(e,'close'):e.close()
    native.close()


def main():
    from exp.ml_suite.tasks import TASKS
    root=Path('results/bench');root.mkdir(parents=True,exist_ok=True)
    records=[]
    selected=os.environ.get('OA_ML_TASK')
    names=[selected] if selected else list(TASKS)
    for task in names:
        if task not in TASKS:raise ValueError('unknown ML benchmark task '+task)
        proc=subprocess.run([sys.executable,'-m','exp.ml_suite.bench_campaign',task],
            text=True,capture_output=True,timeout=400)
        (root/(task+'.log')).write_text(proc.stdout+proc.stderr)
        path=root/task/'result.json'
        records.append(json.loads(path.read_text()) if path.exists() else {'task':task,'passed':False,'error':(proc.stdout+proc.stderr)[-8000:]})
    (root/'result.json').write_text(json.dumps({'passed':all(r['passed'] for r in records),'tasks':records},indent=2))


if __name__=='__main__':
    if len(sys.argv)>1:worker(sys.argv[1])
    else:main()
