"""Complete FP32 native model updates with baseline versus exact matrix candidates."""
import hashlib
import json
import random
import statistics
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import torch
from openassembler.modelfront.graph import capture_model,extract_weights
from openassembler.modelfront.optimization import optimize_plan
from openassembler.modelfront.runtime import Runtime,build_bundle
from openassembler.modelfront.reference import CPUReference
from openassembler.modelfront import kernels,runtime
from exp.ml_suite.matrix_candidates import pointer,tiled
from exp.ml_suite.common import paired
from exp.ml_suite.tasks import create
from exp.ml_suite.matrices import selected_kernel


def main():
    torch.manual_seed(611);torch.set_num_threads(1)
    benchmark_stream=torch.cuda.Stream();benchmark_stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(benchmark_stream)
    cases=[('tiny_regression','native_model',(17,8)),
           ('dense_regression_fp32','native_tensor_model',(128,128)),
           ('attention_training','native_attention_model',(2,7,8))]
    records=[]
    original=kernels.emit_kernel
    for case,factory,shape in cases:
        model=create(case).train()
        plan=optimize_plan(capture_model(model,[torch.randn(shape)*.25],math_mode='fp32'),reuse_storage=True)
        weights=extract_weights(model,plan);folder=Path('results/matrix_graph')/case;folder.mkdir(parents=True,exist_ok=True)
        base=root=folder/'baseline'
        build_bundle(plan,weights,base,sass_listings=False)
        enabled=[False]
        def dispatch(op,inputs,out,attrs=None):
            candidate=original(op,inputs,out,attrs)
            return selected_kernel(candidate,pointer,tiled) if enabled[0] else candidate
        patch_stack=ExitStack()
        patch_stack.enter_context(patch.object(kernels,'emit_kernel',dispatch))
        patch_stack.enter_context(patch.object(runtime,'emit_kernel',dispatch))
        try:
            enabled[0]=True
            build_bundle(plan,weights,folder/'candidate',sass_listings=False)
            plans={'baseline':json.loads((base/'plan.json').read_text()),
                   'candidate':json.loads((folder/'candidate/plan.json').read_text())}
            enabled[0]=False;runs={'baseline':Runtime.load(base)}
            enabled[0]=True;runs['candidate']=Runtime.load(folder/'candidate')
        except Exception:
            patch_stack.close()
            raise
        references={n:CPUReference(plans[n],weights) for n in plans}
        name=plan['inputs'][0]['name'];checks=[];inputs={n:torch.empty(shape,device='cuda') for n in [name]}
        target=torch.empty(shape,device='cuda')
        for i in range(4):
            x=torch.randn(shape)*.25;y=torch.randn(shape)*.25
            inputs[name].copy_(x);target.copy_(y)
            expected=references['baseline'].train_step({name:x},y,.01)
            expected_fields={'prediction':expected['outputs'][plan['outputs'][0]],'loss':expected['loss'].reshape(()),
                **{'dx/'+n:v for n,v in expected['gradients']['inputs'].items()},
                **{'dw/'+n:v for n,v in expected['gradients']['parameters'].items()},
                **{'w/'+n:v for n,v in references['baseline'].weights.items()}}
            paired_values={}
            for method,run in runs.items():
                if i:run.poison_workspace()
                result=run.train_step(inputs,target,.01);run.synchronize()
                values={'prediction':result['prediction'],'loss':result['loss'].reshape(()),
                    **{'dx/'+n:v for n,v in result['gradients']['inputs'].items()},
                    **{'dw/'+n:v for n,v in result['gradients']['parameters'].items()},
                    **{'w/'+n:v for n,v in run.weights.items()}}
                paired_values[method]={}
                for field,value in values.items():
                    got=value.cpu();ref=expected_fields[field].cpu()
                    good=torch.isclose(got,ref,rtol=1e-4,atol=1e-5)
                    assert good.all(),(case,i,method,field)
                    paired_values[method][field]=got
                    checks.append({'iteration':i,'method':method,'field':field,'passed':True})
                run.check_canaries()
            for field,baseline_value in paired_values['baseline'].items():
                assert torch.equal(baseline_value.view(torch.int32),paired_values['candidate'][field].view(torch.int32)),(case,i,field)
            assert torch.equal(inputs[name].cpu(),x) and torch.equal(target.cpu(),y)
        # Bind fresh stable storage for graph replay timing after correctness gates.
        inp={name:torch.randn(shape,device='cuda')*.25};tar=torch.randn(shape,device='cuda')*.25
        timings={}
        for method in runs:
            from openassembler.modelfront.replay import TrainingReplay
            enabled[0]=(method=='candidate')
            replay=TrainingReplay(plans[method],runs[method].weights,folder/method,inp,tar)
            timings[method]=replay
        # TrainingReplay owns a CUDA graph already; CUDA forbids capturing its
        # replay operation inside another graph, so pair event timings directly.
        names=list(timings);rng=random.Random(4881);graph=[]
        for method in timings:timings[method].run()
        for round_index in range(3):
            samples={n:[] for n in names};orders=[]
            for _ in range(20):
                order=names[:];rng.shuffle(order);orders.append(order)
                for n in order:
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(8):timings[n].run()
                    end.record();end.synchronize();samples[n].append(start.elapsed_time(end)/8)
            graph.append({'round':round_index,'orders':orders,'samples_ms':samples,
                          'medians_ms':{n:statistics.median(v) for n,v in samples.items()}})
        audit={}
        from openassembler.modelfront.profiling import audit_profile
        for name_,rep in timings.items():
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                rep.run();rep.runtime.synchronize()
            audit[name_]=audit_profile(prof,rep.launch_ledger)
            rep.close()
        base_entries={(s['phase'],s['output']):s['entry'] for s in runs['baseline'].steps if not s.get('alias')}
        selected=[{'entry':s['entry'],'op':s['kernel'],'inputs':s['inputs'],'output':s['output']}
                  for s in runs['candidate'].steps
                  if not s.get('alias') and s['entry']!=base_entries.get((s['phase'],s['output']))]
        records.append({'case':case,'checks':checks,'paired_bitwise':True,'selected_matrix_steps':selected,
            'rounds':graph,'speedups':[r['medians_ms']['baseline']/r['medians_ms']['candidate'] for r in graph],
            'audits':audit,'device':torch.cuda.get_device_name()})
        for run in runs.values():run.close()
        patch_stack.close()
        print(case,records[-1]['speedups'],flush=True)
    Path('results/matrix_graph/result.json').write_text(json.dumps({'passed':True,'records':records,
        'scope':'complete FP32 OA native training step; exact matched baseline/candidate bits, CPUReference full state, guarded graph; graph replay resident inputs'},indent=2))


if __name__=='__main__':main()
