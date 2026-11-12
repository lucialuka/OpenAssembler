"""Numerical, guard and complete-step A100 trial for a parallel MSE candidate."""
import json
from pathlib import Path
from statistics import median
from unittest.mock import patch
import torch
from openassembler.modelfront.graph import capture_model,extract_weights
from openassembler.modelfront.optimization import optimize_plan
from openassembler.modelfront.runtime import Runtime,build_bundle
from openassembler.modelfront.reference import CPUReference
from openassembler.modelfront import kernels,runtime
from openassembler.modelfront.replay import TrainingReplay
from exp.ml_suite.loss_candidate import parallel_mse
from exp.ml_suite.tasks import create


def main():
    torch.set_num_threads(1);torch.manual_seed(788)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());torch.cuda.set_stream(stream)
    records=[];original=kernels.emit_kernel
    for case,shape,mode in [('dense_regression_fp32',(128,128),'fp32'),
                            ('dense_regression_fp16',(128,128),'f16_rne_f32_accum')]:
        model=create(case).train();plan=optimize_plan(capture_model(model,[torch.randn(shape)*.25],math_mode=mode),reuse_storage=True)
        weights=extract_weights(model,plan);folder=Path('results/loss')/case;enabled=[True]
        def dispatch(op,inputs,out,attrs=None):
            base=original(op,inputs,out,attrs)
            return parallel_mse(base) if enabled[0] and op=='mse' and __import__('math').prod(base.inputs[0])==16384 else base
        patch_stack=__import__('contextlib').ExitStack()
        patch_stack.enter_context(patch.object(kernels,'emit_kernel',dispatch));patch_stack.enter_context(patch.object(runtime,'emit_kernel',dispatch))
        try:
            build_bundle(plan,weights,folder/'candidate',sass_listings=False)
            enabled[0]=False;build_bundle(plan,weights,folder/'baseline',sass_listings=False)
            plans={n:json.loads((folder/n/'plan.json').read_text()) for n in ('baseline','candidate')}
            runs={'baseline':Runtime.load(folder/'baseline')}
            enabled[0]=True;runs['candidate']=Runtime.load(folder/'candidate')
        except Exception:
            patch_stack.close();raise
        name=plan['inputs'][0]['name'];x=torch.empty(shape,device='cuda');target=torch.empty(shape,device='cuda')
        ref=CPUReference(plans['baseline'],weights);checks=[]
        for i in range(4):
            cpu_x=torch.randn(shape)*.25;cpu_y=torch.randn(shape)*.25
            x.copy_(cpu_x);target.copy_(cpu_y)
            expected=ref.train_step({name:cpu_x},cpu_y,.01)
            vals={}
            for method,run in runs.items():
                enabled[0]=(method=='candidate')
                if i:run.poison_workspace()
                result=run.train_step({name:x},target,.01);run.synchronize()
                values={'prediction':result['prediction'],'loss':result['loss'].reshape(()),
                    **{'dx/'+n:v for n,v in result['gradients']['inputs'].items()},
                    **{'dw/'+n:v for n,v in result['gradients']['parameters'].items()},
                    **{'w/'+n:v for n,v in run.weights.items()}}
                truth={'prediction':expected['prediction'],'loss':expected['loss'].reshape(()),
                    **{'dx/'+n:v for n,v in expected['gradients']['inputs'].items()},
                    **{'dw/'+n:v for n,v in expected['gradients']['parameters'].items()},
                    **{'w/'+n:v for n,v in ref.weights.items()}}
                vals[method]={}
                for key,value in values.items():
                    got=value.detach().cpu();want=truth[key].detach().cpu()
                    tol=.002 if mode!='fp32' else 1e-4
                    good=torch.isclose(got,want,rtol=tol,atol=2e-5 if mode!='fp32' else 1e-5)
                    assert bool(good.all()),(case,i,method,key,float((got-want).abs().max()))
                    vals[method][key]=got;checks.append({'iteration':i,'method':method,'field':key,'passed':True})
                run.check_canaries()
            for key in vals['baseline']:
                if key!='loss':assert torch.equal(vals['baseline'][key].view(torch.int32),vals['candidate'][key].view(torch.int32)),(case,i,key)
            assert torch.equal(x.cpu(),cpu_x) and torch.equal(target.cpu(),cpu_y)
        # New scalar reduction is checked against PyTorch's mathematical mean.
        reduction_probes=[]
        from exp.ml_suite.common import launch
        loss_kernel=next(k for k in runs['candidate'].kernels.values() if k.op=='mse')
        result_buf=runs['candidate']._training[plan['outputs'][0]]['loss']
        for cpu_x,cpu_y in [(torch.zeros(shape),torch.ones(shape)),
                            (torch.arange(16384).reshape(shape).float()/100,torch.zeros(shape)),
                            (torch.randn(shape),torch.randn(shape))]:
            x.copy_(cpu_x);target.copy_(cpu_y);prediction=runs['candidate'].forward({name:x})[plan['outputs'][0]]
            runs['candidate']._launch(loss_kernel,[prediction,target],result_buf)
            runs['candidate'].synchronize()
            expected_loss=torch.nn.functional.mse_loss(prediction.cpu(),cpu_y)
            actual_loss=result_buf.cpu().reshape(())
            reduction_probes.append({'got':float(actual_loss),'expected':float(expected_loss)})
            assert torch.isclose(actual_loss,expected_loss,rtol=1e-6,atol=1e-7),(case,reduction_probes[-1])
        replay={}
        inp={name:torch.randn(shape,device='cuda')*.25};tar=torch.randn(shape,device='cuda')*.25
        for n,run in runs.items():
            enabled[0]=(n=='candidate');replay[n]=TrainingReplay(plans[n],run.weights,folder/n,inp,tar)
        for r in replay.values():r.run()
        pairs=[];rng=torch.Generator().manual_seed(991)
        order=['baseline','candidate']
        for round_index in range(3):
            samples={n:[] for n in order}
            for sample_index in range(20):
                order=['candidate','baseline'] if (sample_index+round_index)%2 else ['baseline','candidate']
                for n in order:
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(8):replay[n].run()
                    end.record();end.synchronize();samples[n].append(start.elapsed_time(end)/8)
            pairs.append({'round':round_index,'samples_ms':samples,'medians_ms':{n:median(v) for n,v in samples.items()}})
        for r in replay.values():r.close()
        for r in runs.values():r.close()
        patch_stack.close()
        records.append({'case':case,'device':torch.cuda.get_device_name(),'checks':checks,'rounds':pairs,
            'speedups':[r['medians_ms']['baseline']/r['medians_ms']['candidate'] for r in pairs],
            'candidate_kernel':next(k.ptx for k in plans.values() if False) if False else 'parallel local sum plus shared tree',
            'candidate_vs_torch_mse':True,'reduction_probes':reduction_probes})
        print(case,records[-1]['speedups'],flush=True)
    Path('results/loss/result.json').write_text(json.dumps({'passed':True,'records':records,
        'scope':'large-output training; loss reduction changed within FP32 tolerance; all other state bitwise-identical; complete replay'},indent=2))


if __name__=='__main__':main()
