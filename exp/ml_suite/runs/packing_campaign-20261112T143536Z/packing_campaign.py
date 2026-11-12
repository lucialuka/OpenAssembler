"""Complete native training and inference pack reuse with mutation controls."""
import json
from pathlib import Path
import torch
from openassembler.modelfront.graph import capture_model,extract_weights
from openassembler.modelfront.optimization import optimize_plan
from openassembler.modelfront.runtime import Runtime,build_bundle
from openassembler.modelfront.reference import CPUReference
from openassembler.modelfront.profiling import audit_profile
from exp.ml_suite.packing_candidate import PackedRuntime
from exp.ml_suite.common import paired


class Projections(torch.nn.Module):
    def __init__(self):
        super().__init__();self.q=torch.nn.Linear(128,128);self.k=torch.nn.Linear(128,128);self.v=torch.nn.Linear(128,128)
    def forward(self,x):return self.q(x)+self.k(x)+self.v(x)


def main():
    import runpy
    torch.manual_seed(241);torch.set_num_threads(1)
    root=Path('results/packing');root.mkdir(parents=True,exist_ok=True)
    records=[]
    for case,model in [('mlp',runpy.run_path('examples/native_tensor_model/model.py')['create_model']()),('projections',Projections())]:
        sample=torch.randn(128,128)*.25
        plan=optimize_plan(capture_model(model,[sample],math_mode='f16_rne_f32_accum'),reuse_storage=True)
        weights=extract_weights(model,plan);folder=root/case
        build_bundle(plan,weights,folder,sass_listings=False)
        runs={name:cls.load(folder) for name,cls in [('baseline',Runtime),('reuse',PackedRuntime)]}
        reference=CPUReference(plan,weights)
        name=plan['inputs'][0]['name']
        checks=[];audits={}
        for iteration in range(4):
            x,y=torch.randn(128,128)*.25,torch.randn(128,128)*.25
            inputs={name:x.cuda()};target=y.cuda()
            truth=reference.train_step({name:x},y,.01);baseline=None
            expected={'prediction':truth['outputs'][plan['outputs'][0]],'loss':truth['loss'].reshape(()),
                **{'dx/'+n:v for n,v in truth['gradients']['inputs'].items()},
                **{'dw/'+n:v for n,v in truth['gradients']['parameters'].items()},
                **{'w/'+n:v for n,v in reference.weights.items()}}
            for method,runtime in runs.items():
                if iteration:runtime.poison_workspace()
                actual=runtime.train_step(inputs,target,.01);runtime.synchronize()
                fields={'prediction':actual['prediction'],'loss':actual['loss'].reshape(()),
                    **{'dx/'+n:v for n,v in actual['gradients']['inputs'].items()},
                    **{'dw/'+n:v for n,v in actual['gradients']['parameters'].items()},
                    **{'w/'+n:v for n,v in runtime.weights.items()}}
                cpu={n:v.cpu() for n,v in fields.items()}
                for n,v in cpu.items():
                    assert torch.isclose(v,expected[n],rtol=.002,atol=2e-5).all(),(case,method,n)
                    if baseline is not None:assert torch.equal(v.view(torch.int32),baseline[n].view(torch.int32)),(case,n)
                if baseline is None:baseline=cpu
                runtime.check_canaries()
                assert torch.equal(inputs[name].cpu(),x) and torch.equal(target.cpu(),y)
                checks.append({'iteration':iteration,'method':method,'fields':len(fields),'passed':True,'bitwise':True})
        training=paired({n:lambda r=r:r.train_step(inputs,target,.01) for n,r in runs.items()},repeats=4,bound_runtimes=runs)
        for n,r in runs.items():
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                r.train_step(inputs,target,.01);r.synchronize()
            audits['training/'+n]=audit_profile(prof,r.launch_ledger)
        # New runtimes isolate inference state and exercise updated/replaced weights.
        for r in runs.values():r.close()
        runs={n:cls.load(folder) for n,cls in [('baseline',Runtime),('reuse',PackedRuntime)]}
        inference_launches=[]
        for iteration in range(4):
            x=torch.randn(128,128)*.25;inputs={name:x.cuda()}
            if iteration==2:
                for n in weights:
                    new=(weights[n]+.03125).contiguous()
                    for r in runs.values():r.weights[n].copy_(new)
                reference=CPUReference(plan,{n:v+.03125 for n,v in weights.items()})
            elif iteration==3:
                for n in weights:
                    new=(weights[n]-.0625).contiguous()
                    for r in runs.values():r.weights[n]=new.cuda()
                reference=CPUReference(plan,{n:v-.0625 for n,v in weights.items()})
            else:
                reference=CPUReference(plan,weights)
            expected=reference.forward({name:x})[plan['outputs'][0]];baseline=None
            for method,r in runs.items():
                actual=r.forward(inputs)[plan['outputs'][0]];r.synchronize();cpu=actual.cpu()
                assert torch.isclose(cpu,expected,rtol=.002,atol=2e-5).all(),(case,method,'inference')
                if baseline is None:baseline=cpu
                else:assert torch.equal(cpu.view(torch.int32),baseline.view(torch.int32))
                r.check_canaries()
                inference_launches.append({'iteration':iteration,'method':method,'launch_count':len(r.launch_ledger)})
        # Capture conservatively refreshes parameters: common-operand reuse only.
        inference=paired({n:lambda r=r:r.forward(inputs) for n,r in runs.items()},repeats=8,bound_runtimes=runs)
        records.append({'case':case,'checks':checks,'audits':audits,'training':training,'inference_graph':inference,
            'inference_launches':inference_launches,'allocations':{n:len(r._guards) for n,r in runs.items()},
            'training_speedups':[v['medians_ms']['baseline']/v['medians_ms']['reuse'] for v in training],
            'inference_graph_speedups':[v['medians_ms']['baseline']/v['medians_ms']['reuse'] for v in inference]})
        for r in runs.values():r.close()
    (root/'result.json').write_text(json.dumps({'passed':True,'device':torch.cuda.get_device_name(),'records':records,
        'scope':'complete half-policy OA graphs; duplicate input packs plus versioned inference weights; captured graphs refresh parameters'},indent=2))


if __name__=='__main__':main()
