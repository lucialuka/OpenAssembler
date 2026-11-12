"""FP32 ordered forward/dX/dW selection on complete output/guard gates."""
import json
from pathlib import Path
import torch
from openassembler import build
from openassembler.gpu.cuda_driver import Driver
from openassembler.modelfront.kernels import emit_kernel
from openassembler.modelfront.profiling import audit_profile
from exp.ml_suite.matrix_candidates import pointer,tiled
from exp.ml_suite.common import launch,paired,guarded,guards


def main():
    torch.manual_seed(519);torch.set_num_threads(1)
    root=Path('results/matrix');root.mkdir(parents=True,exist_ok=True)
    cases={}
    for m,n,k in [(17,16,8),(14,8,8),(14,16,8),(33,13,7),(128,256,128),(128,128,256)]:
        for op,shapes,out in [('linear',[(m,k),(n,k),(n,)],(m,n)),
                              ('linear_dx',[(m,n),(n,k)],(m,k)),
                              ('linear_dw',[(m,n),(m,k)],(n,k))]:
            key=f'{op}-{m}-{n}-{k}'
            base=emit_kernel(op,[{'shape':list(s)} for s in shapes],{'shape':list(out)})
            cases[key]={'baseline':base,'pointer128':pointer(base),'pointer32':pointer(base,32),'tiled':tiled(base)}
    # Multiple logical cases can share a shape-specialized entry.
    kernels={k.name:k for values in cases.values() for k in values.values()}
    source=root/'matrix.ptx';source.write_text('.version 8.7\n.target sm_80\n.address_size 64\n'+
        chr(10).join(k.ptx for k in kernels.values()))
    assert build.compile_paths(source,root/'native',build.Options()).rc==0
    manifest=json.loads((root/'native/manifest.json').read_text())
    entry=manifest.get('sources',manifest.get('entries'))[0]
    torch.empty(1,device='cuda');driver=Driver()
    module=driver.load((root/'native'/entry['cubin']).read_bytes())
    records=[]
    for key,variants in cases.items():
        base=variants['baseline']
        ins=[guarded(driver,s) for s in base.inputs]
        outs={name:guarded(driver,base.output) for name in variants}
        functions={name:driver.function(module,k.name) for name,k in variants.items()}
        runners={name:lambda name=name:launch(driver,functions[name],variants[name].launch,
                    [*[v for _,v in ins],outs[name][1]]) for name in variants}
        checks=[]
        for pattern in ['zero','integer','normal']:
            data=[torch.zeros(s) if pattern=='zero' else torch.randint(-2,3,s).float() if pattern=='integer'
                  else torch.randn(s)*.125 for s in base.inputs]
            for (_,tensor),value in zip(ins,data):tensor.copy_(value)
            torch.cuda.synchronize()
            expected=data[0].double()@data[1].double().T if base.op=='linear' else (
                data[0].double()@data[1].double() if base.op=='linear_dx' else data[0].double().T@data[1].double())
            if len(data)==3:expected+=data[2].double()
            baseline=None
            for name,fn in runners.items():
                driver.memset(outs[name][1].data_ptr(),0xFF,outs[name][1].numel()*4)
                fn();torch.cuda.synchronize()
                got=outs[name][1].cpu()
                assert torch.isclose(got.double(),expected,rtol=1e-4,atol=1e-5).all(),(key,name,'reference')
                if baseline is None:baseline=got
                else:assert torch.equal(got.view(torch.int32),baseline.view(torch.int32)),(key,name,'bitwise')
                guards(driver,[*[b for b,_ in ins],outs[name][0]])
                assert all(torch.equal(v.cpu(),x) for (_,v),x in zip(ins,data))
                checks.append({'method':name,'pattern':pattern,'passed':True,'bitwise_baseline':True,'guards':True,'inputs_immutable':True})
        audits={}
        for name,fn in runners.items():
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                fn();torch.cuda.synchronize()
            audits[name]=audit_profile(prof,[{'entry':variants[name].name}])
        timing=paired(runners,repeats=16)
        records.append({'case':key,'signature':base.signature,'checks':checks,'audits':audits,'rounds':timing,
            'resources':{n:driver.func_attrs(f) for n,f in functions.items()},
            'speedups':{n:[r['medians_ms']['baseline']/r['medians_ms'][n] for r in timing] for n in variants}})
        print(key,records[-1]['speedups'],flush=True)
    torch.cuda.synchronize();driver.unload(module)
    (root/'result.json').write_text(json.dumps({'passed':True,'device':torch.cuda.get_device_name(),
        'torch':torch.__version__,'cuda':torch.version.cuda,'records':records,
        'scope':'ordered FP32 matrix kernels only; full native outputs/tails/guards/bitwise baseline; no library comparison'},indent=2))


if __name__=='__main__':main()
