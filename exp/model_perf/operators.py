"""Full-output pattern/guard/replay gates for exact proposed contracts."""
from pathlib import Path
import json,math
import torch
from openassembler.modelfront.kernels import emit_kernel
from openassembler.modelfront.kernels_tensor import TensorStage,TensorRecipe,build_tensor_kernels
from openassembler.kernels.gemm import GemmPlan,generate
from openassembler.gpu.cuda_driver import Driver
from exp.ml_suite.common import guarded,guards,launch,paired
from exp.model_perf.candidate import bias_reduce,warp_mse


def main():
    torch.manual_seed(1917);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());torch.cuda.set_stream(stream)
    driver=Driver();root=Path('results/operators');root.mkdir(parents=True,exist_ok=True);records=[];timings={}
    cases=[]
    for shape in [(128,128),(128,256)]:
        base=emit_kernel('bias_grad',[{'shape':list(shape)}],{'shape':[shape[1]]})
        cases.append(('bias'+str(shape),shape,[base,bias_reduce(base)]))
    base=emit_kernel('mse',[{'shape':[128,128]}]*2,{'shape':[]})
    cases.append(('mse',(128,128),[base,warp_mse(base,256),warp_mse(base,512),warp_mse(base,1024)]))
    for label,shape,variants in cases:
        stages=[TensorStage(v.name,v.op,('a0','a1') if v.op=='mse' else ('a0',),'output','.version 8.7\n.target sm_80\n.address_size 64\n'+v.ptx,v.launch,v.signature) for v in variants]
        ledger=build_tensor_kernels(root/label,[TensorRecipe('operators','probe',{}, {},tuple(stages))],sass_listings=False)
        arrays=[];back,x=guarded(driver,shape);arrays.append(back);back,y=guarded(driver,shape);arrays.append(back)
        outputs={};modules=[];functions={}
        for v in variants:
            back,o=guarded(driver,[] if v.op=='mse' else [shape[1]]);arrays.append(back);outputs[v.name]=o
            module=driver.load((root/label/ledger[v.name]['module']).read_bytes());modules.append(module);fn=driver.function(module,v.name)
            functions[v.name]=lambda v=v,fn=fn,o=o:launch(driver,fn,v.launch,[x,y,o] if v.op=='mse' else [x,o])
        for pattern in ['normal','zero','constant','cancellation','offset']:
            cx=torch.randn(shape)*.25;cy=torch.randn(shape)*.25
            if pattern=='zero':cx.zero_();cy.zero_()
            elif pattern=='constant':cx.fill_(.125);cy.fill_(-.25)
            elif pattern=='cancellation':cx[1::2]=-cx[::2]+.00001
            elif pattern=='offset':cx+=32;cy+=32
            x.copy_(cx);y.copy_(cy)
            want=((cx-cy).square().mean() if label=='mse' else cx.sum(0))
            for v in variants:
                driver.memset(outputs[v.name].data_ptr(),0xFF,outputs[v.name].numel()*4);functions[v.name]();stream.synchronize()
                actual=outputs[v.name].cpu();torch.testing.assert_close(actual,want,rtol=1e-4,atol=1e-5)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):functions[v.name]()
                driver.memset(outputs[v.name].data_ptr(),0xFF,outputs[v.name].numel()*4);graph.replay();stream.synchronize()
                torch.testing.assert_close(outputs[v.name].cpu(),want,rtol=1e-4,atol=1e-5)
                assert torch.equal(x.cpu(),cx) and torch.equal(y.cpu(),cy);guards(driver,arrays)
                records.append({'operator':label,'pattern':pattern,'name':v.name,'passed':True,'max_abs':float((actual-want).abs().max())})
        timings[label]=paired(functions,repeats=16,samples=15,rounds=3)
        for module in modules:driver.unload(module)
    Path('results/result.json').write_text(json.dumps({'passed':True,'checks':records,'timings':timings,'device':torch.cuda.get_device_name()},indent=2))
if __name__=='__main__':main()
