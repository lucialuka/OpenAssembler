"""Full matrix output/immutability/guard/replay checks for four exact tiles."""
from dataclasses import asdict
from pathlib import Path
import json
import torch
from openassembler.kernels.gemm import GemmPlan,generate
from openassembler.modelfront.kernels_tensor import TensorStage,TensorRecipe,build_tensor_kernels
from openassembler.gpu.cuda_driver import Driver
from exp.ml_suite.common import guarded,guards,launch


def main():
    torch.manual_seed(1918);torch.set_num_threads(1);stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());torch.cuda.set_stream(stream)
    d=Driver();records=[];inventory={}
    for m,n,k in [(128,128,128),(128,256,128),(128,128,256),(256,128,128)]:
        root=Path('results/gemm')/f'{m}-{n}-{k}';stages=[]
        for tile in ['128x128','64x64']:
            p=GemmPlan(m,n,k,'f16',tile=tile);name='oa_probe_'+tile.replace('x','_')
            stages.append(TensorStage(name,'probe',('a','b'),'output',generate(p).replace('.entry gemm(',f'.entry {name}('),p.launch,{'plan':asdict(p)},p))
        ledger=build_tensor_kernels(root,[TensorRecipe('probe','probe',{}, {},tuple(stages))],sass_listings=False)
        inventory[str((m,n,k))]=ledger
        ba,a=guarded(d,(m,k),torch.float16);bb,b=guarded(d,(n,k),torch.float16);bc,c=guarded(d,(m,n));back=[ba,bb,bc]
        for stage in stages:
            module=d.load((root/ledger[stage.name]['module']).read_bytes());fn=d.function(module,stage.name)
            for pattern in ['zero','integer','normal','alternating']:
                ca=torch.randn(m,k)*.25;cb=torch.randn(n,k)*.25
                if pattern=='zero':ca.zero_();cb.zero_()
                elif pattern=='integer':ca=torch.randint(-3,4,(m,k)).float();cb=torch.randint(-3,4,(n,k)).float()
                elif pattern=='alternating':ca[:,1::2]=-ca[:,::2];cb.fill_(.125)
                ca=ca.half();cb=cb.half();a.copy_(ca);b.copy_(cb);want=ca.float()@cb.float().T
                if stage.launch['smem']>49152:d.ck(d.cu.cuFuncSetAttribute(fn,8,stage.launch['smem']),'shared optin')
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):launch(d,fn,stage.launch,[a,b,c])
                for replay in range(2):
                    d.memset(c.data_ptr(),0xFF,c.numel()*4);graph.replay();stream.synchronize()
                    torch.testing.assert_close(c.cpu(),want,rtol=0 if pattern in ('zero','integer') else .002,atol=0 if pattern in ('zero','integer') else 2e-5)
                    assert torch.equal(a.cpu().view(torch.int16),ca.view(torch.int16)) and torch.equal(b.cpu().view(torch.int16),cb.view(torch.int16));guards(d,back)
                records.append({'shape':[m,n,k],'tile':stage.gemm_plan.tile,'pattern':pattern,'passed':True,'max_abs':float((c.cpu()-want).abs().max())})
            d.unload(module)
    Path('results/result.json').write_text(json.dumps({'passed':True,'checks':records,'inventory':inventory,'device':torch.cuda.get_device_name()},indent=2))
if __name__=='__main__':main()
