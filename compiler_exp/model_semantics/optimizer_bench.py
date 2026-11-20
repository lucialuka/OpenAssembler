"""Split/fused matched update microbenchmark, not whole training throughput."""
import statistics,time
import torch
from openassembler.modelfront.kernels import emit_kernel

def benchmark(root,make,check):
    class Point(torch.nn.Module):
        def __init__(self,n):super().__init__();self.weight=torch.nn.Parameter(torch.linspace(-.3,.3,n))
        def forward(self,x):return x*self.weight
    results=[]
    for n in (33,4096,262144):
        policy=dict(name='adamw',beta1=.8,beta2=.95,epsilon=1e-8,weight_decay=.03)
        model=Point(n);plan,rt=make(model,(torch.randn(n),),'optimizer-bench-'+str(n),optimizer=policy)
        spec=plan['parameters'][0];name=spec['name'];rt.forward({'x':torch.randn(n,device='cuda')})
        weight=rt.weights[name];m=torch.linspace(-.02,.02,n).cuda();v=torch.linspace(.001,.01,n).cuda();g=torch.linspace(-.1,.1,n).cuda()
        initial=[x.cpu().clone() for x in (weight,m,v,g)];correction=(.001/(1-.8**3),(1-.95**3)**.5)
        def launch(fused):
            if fused:rt._launch(emit_kernel('adam_fused',[spec]*4,spec,policy),[weight,m,v,g],weight,lr=.001,correction=correction)
            else:
                for key,target in (('m',m),('v',v)):
                    rt._launch(emit_kernel('adam_'+key,[spec]*3,spec,policy),[target,g,weight],target)
                rt._launch(emit_kernel('adam_update',[spec]*4,spec,policy),[weight,m,v,g],weight,lr=.001,correction=correction)
        graphs={}
        original=rt._stream;stream=torch.cuda.Stream();stream.wait_stream(original)
        for fused in (False,True):
            graph=torch.cuda.CUDAGraph();rt._stream=stream;rt._stream_id=stream.cuda_stream
            with torch.cuda.graph(graph,stream=stream):launch(fused)
            graphs[fused]=graph
        rt._stream=original;rt._stream_id=original.cuda_stream
        for fused in (False,True):
            for target,source in zip((weight,m,v,g),initial):target.copy_(source)
            graphs[fused].replay();torch.cuda.synchronize()
            w0,m0,v0,g0=initial;expected_m=m0+(g0-m0)*(1-.8);expected_v=.95*v0+(1-.95)*g0*g0
            expected_w=w0*(1-.001*.03)-correction[0]*expected_m/(expected_v.sqrt()/correction[1]+1e-8)
            for key,got,expected in [('weight',weight,expected_w),('m',m,expected_m),('v',v,expected_v)]:check(got,expected,f'optimizer-bench/{n}/{fused}/{key}')
        samples={False:[],True:[]}
        for repetition in range(15):
            for fused in ((False,True) if repetition%2==0 else (True,False)):
                for target,source in zip((weight,m,v,g),initial):target.copy_(source)
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):graphs[fused].replay()
                end.record();end.synchronize();samples[fused].append(start.elapsed_time(end)*1000/20)
        results.append({'elements':n,'split_us':samples[False],'fused_us':samples[True],
                        'split_median_us':statistics.median(samples[False]),'fused_median_us':statistics.median(samples[True]),'guards':rt.check_canaries(),
                        'scope':'resident fixed-step-3 AdamW coefficient update, graph replay, 20 repetitions/sample; reset before each sample; excludes host, transfers, initialization, full training and step advancement'})
        rt.close()
    return results
