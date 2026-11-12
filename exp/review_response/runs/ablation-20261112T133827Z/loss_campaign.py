"""Isolated native loss timing, exact outputs and full canary proof."""
import ctypes as C
import hashlib
import json
from pathlib import Path
import random
import statistics
import torch
from openassembler import build
from openassembler.modelfront.kernels import emit_kernel
from openassembler.modelfront.runtime import Runtime
from exp.review_response.ordered_loss import ordered_mse
from exp.review_response.audit import audit_profile


def main():
    torch.manual_seed(113)
    rng = random.Random(331)
    root = Path('results/loss')
    root.mkdir(parents=True, exist_ok=True)
    methods = {}
    for size in [1, 33, 112, 136, 513, 16384, 65539]:
        base = emit_kernel('mse', [{'shape':[size]}]*2, {'shape':[]})
        methods[size] = {'serial':base, 'ordered_warp':ordered_mse(base)}
    source = root/'loss.ptx'
    source.write_text('.version 8.7\n.target sm_80\n.address_size 64\n'+chr(10).join(
        k.ptx for ms in methods.values() for k in ms.values()))
    assert build.compile_paths(source, root/'build', build.Options()).rc == 0
    manifest = json.loads((root/'build/manifest.json').read_text())
    entries = manifest.get('sources', manifest.get('entries'))
    binary = (root/'build'/entries[0]['cubin']).read_bytes()
    from openassembler.gpu.cuda_driver import Driver
    torch.empty(1,device='cuda')
    driver=Driver()
    module=driver.load(binary)
    records=[]
    for size, specs in methods.items():
        inp = [torch.empty(size+8,device='cuda') for _ in range(2)]
        out = {n:torch.empty(9,device='cuda') for n in specs}
        for t in [*inp,*out.values()]:
            driver.memset(t.data_ptr(),0xA5,t.numel()*4)
        tensors = [t[4:-4] for t in inp]
        funcs={n:driver.function(module,k.name) for n,k in specs.items()}
        params={n:[C.c_uint64(t.data_ptr()) for t in [*tensors, out[n][4:5]]] for n in specs}
        pointers={n:(C.c_void_p*3)(*[C.cast(C.pointer(a),C.c_void_p) for a in args]) for n,args in params.items()}
        def call(name):
            driver.ck(driver.cu.cuLaunchKernel(funcs[name],1,1,1,specs[name].threads,1,1,0,
                      C.c_void_p(torch.cuda.current_stream().cuda_stream),pointers[name],None),'loss proof')
        checks=[]
        for pattern in ['zero','integer','normal','dynamic']:
            a,b = [torch.zeros(size),torch.zeros(size)] if pattern=='zero' else (
                [torch.randint(-3,4,(size,)).float(),torch.randint(-3,4,(size,)).float()] if pattern=='integer' else
                [torch.randn(size)*(.25 if pattern=='normal' else 10),torch.randn(size)*(.25 if pattern=='normal' else .001)])
            tensors[0].copy_(a);tensors[1].copy_(b)
            torch.cuda.synchronize()
            for n in specs:
                driver.memset(out[n][4:5].data_ptr(),0xFF,4)
                call(n)
            torch.cuda.synchronize()
            actual={n:out[n][4:5].cpu() for n in specs}
            assert torch.equal(actual['serial'].view(torch.int32),actual['ordered_warp'].view(torch.int32)),(size,pattern,actual)
            expected=((a.double()-b.double())**2).mean()
            assert torch.isclose(actual['ordered_warp'].double(),expected,rtol=2e-5,atol=1e-6).all()
            for tensor in [*inp,*out.values()]:
                assert driver.dtoh(tensor.data_ptr(),16)==b'\xa5'*16
                assert driver.dtoh(tensor.data_ptr()+tensor.numel()*4-16,16)==b'\xa5'*16
            assert torch.equal(tensors[0].cpu(),a) and torch.equal(tensors[1].cpu(),b)
            checks.append({'pattern':pattern,'passed':True,'bitwise':True,'canaries':True,'inputs_unchanged':True})
        audits={}
        for n in specs:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                call(n);torch.cuda.synchronize()
            audits[n]=audit_profile(prof,[{'entry':specs[n].name}])
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graphs={}
        for n in specs:
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,stream=stream):
                for _ in range(8):call(n)
            graphs[n]=g
        rounds=[]
        for iteration in range(3):
            samples={n:[] for n in specs};orders=[]
            for _ in range(15):
                order=list(specs);rng.shuffle(order);orders.append(order)
                for n in order:
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();graphs[n].replay();end.record();end.synchronize()
                    samples[n].append(start.elapsed_time(end)/8)
            rounds.append({'round':iteration,'samples_ms':samples,'orders':orders,
                           'speedup':statistics.median(samples['serial'])/statistics.median(samples['ordered_warp'])})
        records.append({'size':size,'checks':checks,'audits':audits,'rounds':rounds,
                        'kernels':{n:{'name':k.name,'launch':k.launch,'ptx_sha256':hashlib.sha256(k.ptx.encode()).hexdigest()} for n,k in specs.items()}})
    torch.cuda.synchronize()
    driver.unload(module)
    (root/'result.json').write_text(json.dumps({'passed':True,'records':records,'device':torch.cuda.get_device_name(),
        'scope':'OA serial versus OA coalesced products with identical ordered FP32 summation; kernel-only CUDA graph event time'},indent=2))


if __name__=='__main__':
    main()
