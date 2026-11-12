"""Shared benchmark statistics and explicit native Driver launch helpers."""
import ctypes as C
import random
import statistics
import torch


def launch(driver, fn, contract, tensors):
    args=[C.c_uint64(t.data_ptr()) for t in tensors]
    params=(C.c_void_p*len(args))(*[C.cast(C.pointer(x),C.c_void_p) for x in args])
    c=contract
    driver.ck(driver.cu.cuLaunchKernel(fn,*c['grid'],c['threads'],1,1,c.get('smem',0),
        C.c_void_p(torch.cuda.current_stream().cuda_stream),params,None),'ML experiment launch')


def paired(functions, *, repeats=16, samples=15, rounds=3, bound_runtimes=None):
    # Capture every arm on the current (default) stream. PyTorch autograd uses
    # CUDA stream recording internally; capturing a training step on a separate
    # blocking stream can make its backward graph depend on the legacy stream.
    stream=torch.cuda.current_stream()
    graphs={}
    for name,fn in functions.items():
        runtime=(bound_runtimes or {}).get(name)
        previous=runtime._stream if runtime is not None else None
        try:
            if runtime is not None:
                runtime._stream=stream;runtime._stream_id=stream.cuda_stream
            for _ in range(3):fn()
            stream.synchronize()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,stream=stream):
                for _ in range(repeats):fn()
            graphs[name]=g
        finally:
            if runtime is not None:
                runtime._stream=previous;runtime._stream_id=previous.cuda_stream
    output=[];rng=random.Random(49201)
    for r in range(rounds):
        values={n:[] for n in functions};orders=[]
        for _ in range(samples):
            order=list(functions);rng.shuffle(order);orders.append(order)
            for n in order:
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(stream):
                    a.record();graphs[n].replay();b.record()
                b.synchronize()
                values[n].append(a.elapsed_time(b)/repeats)
        output.append({'round':r,'orders':orders,'samples_ms':values,
                       'medians_ms':{n:statistics.median(v) for n,v in values.items()}})
    return output


def guarded(driver, shape, dtype=torch.float32):
    import math
    padding=16//torch.empty([],dtype=dtype).element_size()
    backing=torch.empty(math.prod(shape)+padding*2,device='cuda',dtype=dtype)
    driver.memset(backing.data_ptr(),0xA5,backing.numel()*backing.element_size())
    return backing,backing[padding:-padding].view(shape)


def guards(driver, backings):
    for index,t in enumerate(backings):
        head=driver.dtoh(t.data_ptr(),16)
        tail=driver.dtoh(t.data_ptr()+t.numel()*t.element_size()-16,16)
        if head!=bytes([0xA5])*16 or tail!=bytes([0xA5])*16:
            raise AssertionError(f'guard allocation {index}, shape={tuple(t.shape)}, dtype={t.dtype}, head={head.hex()}, tail={tail.hex()}')
