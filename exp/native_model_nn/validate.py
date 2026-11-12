"""Pinned A100 same-PTX neural kernel correctness; no performance promotion."""
from __future__ import annotations
from pathlib import Path
import json
import modal

app=modal.App("openassembler-native-model-nn")
image=modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04",add_python="3.12").pip_install(
    "torch==2.7.1",index_url="https://download.pytorch.org/whl/cu128")


@app.function(image=image,gpu="A100-40GB",timeout=900,max_containers=1,retries=0)
def validate(compiler):
    import ctypes as C
    import hashlib
    import os
    import sys
    import tempfile
    import torch
    with tempfile.TemporaryDirectory(prefix="oa-nn-") as directory:
        root=Path(directory)
        for name,raw in compiler.items():
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        (root/"env/cuda.sh").chmod(0o700);sys.path.insert(0,directory);os.environ["PYTHONPATH"]=directory
        from openassembler.modelfront.kernels import emit_kernel
        from openassembler.build import compile_paths,Options
        from openassembler.gpu.cuda_driver import Driver
        cases=[];kernels={}
        for rows,width in [(5,1),(5,3),(5,33),(5,513),(129,1024)]:
            for op in ["softmax","softmax_backward"]:
                spec={"shape":[rows,width],"dtype":"f32"}
                kernel=emit_kernel(op,[spec]*(2 if op.endswith("backward") else 1),spec)
                kernels[kernel.name]=kernel;cases.append((kernel,rows,width))
        for count in [131,65539]:
            for op in ["gelu","gelu_backward"]:
                spec={"shape":[count],"dtype":"f32"}
                kernel=emit_kernel(op,[spec]*(2 if op.endswith("backward") else 1),spec)
                kernels[kernel.name]=kernel;cases.append((kernel,1,count))
        for rows,width in [(5,1),(5,3),(5,33),(5,513),(129,1024)]:
            spec={"shape":[rows,width],"dtype":"f32"};channel={"shape":[width],"dtype":"f32"}
            for op in ["layernorm","layernorm_backward","channel_affine","channel_mul","channel_grad","channel_sum"]:
                inputs=([spec] if op in ["layernorm","channel_sum"] else [spec,spec] if op in ["layernorm_backward","channel_grad"]
                        else [spec,channel,channel] if op=="channel_affine" else [spec,channel])
                kernel=emit_kernel(op,inputs,channel if op in ["channel_grad","channel_sum"] else spec,
                                   {"epsilon":1e-5} if op.startswith("layernorm") else {})
                kernels[kernel.name]=kernel;cases.append((kernel,rows,width))
        path=root/"nn.ptx";path.write_text(".version 8.7\n.target sm_80\n.address_size 64\n"+"\n".join(k.ptx for k in kernels.values()))
        out=root/"build";result=compile_paths(path,out,Options(fallback=None))
        assert result.rc==0
        manifest=json.loads((out/"manifest.json").read_text());entry=manifest["entries"][0]
        assert all(k["compiled_by"]=="openassembler" for k in entry["kernels"])
        anchor=torch.empty(1,device="cuda");driver=Driver()
        modules={method:driver.load((out/entry[key]).read_bytes()) for method,key in [("oa","cubin"),("ptxas","reference")]}
        checks=[];generator=torch.Generator().manual_seed(123)
        for kernel,rows,width in cases:
            shape=kernel.output;input_shape=kernel.inputs[0]
            for pattern in ["zero","normal","saturation"]:
                x=torch.zeros(input_shape) if pattern=="zero" else torch.randn(input_shape,generator=generator)*(12 if pattern=="saturation" else 1)
                dy=torch.randn(input_shape,generator=generator)
                if kernel.op=="softmax":
                    expected=torch.softmax(x.double(),dim=-1).float();inputs=[x]
                elif kernel.op=="softmax_backward":
                    y=torch.softmax(x.double(),dim=-1).float()
                    expected=y.double()*(dy.double()-(y.double()*dy.double()).sum(dim=-1,keepdim=True));inputs=[dy,y]
                elif kernel.op=="gelu":
                    expected=torch.nn.functional.gelu(x.double(),approximate="tanh").float();inputs=[x]
                elif kernel.op=="gelu_backward":
                    xx=x.double().requires_grad_();torch.nn.functional.gelu(xx,approximate="tanh").backward(dy.double())
                    expected=xx.grad.float();inputs=[dy,x]
                elif kernel.op.startswith("layernorm"):
                    if pattern=="saturation":x=x*.01+1000
                    if kernel.op=="layernorm":
                        expected=torch.nn.functional.layer_norm(x.double(),(width,),eps=1e-5).float();inputs=[x]
                    else:
                        xx=x.double().requires_grad_();torch.nn.functional.layer_norm(xx,(width,),eps=1e-5).backward(dy.double())
                        expected=xx.grad.float();inputs=[dy,x]
                elif kernel.op=="channel_sum":
                    expected=x.double().sum(dim=0);inputs=[x]
                elif kernel.op=="channel_grad":
                    expected=(dy.double()*x.double()).sum(dim=0);inputs=[dy,x]
                else:
                    weight=torch.randn(width,generator=generator);bias=torch.randn(width,generator=generator)
                    expected=x.double()*weight.double();inputs=[x,weight]
                    if kernel.op=="channel_affine":expected=expected+bias.double();inputs.append(bias)
                gpu_inputs=[v.cuda() for v in inputs]
                storage=torch.full((expected.numel()+32,),-12345.,device="cuda");output=storage[16:-16]
                for repeat in range(3):
                    for method,module in modules.items():
                        output.fill_(float("nan"))
                        function=driver.function(module,kernel.name)
                        driver.launch(function,kernel.grid,(kernel.threads,1,1),[C.c_uint64(t.data_ptr()) for t in [*gpu_inputs,output]])
                        actual=output.cpu().reshape(shape)
                        close=torch.isclose(actual.double(),expected.double(),rtol=1e-4,atol=1e-5)
                        passed=bool(close.all()) and bool((storage[:16].cpu()==-12345.).all()) and bool((storage[-16:].cpu()==-12345.).all())
                        assert all(torch.equal(t.cpu(),v) for t,v in zip(gpu_inputs,inputs))
                        checks.append({"op":kernel.op,"shape":list(shape),"pattern":pattern,"repeat":repeat,"method":method,
                                       "passed":passed,"mismatches":int((~close).sum()),"max_abs":float((actual.double()-expected.double()).abs().max())})
                        assert passed,checks[-1]
        for module in modules.values():driver.unload(module)
        return {"passed":True,"checks":checks,"device":torch.cuda.get_device_name(),"torch":torch.__version__,
                "source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "compiler_source_sha256":{n:hashlib.sha256(b).hexdigest() for n,b in compiler.items() if n.endswith(".py")},
                "scope":"OA and same-PTX ptxas primitive checks; 1e-4/1e-5 full outputs, poison, canaries, immutable inputs; no speed claim"}


@app.local_entrypoint()
def main():
    from datetime import datetime,timezone
    from openassembler.transport import runtime_files
    result=validate.remote(runtime_files(native=True))
    path=Path(__file__).parent/("a100-nn-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")+".json")
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(path),"passed":True,"checks":len(result["checks"])}))
