"""Research comparators. Vendor modules never enter the production OA runtime."""
import ctypes as C
import glob
import json
import math
from pathlib import Path
import torch
from openassembler.gpu.cuda_driver import Driver
from openassembler.modelfront.kernels import prepare,emit_kernel,tensor_recipes
from exp.ml_suite.common import guarded,guards,launch


def bias_kernel(shape):
    from types import SimpleNamespace
    m,n=shape
    name=f'nvidia_bias_{m}_{n}'
    ptx=f""".version 8.7
.target sm_80
.address_size 64
.visible .entry {name}(.param .u64 a,.param .u64 b,.param .u64 o) {{
 .reg .b32 %r<8>; .reg .b64 %d<10>; .reg .f32 %f<3>; .reg .pred %p;
 ld.param.u64 %d0,[a]; ld.param.u64 %d1,[b]; ld.param.u64 %d2,[o];
 mov.u32 %r0,%tid.x; mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ctaid.y;
 mad.lo.u32 %r3,%r1,128,%r0; setp.ge.u32 %p,%r3,{n}; @%p bra done;
 mad.lo.u32 %r4,%r2,{n},%r3; mul.wide.u32 %d3,%r4,4;
 add.u64 %d4,%d0,%d3; add.u64 %d5,%d2,%d3;
 mul.wide.u32 %d6,%r3,4; add.u64 %d7,%d1,%d6;
 ld.global.f32 %f0,[%d4]; ld.global.f32 %f1,[%d7];
 add.rn.f32 %f2,%f0,%f1; st.global.f32 [%d5],%f2; done: ret; }}
"""
    return SimpleNamespace(name=name,ptx=ptx,launch={'grid':[(n+127)//128,m,1],'threads':128,'smem':0})


class Blas:
    def __init__(self):
        paths=glob.glob('/usr/local/lib/python3*/site-packages/nvidia/cublas/lib/libcublas.so*')
        self.lib=C.CDLL(paths[0] if paths else 'libcublas.so.12')
        self.handle=C.c_void_p()
        self.lib.cublasCreate_v2.argtypes=[C.POINTER(C.c_void_p)]
        self.lib.cublasDestroy_v2.argtypes=[C.c_void_p]
        self.lib.cublasSetStream_v2.argtypes=[C.c_void_p,C.c_void_p]
        self.lib.cublasSetMathMode.argtypes=[C.c_void_p,C.c_int]
        self.ck(self.lib.cublasCreate_v2(C.byref(self.handle)))
        self.ck(self.lib.cublasSetMathMode(self.handle,0))
        self.gemm=self.lib.cublasGemmEx
        self.gemm.argtypes=[C.c_void_p]+[C.c_int]*5+[C.c_void_p,C.c_void_p,C.c_int,C.c_int,
            C.c_void_p,C.c_int,C.c_int,C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int]
        self.batched=self.lib.cublasGemmStridedBatchedEx
        self.batched.argtypes=[C.c_void_p]+[C.c_int]*5+[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_longlong,
            C.c_void_p,C.c_int,C.c_int,C.c_longlong,C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_longlong,
            C.c_int,C.c_int,C.c_int]
        self.alpha,self.beta=C.c_float(1),C.c_float(0)

    def ck(self,status):
        if status:raise RuntimeError('cuBLAS status '+str(status))

    def product(self,a,b,out,ta=False,tb=False):
        m,n=out.shape[-2:];k=a.shape[-1]
        dtype=2 if a.dtype==torch.float16 else 0
        # CUBLAS_COMPUTE_32F_PEDANTIC=69 disables TF32 for FP32 operands.
        compute,algorithm=(68,99) if dtype==2 else (69,-1)
        self.ck(self.lib.cublasSetStream_v2(self.handle,C.c_void_p(torch.cuda.current_stream().cuda_stream)))
        lda=b.shape[-1]
        ldb=a.shape[-1]
        args=[self.handle,int(tb),int(ta),n,m,k,C.byref(self.alpha),C.c_void_p(b.data_ptr()),dtype,lda]
        rest=[C.c_void_p(a.data_ptr()),dtype,ldb]
        tail=[C.byref(self.beta),C.c_void_p(out.data_ptr()),0,n]
        if out.ndim==3:
            batch=out.shape[0]
            self.ck(self.batched(*args,b.shape[-2]*b.shape[-1],*rest,a.shape[-2]*a.shape[-1],
                                  *tail,m*n,batch,compute,algorithm))
        else:self.ck(self.gemm(*args,*rest,*tail,compute,algorithm))

    def close(self):self.lib.cublasDestroy_v2(self.handle)


class Engine:
    """A shared logical launch plan executed by direct Triton or vendor tools."""
    def __init__(self,plan,weights,root,kind):
        self.plan,self.root,self.kind=plan,Path(root),kind
        self.specs,self.steps,self.kernels=prepare(plan)
        self.driver=Driver();self.backings=[];self.backing_labels=[];self.modules=[];self.functions={}
        self.weights={n:v.cuda() for n,v in weights.items() if n in {p['name'] for p in plan['parameters']}}
        self.state={}
        external={s['name'] for s in plan['inputs']+plan['parameters']}
        aliases=plan.get('aliases',{})
        layout=plan.get('storage')
        if layout:
            slots=[self.allocate([s['elements']],label='slot:'+str(i)) for i,s in enumerate(layout['slots'])]
            self.state={n:slots[i].view(self.specs[n]['shape']) for n,i in layout['assignments'].items()}
        else:
            self.state={n:self.allocate(s['shape'],label='state:'+n) for n,s in self.specs.items() if n not in external and n not in aliases}
        self.state.update(self.weights)
        self.inputs={s['name']:self.allocate(s['shape'],label='input:'+s['name']) for s in plan['inputs']}
        self.state.update(self.inputs)
        self.output=plan['outputs'][0]
        self.target=self.allocate(self.specs[self.output]['shape'],label='target')
        self.loss_value=self.allocate([],label='loss')
        self.seed=self.allocate(self.specs[self.output]['shape'],label='seed')
        self.partials=self.allocate([max(1,math.ceil(math.prod(self.specs[self.output]['shape'])/1024))],label='partials')
        self.recipes=tensor_recipes(plan,self.specs);self.packed={}
        for recipe in self.recipes.values():
            for name,spec in recipe.workspaces.items():self.packed[name]=self.allocate(spec['shape'],torch.float16,label='packed:'+name)
        if kind!='triton':
            # Exact compiler-input control: all numerical contracts come from
            # production-emitted PTX, compiled by NVIDIA's ptxas in its own arm.
            from openassembler import build
            source=(self.root/'sources/native.ptx').read_text()
            binary=build.ptxas_build(source,'sm_80')
            self.load(binary,self.kernels)
            for recipe in self.recipes.values():
                for stage in recipe.stages:
                    if stage.name in self.functions:continue
                    binary=build.ptxas_build(stage.ptx,'sm_80')
                    self.load(binary,{stage.name:stage})
        self.blas=Blas() if kind=='nvidia_libraries' else None
        if self.blas:
            from openassembler import build
            for step in self.steps:
                if step['kernel']=='linear' and len(step['inputs'])==3 and not step.get('tensor'):
                    kernel=bias_kernel(tuple(self.specs[step['output']]['shape']))
                    if kernel.name not in self.functions:
                        self.load(build.ptxas_build(kernel.ptx,'sm_80'),{kernel.name:kernel})
        self.launch_log=[];self.capture_snapshots=False;self.snapshots=[]

    def allocate(self,shape,dtype=torch.float32,label='unlabelled'):
        backing,value=guarded(self.driver,shape,dtype)
        self.backings.append(backing)
        self.backing_labels.append(label)
        self.driver.memset(value.data_ptr(),0,value.numel()*value.element_size())
        return value

    def load(self,binary,kernels):
        module=self.driver.load(binary);self.modules.append(module)
        for n,kernel in kernels.items():
            fn=self.driver.function(module,n)
            if kernel.launch.get('smem',0)>49152:
                self.driver.ck(self.driver.cu.cuFuncSetAttribute(fn,8,kernel.launch['smem']),'reference shared opt-in')
            self.functions[n]=fn

    def op(self,op,xs,out,attrs=None,lr=.01):
        attrs=attrs or {}
        self.launch_log.append({'op':op,'backend':self.kind})
        if self.kind=='triton':
            from exp.ml_suite.triton_ops import execute
            execute(op,xs,out,attrs,self.plan.get('math','fp32'),self.partials,lr)
            return
        if self.blas and op in ('linear','linear_dx','linear_dw'):
            if op=='linear':self.blas.product(xs[0],xs[1],out,tb=True)
            elif op=='linear_dx':self.blas.product(xs[0],xs[1],out)
            elif op=='linear_dw':self.blas.product(xs[0],xs[1],out,ta=True)
            if len(xs)==3:
                # Bias uses the same PTX stage as the explicit tensor recipe.
                stage=bias_kernel(tuple(out.shape))
                launch(self.driver,self.functions[stage.name],stage.launch,[out,xs[2],out])
            return
        kernel=emit_kernel(op,[{'shape':list(x.shape)} for x in xs],{'shape':list(out.shape)},attrs)
        args=[C.c_uint64(t.data_ptr()) for t in [*xs,out]]
        if op=='sgd':args.append(C.c_float(lr))
        params=(C.c_void_p*len(args))(*[C.cast(C.pointer(a),C.c_void_p) for a in args])
        c=kernel.launch
        self.driver.ck(self.driver.cu.cuLaunchKernel(self.functions[kernel.name],*c['grid'],c['threads'],1,1,c['smem'],
            C.c_void_p(torch.cuda.current_stream().cuda_stream),params,None),'vendor PTX kernel')

    def run_phase(self,phase):
        for step in self.steps:
            if step['phase']!=phase:continue
            if step.get('alias'):
                self.state[step['output']]=self.state[step['inputs'][0]].view(self.specs[step['output']]['shape'])
                continue
            xs=[self.state[n] for n in step['inputs']];out=self.state[step['output']]
            if step.get('tensor') and self.kind!='triton':
                recipe=self.recipes[step['entry']]
                bindings={f'a{i}':x for i,x in enumerate(xs)};bindings.update(self.packed);bindings['output']=out
                for stage in recipe.stages:
                    if self.blas and stage.gemm_plan is not None:
                        self.blas.product(bindings[stage.inputs[0]],bindings[stage.inputs[1]],out,tb=True)
                    else:launch(self.driver,self.functions[stage.name],stage.launch,
                                [*[bindings[n] for n in stage.inputs],bindings[stage.output]])
            else:self.op(step['kernel'],xs,out,step.get('attrs'))
            if self.capture_snapshots:
                self.snapshots.extend((n,x.detach().cpu().clone()) for n,x in zip(step['inputs'],xs))
            if self.capture_snapshots:self.snapshots.append((step['output'],out.detach().cpu().clone()))

    def forward(self):
        self.run_phase('forward');return self.state[self.output]

    def loss(self):
        self.op('mse',[self.state[self.output],self.target],self.loss_value)
        self.op('mse_grad',[self.state[self.output],self.target],self.seed)

    def backward(self):
        self.state[self.plan['backward']['seed_grads'][self.output]]=self.seed
        self.run_phase('backward')

    def update(self):
        for n,g in self.plan['backward']['grad_parameters'].items():
            self.op('sgd',[self.weights[n],self.state[g]],self.weights[n],lr=.01)

    def train(self):
        self.launch_log=[]
        self.forward();self.loss();self.backward();self.update()

    def fields(self):
        return {'prediction':self.state[self.output],'loss':self.loss_value,
                **{'dx/'+n:self.state[g] for n,g in self.plan['backward']['grad_inputs'].items()},
                **{'dw/'+n:self.state[g] for n,g in self.plan['backward']['grad_parameters'].items()},
                **{'w/'+n:v for n,v in self.weights.items()}}

    def reset(self,weights):
        for n,v in self.weights.items():v.copy_(weights[n])

    def check_guards(self):
        try:guards(self.driver,self.backings)
        except AssertionError as exc:raise AssertionError(f'{self.kind} guarded allocation map {self.backing_labels}: {exc}') from exc

    def close(self):
        torch.cuda.synchronize()
        if self.blas:self.blas.close()
        for module in self.modules:self.driver.unload(module)
