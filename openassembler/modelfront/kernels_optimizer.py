"""Explicit extended SGD state and AMSGrad maximum kernels, FP32 only."""
import hashlib
import math

OPTIMIZER_OPS=frozenset({'sgd_buffer_init','sgd_buffer','sgd_update','adam_max'})

def emit_optimizer_kernel(op,inputs,output,attrs=None):
    from .kernels import KernelSpec,canonical,shape_of,_float
    from .optimizer import optimizer_options
    attrs=optimizer_options(attrs);shapes=tuple(shape_of(s) for s in inputs);out=shape_of(output)
    momentum=attrs.get('momentum',0)
    if op=='adam_max':valid=attrs['name'] in ('adam','adamw') and attrs.get('amsgrad',False) and len(shapes)==2
    else:
        arity={'sgd_buffer_init':2,'sgd_buffer':3,'sgd_update':3 if momentum else 2}[op]
        valid=attrs['name']=='sgd' and len(shapes)==arity and (op=='sgd_update' or momentum>0)
    if not valid or any(s!=out for s in shapes):raise ValueError('invalid extended optimizer contract')
    sig={'op':op,'inputs':shapes,'output':out,'attrs':attrs}
    name='oa_model_'+hashlib.sha256(canonical(sig)).hexdigest()[:20]
    params=[f'.param .u64 a{i}' for i in range(len(inputs))]+['.param .u64 output']
    if op=='sgd_update':params+=['.param .f32 rate']
    predicates='%p,%q' if op=='adam_max' else '%p'
    lines=[f'.visible .entry {name}({",".join(params)}) {{',
           f'.reg .b32 %r<4>; .reg .b64 %d<10>; .reg .f32 %f<8>; .reg .pred {predicates};']
    for i in range(len(inputs)):lines+=[f'ld.param.u64 %d{i},[a{i}];']
    count=math.prod(out)
    lines+=['ld.param.u64 %d3,[output]; mov.u32 %r0,%tid.x; mov.u32 %r1,%ctaid.x;',
            f'mad.lo.u32 %r2,%r1,128,%r0; setp.ge.u32 %p,%r2,{count}; @%p bra done;',
            'mul.wide.u32 %d4,%r2,4;']
    for i in range(len(inputs)):
        lines += [f'add.u64 %d5,%d{i},%d4; ld.global.f32 %f{i},[%d5];']
    if op=='adam_max':
        # Torch maximum propagates either NaN and keeps the left signed zero
        # on equality. PTX max ignores one NaN and need not preserve that tie.
        lines+=['setp.nan.f32 %q,%f0,%f1; setp.ge.f32 %p,%f0,%f1;',
                'selp.f32 %f0,%f0,%f1,%p; selp.f32 %f0,0f7fc00000,%f0,%q;']
    else:
        # Normalize effective gradient before momentum: maximize reverses raw
        # gradient only; coupled decay keeps its ordinary sign.
        gradient,weight=('%f0','%f1') if op=='sgd_buffer_init' else ('%f1','%f2') if op=='sgd_buffer' else ('%f1','%f0')
        if attrs.get('maximize',False):lines += [f'neg.f32 {gradient},{gradient};']
        if attrs.get('weight_decay',0):
            lines += [f'mul.rn.f32 %f4,{weight},{_float(attrs["weight_decay"])}; add.rn.f32 {gradient},{gradient},%f4;']
        if op=='sgd_buffer':
            lines += [f'mul.rn.f32 %f0,%f0,{_float(momentum)};',
                      f'mul.rn.f32 %f1,%f1,{_float(1-attrs.get("dampening",0))}; add.rn.f32 %f0,%f0,%f1;']
        elif op=='sgd_update':
            if momentum:
                if attrs.get('nesterov',False):
                    lines += [f'mul.rn.f32 %f4,%f2,{_float(momentum)}; add.rn.f32 %f1,%f1,%f4;']
                else:lines += ['mov.f32 %f1,%f2;']
            lines += ['ld.param.f32 %f3,[rate]; mul.rn.f32 %f1,%f1,%f3; sub.rn.f32 %f0,%f0,%f1;']
    lines += ['add.u64 %d5,%d3,%d4; st.global.f32 [%d5],%f0;','done:','ret;','}']
    return KernelSpec(name,op,shapes,out,attrs,((count+127)//128,1,1),128,'\n'.join(lines)+'\n')
