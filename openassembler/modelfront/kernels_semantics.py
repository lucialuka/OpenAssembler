"""Exact-shape mixed-input semantics; deterministic FP32 computation only."""
import hashlib
import math

SEMANTIC_OPS=frozenset({'broadcast_add','broadcast_mul','broadcast_to','sum_to','embedding','embedding_grad',
                       'masked_softmax','ce_rows','ce_grad','mean','adam_m','adam_v','adam_update','adam_fused',
                       'sgd_buffer_init','sgd_buffer','sgd_update','adam_max'})
AUX_DTYPES=('f32','i64','bool')

def broadcast_shape(*shapes):
    rank=max(map(len,shapes));result=[1]*rank
    for shape in shapes:
        for i,dim in enumerate((1,)*(rank-len(shape))+tuple(shape)):
            if result[i] not in (1,dim) and dim!=1:raise ValueError('incompatible broadcast dimensions')
            result[i]=max(result[i],dim)
    if rank>4:raise ValueError('broadcast rank must be at most four')
    return tuple(result)


def emit_semantic_kernel(op,inputs,output,attrs=None):
    from .kernels_optimizer import OPTIMIZER_OPS,emit_optimizer_kernel
    if op in OPTIMIZER_OPS:return emit_optimizer_kernel(op,inputs,output,attrs)
    from .kernels import KernelSpec,canonical,shape_of,_float
    attrs=dict(attrs or {});out=shape_of(output)
    shapes=tuple(shape_of(s,dtypes=AUX_DTYPES) for s in inputs)
    types=[s.get('dtype','f32') for s in inputs]
    valid=False
    if op=='broadcast_to':
        valid=len(shapes)==1 and types==['f32'] and broadcast_shape(shapes[0],out)==out and len(shapes[0])<=len(out) and not attrs
    elif op in ('broadcast_add','broadcast_mul'):
        valid=len(shapes)==2 and types==['f32','f32'] and broadcast_shape(*shapes)==out and not attrs
    elif op=='sum_to':
        valid=len(shapes)==1 and types==['f32'] and broadcast_shape(shapes[0],out)==shapes[0] and len(out)<=len(shapes[0]) and not attrs
    elif op=='embedding':
        valid=len(shapes)==2 and types==['i64','f32'] and len(shapes[1])==2 and out==(*shapes[0],shapes[1][1]) and not attrs
    elif op=='embedding_grad':
        valid=len(shapes)==2 and types==['i64','f32'] and len(out)==2 and shapes[1]==(*shapes[0],out[1]) and not attrs
    elif op=='masked_softmax':
        valid=(len(shapes) in (1,2) and types[0]=='f32' and out==shapes[0] and 2<=len(out)<=4 and out[-1]<=1024
               and set(attrs)=={'causal'} and type(attrs['causal']) is bool
               and (len(shapes)==1 or types[1]=='bool' and broadcast_shape(shapes[0],shapes[1])==out and not attrs['causal']))
    elif op in ('ce_rows','ce_grad'):
        valid=(len(shapes)==2 and types==['f32','i64'] and len(shapes[0])==2 and shapes[0][1]<=1024
               and shapes[1]==shapes[0][:1] and out==(shapes[0][:1] if op=='ce_rows' else shapes[0]) and not attrs)
    elif op=='mean':valid=len(shapes)==1 and types==['f32'] and math.prod(out)==1 and not attrs
    elif op.startswith('adam_'):
        from .optimizer import optimizer_options
        attrs=optimizer_options(attrs)
        if op=='adam_fused' and attrs.get('amsgrad',False):raise ValueError('AMSGrad requires split optimizer updates')
        valid=attrs['name'] in ('adam','adamw') and len(shapes)==(4 if op in ('adam_update','adam_fused') else 3) and all(s==out for s in shapes) and all(t=='f32' for t in types)
    if not valid:raise ValueError(f'invalid {op} shapes, dtypes or attributes')
    # Dtypes are fixed by each op contract, hence cannot collide with FP32 operators.
    sig={'op':op,'inputs':shapes,'output':out,'attrs':attrs}
    name='oa_model_'+hashlib.sha256(canonical(sig)).hexdigest()[:20]
    params=[f'.param .u64 a{i}' for i in range(len(inputs))]+['.param .u64 output']
    if op in ('adam_update','adam_fused'):params+=['.param .f32 rate','.param .f32 correction1','.param .f32 correction2']
    lines=[f'.visible .entry {name}({",".join(params)}) {{',
           '.reg .b32 %r<64>; .reg .b64 %d<16>; .reg .f32 %f<120>; .reg .pred %p<8>;']
    for i in range(len(inputs)):lines += [f'ld.param.u64 %d{i},[a{i}];']
    lines+=['ld.param.u64 %d4,[output]; mov.u32 %r0,%tid.x; mov.u32 %r1,%ctaid.x;']
    count=math.prod(out);grid=((count+127)//128,1,1);threads=128
    def index(flat,full,target,dst):
        padded=(1,)*(len(full)-len(target))+tuple(target)
        lines.append(f'mov.u32 {dst},0;')
        for axis,dim in enumerate(full):
            if padded[axis]==1:continue
            stride=math.prod(full[axis+1:]);ts=math.prod(padded[axis+1:])
            lines.extend([f'div.u32 %r30,{flat},{stride}; rem.u32 %r30,%r30,{dim};',f'mad.lo.u32 {dst},%r30,{ts},{dst};'])
    def load(base,idx,reg,size=4,kind='f32'):
        lines.extend([f'mul.wide.u32 %d10,{idx},{size}; add.u64 %d11,%d{base},%d10;',f'ld.global.{kind} {reg},[%d11];'])
    def store(idx,reg='%f0'):
        lines.extend([f'mul.wide.u32 %d10,{idx},4; add.u64 %d11,%d4,%d10;',f'st.global.f32 [%d11],{reg};'])
    if op in ('masked_softmax','ce_rows','ce_grad'):
        rows=math.prod(shapes[0][:-1]);width=shapes[0][-1];chunks=(width+31)//32
        grid=((rows+3)//4,1,1)
        lines += ['and.b32 %r2,%r0,31; shr.u32 %r3,%r0,5; mad.lo.u32 %r4,%r1,4,%r3;',
                  f'setp.ge.u32 %p0,%r4,{rows}; @%p0 bra done;']
        for j in range(chunks):
            lines += [f'add.u32 %r5,%r2,{j*32}; setp.lt.u32 %p1,%r5,{width};',
                      f'mad.lo.u32 %r6,%r4,{width},%r5;']
            if op=='masked_softmax' and attrs['causal']:
                lines += [f'rem.u32 %r7,%r4,{out[-2]}; setp.le.u32 %p2,%r5,%r7; and.pred %p1,%p1,%p2;']
            elif op=='masked_softmax' and len(inputs)==2:
                index('%r6',out,shapes[1],'%r8')
                lines += ['mov.u32 %r9,0; mul.wide.u32 %d10,%r8,1; add.u64 %d11,%d1,%d10;',
                          '@%p1 ld.global.u8 %r9,[%d11]; setp.ne.u32 %p2,%r9,0; and.pred %p1,%p1,%p2;']
            lines += [f'selp.u32 %r{32+j},1,0,%p1;',f'mov.f32 %f{j},0fff800000;',
                      'mul.wide.u32 %d10,%r6,4; add.u64 %d11,%d0,%d10;',f'@%p1 ld.global.f32 %f{j},[%d11];']
        def reduce(kind,base,target):
            lines.append(f'mov.f32 %f{target},%f{base};')
            for j in range(1,chunks):lines.append(f'{kind}.f32 %f{target},%f{target},%f{base+j};')
            for delta in (16,8,4,2,1):lines.extend([f'shfl.sync.bfly.b32 %f119,%f{target},{delta},31,0xffffffff;',f'{kind}.f32 %f{target},%f{target},%f119;'])
        reduce('max',0,96)
        lines += ['setp.eq.f32 %p3,%f96,0fff800000; @%p3 mov.f32 %f96,0f00000000;']
        for j in range(chunks):
            lines += [f'sub.rn.f32 %f{64+j},%f{j},%f96; mul.rn.f32 %f{64+j},%f{64+j},{_float(math.log2(math.e))};',
                      f'ex2.approx.ftz.f32 %f{64+j},%f{64+j}; setp.eq.u32 %p2,%r{32+j},0; @%p2 mov.f32 %f{64+j},0f00000000;']
        reduce('add',64,97)
        lines += [f'max.f32 %f98,%f97,{_float(1e-30)}; rcp.approx.ftz.f32 %f98,%f98;']
        if op=='ce_rows':
            load(1,'%r4','%d12',8,'u64')
            lines += ['cvt.u32.u64 %r8,%d12;',f'mad.lo.u32 %r8,%r4,{width},%r8;']
            load(0,'%r8','%f99')
            lines += [f'lg2.approx.ftz.f32 %f100,%f97; mul.rn.f32 %f100,%f100,{_float(math.log(2))};',
                      'sub.rn.f32 %f0,%f96,%f99; add.rn.f32 %f0,%f0,%f100; setp.ne.u32 %p2,%r2,0; @%p2 bra done;']
            store('%r4')
        else:
            if op=='ce_grad':
                load(1,'%r4','%d12',8,'u64');lines+=['cvt.u32.u64 %r8,%d12;']
            for j in range(chunks):
                lines += [f'add.u32 %r5,%r2,{32*j}; setp.lt.u32 %p1,%r5,{width};',f'mad.lo.u32 %r6,%r4,{width},%r5;',f'mul.rn.f32 %f0,%f{64+j},%f98;']
                if op=='ce_grad':
                    lines += [f'setp.eq.u32 %p2,%r5,%r8; selp.f32 %f1,{_float(1)},0f00000000,%p2;',f'sub.rn.f32 %f0,%f0,%f1; mul.rn.f32 %f0,%f0,{_float(1/rows)};']
                lines+=['mul.wide.u32 %d10,%r6,4; add.u64 %d11,%d4,%d10; @%p1 st.global.f32 [%d11],%f0;']
    else:
        lines += [f'mad.lo.u32 %r2,%r1,128,%r0; setp.ge.u32 %p0,%r2,{count}; @%p0 bra done;']
        if op=='broadcast_to':
            index('%r2',out,shapes[0],'%r3');load(0,'%r3','%f0')
        elif op in ('broadcast_add','broadcast_mul'):
            for i,s in enumerate(shapes):index('%r2',out,s,'%r3');load(i,'%r3',f'%f{i}')
            lines.append(f'{"add" if op.endswith("add") else "mul"}.rn.f32 %f0,%f0,%f1;')
        elif op=='sum_to':
            full=shapes[0];padded=(1,)*(len(full)-len(out))+out
            # Base coordinates in full tensor; enumerate only reduced axes.
            lines += ['mov.u32 %r3,0;']
            for a,dim in enumerate(padded):
                if dim!=1:
                    lines += [f'div.u32 %r30,%r2,{math.prod(padded[a+1:])}; rem.u32 %r30,%r30,{dim};',f'mad.lo.u32 %r3,%r30,{math.prod(full[a+1:])},%r3;']
            reduced=tuple(full[a] if padded[a]==1 else 1 for a in range(len(full)))
            lines += ['mov.u32 %r4,0; mov.f32 %f0,0f00000000;','loop:','mov.u32 %r5,%r3;']
            for a,dim in enumerate(reduced):
                if dim!=1:lines += [f'div.u32 %r30,%r4,{math.prod(reduced[a+1:])}; rem.u32 %r30,%r30,{dim};',f'mad.lo.u32 %r5,%r30,{math.prod(full[a+1:])},%r5;']
            load(0,'%r5','%f1')
            lines += ['add.rn.f32 %f0,%f0,%f1;',f'add.u32 %r4,%r4,1; setp.lt.u32 %p1,%r4,{math.prod(reduced)}; @%p1 bra loop;']
        elif op=='embedding':
            width=shapes[1][1]
            lines += [f'div.u32 %r3,%r2,{width}; rem.u32 %r4,%r2,{width};']
            load(0,'%r3','%d12',8,'u64');lines+=['cvt.u32.u64 %r3,%d12;',f'mad.lo.u32 %r3,%r3,{width},%r4;']
            load(1,'%r3','%f0')
        elif op=='embedding_grad':
            width=out[1];tokens=math.prod(shapes[0])
            lines += [f'div.u32 %r3,%r2,{width}; rem.u32 %r4,%r2,{width}; mov.u32 %r5,0; mov.f32 %f0,0f00000000;','loop:']
            load(0,'%r5','%d12',8,'u64');lines+=['cvt.u32.u64 %r6,%d12; setp.ne.u32 %p1,%r6,%r3; @%p1 bra next;',f'mad.lo.u32 %r6,%r5,{width},%r4;']
            load(1,'%r6','%f1');lines+=['add.rn.f32 %f0,%f0,%f1;','next:',f'add.u32 %r5,%r5,1; setp.lt.u32 %p1,%r5,{tokens}; @%p1 bra loop;']
        elif op=='mean':
            lines += ['mov.f32 %f0,0f00000000; mov.u32 %r3,0;','loop:'];load(0,'%r3','%f1')
            lines += ['add.rn.f32 %f0,%f0,%f1;',f'add.u32 %r3,%r3,1; setp.lt.u32 %p1,%r3,{math.prod(shapes[0])}; @%p1 bra loop;',f'mul.rn.f32 %f0,%f0,{_float(1/math.prod(shapes[0]))};']
        elif op in ('adam_m','adam_v'):
            # Inputs: previous moment, raw gradient, weight. Coupled decay for Adam only.
            for i in range(3):load(i,'%r2',f'%f{i}')
            if attrs.get('maximize',False):lines += ['neg.f32 %f1,%f1;']
            if attrs['name']=='adam':lines += [f'mul.rn.f32 %f2,%f2,{_float(attrs["weight_decay"])}; add.rn.f32 %f1,%f1,%f2;']
            if op=='adam_m':
                lines += [f'sub.rn.f32 %f1,%f1,%f0; mul.rn.f32 %f1,%f1,{_float(1-attrs["beta1"])}; add.rn.f32 %f0,%f0,%f1;']
            else:
                lines += [f'mul.rn.f32 %f0,%f0,{_float(attrs["beta2"])}; mul.rn.f32 %f1,%f1,%f1; mul.rn.f32 %f1,%f1,{_float(1-attrs["beta2"])}; add.rn.f32 %f0,%f0,%f1;']
        elif op=='adam_fused':
            # ABI: weight, m, v, gradient, output weight; all reads precede writes.
            for i in range(4):load(i,'%r2',f'%f{i}')
            if attrs.get('maximize',False):lines += ['neg.f32 %f3,%f3;']
            lines += ['ld.param.f32 %f4,[rate]; ld.param.f32 %f5,[correction1]; ld.param.f32 %f6,[correction2];']
            if attrs['name']=='adam':
                lines += [f'mul.rn.f32 %f7,%f0,{_float(attrs["weight_decay"])}; add.rn.f32 %f3,%f3,%f7;']
            lines += [f'sub.rn.f32 %f7,%f3,%f1; mul.rn.f32 %f7,%f7,{_float(1-attrs["beta1"])}; add.rn.f32 %f1,%f1,%f7;',
                      f'mul.rn.f32 %f2,%f2,{_float(attrs["beta2"])}; mul.rn.f32 %f7,%f3,%f3; mul.rn.f32 %f7,%f7,{_float(1-attrs["beta2"])}; add.rn.f32 %f2,%f2,%f7;',
                      'mul.wide.u32 %d10,%r2,4; add.u64 %d11,%d1,%d10; st.global.f32 [%d11],%f1;',
                      'add.u64 %d11,%d2,%d10; st.global.f32 [%d11],%f2;',
                      'sqrt.rn.f32 %f2,%f2; div.rn.f32 %f2,%f2,%f6;',
                      f'add.rn.f32 %f2,%f2,{_float(attrs["epsilon"])}; div.rn.f32 %f1,%f1,%f2; mul.rn.f32 %f1,%f1,%f5;']
            if attrs['name']=='adamw':lines += [f'mul.rn.f32 %f7,%f4,{_float(attrs["weight_decay"])}; sub.rn.f32 %f7,{_float(1)},%f7; mul.rn.f32 %f0,%f0,%f7;']
            lines += ['sub.rn.f32 %f0,%f0,%f1;']
        elif op=='adam_update':
            # Inputs: weight, new m, new v, gradient (last pointer records use and identity).
            for i in range(3):load(i,'%r2',f'%f{i}')
            lines += ['ld.param.f32 %f3,[rate]; ld.param.f32 %f4,[correction1]; ld.param.f32 %f5,[correction2];',
                      'sqrt.rn.f32 %f2,%f2; div.rn.f32 %f2,%f2,%f5;',f'add.rn.f32 %f2,%f2,{_float(attrs["epsilon"])};',
                      'div.rn.f32 %f1,%f1,%f2; mul.rn.f32 %f1,%f1,%f4;']
            if attrs['name']=='adamw':lines += [f'mul.rn.f32 %f6,%f3,{_float(attrs["weight_decay"])}; sub.rn.f32 %f6,{_float(1)},%f6; mul.rn.f32 %f0,%f0,%f6;']
            lines += ['sub.rn.f32 %f0,%f0,%f1;']
        store('%r2')
    lines += ['done:','ret;','}']
    return KernelSpec(name,op,shapes,out,attrs,grid,threads,'\n'.join(lines)+'\n')
