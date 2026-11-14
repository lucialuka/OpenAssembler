"""Explicit dual-layout FP16 packing and padded native Tensor Core recipes.

Admitted after full training/replay validation under exp/pack_training. The
schedule refreshes every logical operand each iteration; it never caches across
parameter updates. Padded shapes use the same OA full-tile GEMM and an exact
FP32 crop/bias stage. Numerical policy remains FP16 RNE / FP32 accumulation.
"""
from dataclasses import asdict
import hashlib
from .kernels import canonical,shape_of,MAX_ELEMENTS
from .kernels_tensor import TensorStage,TensorRecipe,MATH_MODE
from openassembler.kernels.gemm import GemmPlan,generate


def convert():
    return ['shr.u32 %r8,%r7,16; and.b32 %r8,%r8,32768;',
        'and.b32 %r9,%r7,2147483647;',
        'shr.u32 %r10,%r9,13; and.b32 %r10,%r10,1;',
        'add.u32 %r11,%r9,3355447295; add.u32 %r11,%r11,%r10; shr.u32 %r11,%r11,13;',
        'mov.b32 %f0,%r9; add.rn.f32 %f1,%f0,0f3f000000; mov.b32 %r12,%f1;',
        'sub.u32 %r12,%r12,1056964608;',
        'setp.lt.u32 %p1,%r9,947912704; selp.u32 %r13,%r12,%r11,%p1;',
        'setp.ge.u32 %p1,%r9,1199566848; selp.u32 %r13,31744,%r13,%p1;',
        'setp.gt.u32 %p1,%r9,2139095040; selp.u32 %r13,32256,%r13,%p1;']


def dual_pack(shape,output):
    rows,cols=shape;pr=(rows+127)//128*128;pc=(cols+127)//128*128
    signature={'op':'pack_dual_f16_rne','shape':list(shape),'padded':[pr,pc],
               'layouts':['row_major','transpose'],'tile':[32,32,33],'nan':'canonical_quiet_preserve_sign'}
    name='oa_dual_pack_'+hashlib.sha256(canonical(signature)).hexdigest()[:20]
    lines=['.version 8.7\n.target sm_80\n.address_size 64',
        f'.visible .entry {name}(.param .u64 input,.param .u64 output) {{',
        '.reg .b32 %r<50>; .reg .b64 %d<12>; .reg .f32 %f<4>; .reg .pred %p<5>;',
        '.shared .align 4 .b8 tile[4224];',
        'ld.param.u64 %d0,[input]; ld.param.u64 %d1,[output];',
        'mov.u32 %r0,%tid.x; mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ctaid.y;',
        'shr.u32 %r30,%r0,4; and.b32 %r31,%r0,15; mul.lo.u32 %r31,%r31,2;',
        'mul.lo.u32 %r32,%r2,32; mul.lo.u32 %r33,%r1,32; mov.u32 %r40,tile;']
    for dr in (0,16):
        lines += [f'add.u32 %r34,%r30,{dr}; add.u32 %r3,%r32,%r34; add.u32 %r4,%r33,%r31;']
        for j in range(2):
            lines += [f'add.u32 %r6,%r4,{j};',f'setp.lt.u32 %p2,%r3,{rows}; setp.lt.u32 %p3,%r6,{cols}; and.pred %p4,%p2,%p3;',
                      f'mad.lo.u32 %r14,%r3,{cols},%r6; mul.wide.u32 %d2,%r14,4; add.u64 %d3,%d0,%d2;',
                      'mov.u32 %r7,0; @%p4 ld.global.b32 %r7,[%d3];',*convert(),f'or.b32 %r{20+j},%r13,%r8;',
                      f'add.u32 %r35,%r31,{j}; mad.lo.u32 %r36,%r34,33,%r35;',
                      'mul.lo.u32 %r36,%r36,4; add.u32 %r36,%r40,%r36;',f'st.shared.b32 [%r36],%r{20+j};']
        lines += ['shl.b32 %r24,%r21,16; or.b32 %r25,%r20,%r24;',
                  f'mad.lo.u32 %r15,%r3,{pc},%r4;',
                  'mul.wide.u32 %d4,%r15,2; add.u64 %d5,%d1,%d4; st.global.b32 [%d5],%r25;']
    lines+=['bar.sync 0;']
    for dr in (0,16):
        lines += [f'add.u32 %r34,%r30,{dr}; add.u32 %r3,%r33,%r34; add.u32 %r4,%r32,%r31;']
        for j in range(2):
            lines += [f'add.u32 %r35,%r31,{j}; mad.lo.u32 %r36,%r35,33,%r34;',
                      'mul.lo.u32 %r36,%r36,4; add.u32 %r36,%r40,%r36;',f'ld.shared.b32 %r{20+j},[%r36];']
        lines += ['shl.b32 %r24,%r21,16; or.b32 %r25,%r20,%r24;',
                  f'mad.lo.u32 %r15,%r3,{pr},%r4; add.u32 %r15,%r15,{pr*pc};',
                  'mul.wide.u32 %d4,%r15,2; add.u64 %d5,%d1,%d4; st.global.b32 [%d5],%r25;']
    lines+=['ret;','}']
    return TensorStage(name,'pack_dual_f16_rne',('a0',),output,'\n'.join(lines),
                       {'grid':[pc//32,pr//32,1],'threads':256,'smem':0},signature),pr,pc


def crop(shape,padded,bias,input_name):
    m,n=shape;pm,pn=padded
    signature={'op':'crop_bias','shape':list(shape),'padded':list(padded),'bias':bias}
    name='oa_crop_'+hashlib.sha256(canonical(signature)).hexdigest()[:20]
    params='.param .u64 input,'+('.param .u64 bias,' if bias else '')+'.param .u64 output'
    lines=['.version 8.7\n.target sm_80\n.address_size 64',f'.visible .entry {name}({params}) {{',
        '.reg .b32 %r<12>; .reg .b64 %d<10>; .reg .f32 %f<3>; .reg .pred %p;',
        'ld.param.u64 %d0,[input]; ld.param.u64 %d1,[output];',
        'mov.u32 %r0,%tid.x; mov.u32 %r1,%ctaid.x; mad.lo.u32 %r2,%r1,128,%r0;',
        f'setp.ge.u32 %p,%r2,{m*n}; @%p bra done;',
        f'div.u32 %r3,%r2,{n}; rem.u32 %r4,%r2,{n};',
        f'mad.lo.u32 %r5,%r3,{pn},%r4; mul.wide.u32 %d2,%r5,4; add.u64 %d3,%d0,%d2;',
        'ld.global.f32 %f0,[%d3];']
    if bias:lines+=['ld.param.u64 %d4,[bias]; mul.wide.u32 %d5,%r4,4; add.u64 %d6,%d4,%d5;',
                    'ld.global.f32 %f1,[%d6]; add.rn.f32 %f0,%f0,%f1;']
    lines+=['mul.wide.u32 %d7,%r2,4; add.u64 %d8,%d1,%d7; st.global.f32 [%d8],%f0; done: ret;','}']
    return TensorStage(name,'crop_bias',(input_name,'a2') if bias else (input_name,), 'output','\n'.join(lines),
                       {'grid':[(m*n+127)//128,1,1],'threads':128,'smem':0},signature)


def emit_dual_recipe(op,inputs,output):
    if op not in ('linear','linear_dx','linear_dw') or len(inputs) not in ((2,3) if op=='linear' else (2,)):
        raise ValueError('dual packing requires linear, dX or dW operands')
    a,b=[shape_of(s) for s in inputs[:2]];out=shape_of(output)
    if len(a)!=2 or len(b)!=2 or len(out)!=2:raise ValueError('dual packing requires rank-two matrices')
    if op=='linear':m,k=a;n=b[0];ta,tb=False,False;expected=(m,n)
    elif op=='linear_dx':m,k=a;n=b[1];ta,tb=False,True;expected=(m,n)
    elif op=='linear_dw':k,m=a;n=b[1];ta,tb=True,True;expected=(m,n)
    else:raise ValueError(op)
    if out!=expected:raise ValueError('output shape mismatch')
    if (op=='linear' and b[1]!=k or op=='linear_dx' and b[0]!=k or op=='linear_dw' and b[0]!=k):
        raise ValueError('dual matrix reduction dimensions disagree')
    if len(inputs)==3 and shape_of(inputs[2])!=(n,):raise ValueError('linear bias shape mismatch')
    # Bounded first implementation: no unbounded temporary allocation or
    # extrapolated large-matrix performance claim.
    if max(m,n,k)>512:raise ValueError('dual packing dimensions must be at most 512')
    pm,pn,pk=[(v+127)//128*128 for v in (m,n,k)]
    from openassembler.gemm_dispatch import select_plan
    plan,decision=select_plan(GemmPlan(pm,pn,pk,'f16'))
    signature={'op':op,'inputs':[list(shape_of(s)) for s in inputs],'output':list(out),'math_mode':MATH_MODE,
               'plan':asdict(plan),'pack':'dual_layout_per_iteration','padded':[pm,pn,pk]}
    name='oa_dual_tensor_'+hashlib.sha256(canonical(signature)).hexdigest()[:20]
    work={};stages=[];gemm_inputs=[]
    for i,(shape,tr) in enumerate(zip((a,b),(ta,tb))):
        base=name+f'_dual_{i}';view=name+f'_view_{i}'
        stage,pr,pc=dual_pack(shape,base)
        stage=__import__('dataclasses').replace(stage,inputs=(f'a{i}',))
        work[base]={'shape':[2*pr*pc],'dtype':'f16','alignment':16}
        work[view]={'shape':[pc,pr] if tr else [pr,pc],'dtype':'f16','base':base,'offset':pr*pc if tr else 0}
        stages.append(stage);gemm_inputs.append(view)
    padded=out!=(pm,pn);destination=name+'_padded_output' if padded else 'output'
    if padded:work[destination]={'shape':[pm,pn],'dtype':'f32','alignment':16}
    entry=name+'_gemm';ptx=generate(plan).replace('.entry gemm(',f'.entry {entry}(')
    stages.append(TensorStage(entry,op,tuple(gemm_inputs),destination,ptx,plan.launch,{'op':op,'plan':asdict(plan),'math_mode':MATH_MODE},plan))
    if padded or len(inputs)==3:stages.append(crop(out,(pm,pn),len(inputs)==3,destination))
    return TensorRecipe(name,op,signature,work,tuple(stages))
