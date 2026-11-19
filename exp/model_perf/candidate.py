"""Experimental coalesced transpose packing and small-grid tensor policy."""
from dataclasses import replace, asdict
import hashlib
from openassembler.modelfront.kernels import canonical
from openassembler.modelfront.kernels_tensor import TensorStage
from openassembler.modelfront.kernels_tensor_dual import dual_pack
from openassembler.kernels.gemm import generate


def transpose_pack(stage):
    shape=stage.signature['shape']
    if not stage.signature.get('transpose') or any(v%128 for v in shape):return stage
    raw,pr,pc=dual_pack(shape,stage.output)
    lines=raw.ptx.splitlines()
    # Preserve the tiled conversion/shared-memory synchronization, omit normal
    # output stores. The sole output is the compact transposed half matrix.
    normal_store='mul.wide.u32 %d4,%r15,2; add.u64 %d5,%d1,%d4; st.global.b32 [%d5],%r25;'
    seen=0;out=[]
    for line in lines:
        if line==normal_store:
            seen+=1
            if seen<=2:continue
        out.append(line.replace(f'add.u32 %r15,%r15,{pr*pc};','add.u32 %r15,%r15,0;'))
    assert seen==4
    signature={**stage.signature,'implementation':'coalesced_transpose_32x32'}
    name='oa_transpose_pack_'+hashlib.sha256(canonical(signature)).hexdigest()[:20]
    source='\n'.join(out).replace(raw.name,name)
    return replace(stage,name=name,ptx=source,signature=signature,launch=raw.launch)


def recipe(original, variant, op, inputs, output, attrs=None):
    rec=original(op,inputs,output,attrs)
    if variant=='baseline':return rec
    stages=[]
    for stage in rec.stages:
        if variant in ('pack','combined') and stage.op=='pack_f16_rne':stage=transpose_pack(stage)
        if variant in ('tile','combined') and stage.gemm_plan is not None:
            plan=stage.gemm_plan
            if max(plan.m,plan.n,plan.k)<=512 and plan.tile=='128x128':
                plan=replace(plan,tile='64x64')
                signature={**stage.signature,'plan':asdict(plan)}
                name='oa_small_gemm_'+hashlib.sha256(canonical(signature)).hexdigest()[:20]
                stage=replace(stage,name=name,ptx=generate(plan).replace('.entry gemm(',f'.entry {name}('),launch=plan.launch,signature=signature,gemm_plan=plan)
        stages.append(stage)
    return replace(rec,stages=tuple(stages))


def bias_reduce(base, warps=4):
    rows,cols=base.inputs[0];threads=32*warps
    sig={'baseline':base.signature,'warps':warps,'implementation':'coalesced_row_partials'}
    name='oa_bias_reduce_'+hashlib.sha256(canonical(sig)).hexdigest()[:20]
    ptx=f'''.visible .entry {name}(.param .u64 a0,.param .u64 output) {{
.reg .b32 %r<12>; .reg .b64 %d<6>; .reg .f32 %f<4>; .reg .pred %p<4>;
.shared .align 4 .b8 partials[{threads*4}];
ld.param.u64 %d0,[a0]; ld.param.u64 %d1,[output];
mov.u32 %r0,%tid.x; and.b32 %r1,%r0,31; shr.u32 %r2,%r0,5;
mov.u32 %r3,%ctaid.x; mad.lo.u32 %r3,%r3,32,%r1;
mov.u32 %r4,%r2; mov.f32 %f0,0f00000000;
loop: setp.ge.u32 %p0,%r4,{rows}; @%p0 bra loaded;
mad.lo.u32 %r5,%r4,{cols},%r3; mul.wide.u32 %d2,%r5,4; add.u64 %d3,%d0,%d2;
setp.lt.u32 %p1,%r3,{cols}; mov.f32 %f1,0f00000000; @%p1 ld.global.f32 %f1,[%d3];
add.rn.f32 %f0,%f0,%f1; add.u32 %r4,%r4,{warps}; bra loop;
loaded: mov.u32 %r6,partials; shl.b32 %r7,%r0,2; add.u32 %r7,%r6,%r7;
st.shared.f32 [%r7],%f0; bar.sync 0;
setp.ne.u32 %p2,%r2,0; @%p2 bra done;
mov.f32 %f0,0f00000000; mad.lo.u32 %r7,%r1,4,%r6;
'''
    for w in range(warps):
        ptx+=f'add.u32 %r8,%r7,{w*128}; ld.shared.f32 %f1,[%r8]; add.rn.f32 %f0,%f0,%f1;\n'
    ptx+=f'''mul.wide.u32 %d2,%r3,4; add.u64 %d3,%d1,%d2;
@%p1 st.global.f32 [%d3],%f0;
done: ret; }}
'''
    return replace(base,name=name,ptx=ptx,grid=((cols+31)//32,1,1),threads=threads)


def warp_mse(base, threads=1024):
    import math
    from openassembler.modelfront.kernels import _float
    count=math.prod(base.inputs[0]);warps=threads//32
    name='oa_warp_mse_'+hashlib.sha256((base.ptx+str(threads)).encode()).hexdigest()[:20]
    ptx=f'''.visible .entry {name}(.param .u64 a0,.param .u64 a1,.param .u64 output) {{
.reg .b32 %r<12>; .reg .b64 %d<8>; .reg .f32 %f<8>; .reg .pred %p<4>;
.shared .align 4 .b8 partials[{warps*4}];
ld.param.u64 %d0,[a0]; ld.param.u64 %d1,[a1]; ld.param.u64 %d2,[output];
mov.u32 %r0,%tid.x; mov.u32 %r1,%r0; mov.f32 %f0,0f00000000;
loop: setp.ge.u32 %p0,%r1,{count}; @%p0 bra loaded;
mul.wide.u32 %d3,%r1,4; add.u64 %d4,%d0,%d3; add.u64 %d5,%d1,%d3;
ld.global.f32 %f1,[%d4]; ld.global.f32 %f2,[%d5];
sub.rn.f32 %f3,%f1,%f2; mul.rn.f32 %f3,%f3,%f3; add.rn.f32 %f0,%f0,%f3;
add.u32 %r1,%r1,{threads}; bra loop;
loaded:
'''
    for offset in [16,8,4,2,1]:ptx+=f'shfl.sync.bfly.b32 %f1,%f0,{offset},31,0xffffffff; add.rn.f32 %f0,%f0,%f1;\n'
    ptx+='and.b32 %r2,%r0,31; shr.u32 %r3,%r0,5; mov.u32 %r4,partials; mad.lo.u32 %r5,%r3,4,%r4; setp.eq.u32 %p1,%r2,0; @%p1 st.shared.f32 [%r5],%f0; bar.sync 0;\n'
    ptx+=f'setp.ne.u32 %p2,%r3,0; @%p2 bra done; setp.lt.u32 %p1,%r2,{warps}; mad.lo.u32 %r5,%r2,4,%r4; mov.f32 %f0,0f00000000; @%p1 ld.shared.f32 %f0,[%r5];\n'
    for offset in [16,8,4,2,1]:ptx+=f'shfl.sync.bfly.b32 %f1,%f0,{offset},31,0xffffffff; add.rn.f32 %f0,%f0,%f1;\n'
    ptx+=f'mul.rn.f32 %f0,%f0,{_float(1/count)}; setp.eq.u32 %p1,%r0,0; @%p1 st.global.f32 [%d2],%f0; done: ret; }}\n'
    return replace(base,name=name,ptx=ptx,grid=(1,1,1),threads=threads)


def frozen_module(stem):
    """Load retained pre-change emitters with their original relative imports."""
    import importlib.util,sys
    from pathlib import Path
    name='openassembler.modelfront._perf_'+stem
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name('baseline_'+stem+'.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]
