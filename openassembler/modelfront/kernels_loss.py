"""Shape-admitted deterministic FP32 MSE reductions."""
from dataclasses import replace
import hashlib
import math


# Exact sizes with repeated A100 paired timing and complete training gates.
ORDERED_MSE_ELEMENTS = frozenset({112, 136, 513, 65539})
PARALLEL_MSE_SHAPES = frozenset({(128,128)})


def parallel_mse(baseline,threads=256):
    """Stable single-CTA FP32 reduction for the measured 16K loss contract."""
    from .kernels import _float
    count=math.prod(baseline.inputs[0])
    name='oa_parallel_mse_'+hashlib.sha256((baseline.ptx+str(threads)).encode()).hexdigest()[:16]
    lines=[f'.visible .entry {name}(.param .u64 a0,.param .u64 a1,.param .u64 output) {{',
        '.reg .b32 %r<12>; .reg .b64 %d<8>; .reg .f32 %f<8>; .reg .pred %p<4>;',
        f'.shared .align 4 .b8 smem[{threads*4}];',
        'ld.param.u64 %d0,[a0]; ld.param.u64 %d1,[a1]; ld.param.u64 %d2,[output];',
        'mov.u32 %r0,%tid.x; mov.u32 %r1,%tid.x; mov.f32 %f0,0f00000000;',
        'load_loop:', f'setp.ge.u32 %p0,%r1,{count}; @%p0 bra load_done;',
        'mul.wide.u32 %d3,%r1,4; add.u64 %d4,%d0,%d3; add.u64 %d5,%d1,%d3;',
        'ld.global.f32 %f1,[%d4]; ld.global.f32 %f2,[%d5];',
        'sub.rn.f32 %f3,%f1,%f2; mul.rn.f32 %f3,%f3,%f3; add.rn.f32 %f0,%f0,%f3;',
        f'add.u32 %r1,%r1,{threads}; bra load_loop;', 'load_done:',
        'mov.u32 %r2,smem; shl.b32 %r3,%r0,2; add.u32 %r2,%r2,%r3;',
        'st.shared.f32 [%r2],%f0; bar.sync 0;']
    for offset in (threads//2**i for i in range(1,int(math.log2(threads))+1)):
        label=f'reduce_{offset}_done'
        lines += [f'setp.ge.u32 %p1,%r0,{offset}; @%p1 bra {label};',
                  'ld.shared.f32 %f1,[%r2];',f'add.u32 %r3,%r2,{offset*4};',
                  'ld.shared.f32 %f2,[%r3];','add.rn.f32 %f1,%f1,%f2;',
                  'st.shared.f32 [%r2],%f1;',f'{label}:','bar.sync 0;']
    lines += ['setp.ne.u32 %p2,%r0,0; @%p2 bra done;',
              'ld.shared.f32 %f0,[%r2];',f'mul.rn.f32 %f0,%f0,{_float(1/count)};',
              'st.global.f32 [%d2],%f0;','done: ret;','}']
    return replace(baseline,name=name,ptx='\n'.join(lines)+'\n',grid=(1,1,1),threads=threads)


def ordered_mse(baseline):
    from .kernels import _float
    assert baseline.op == 'mse' and len(baseline.inputs) == 2
    count = math.prod(baseline.inputs[0])
    name = 'oa_ordered_mse_' + hashlib.sha256(baseline.ptx.encode()).hexdigest()[:16]
    lines = [f'.visible .entry {name}(.param .u64 a0,.param .u64 a1,.param .u64 output) {{',
             '.reg .b32 %r<8>; .reg .b64 %d<8>; .reg .f32 %f<8>; .reg .pred %p<4>;',
             'ld.param.u64 %d0,[a0]; ld.param.u64 %d1,[a1]; ld.param.u64 %d2,[output];',
             'mov.u32 %r0,%tid.x; mov.u32 %r1,0; mov.f32 %f0,0f00000000;',
             'loop:', 'add.u32 %r2,%r1,%r0;',
             f'setp.lt.u32 %p0,%r2,{count};',
             'mul.wide.u32 %d3,%r2,4; add.u64 %d4,%d0,%d3; add.u64 %d5,%d1,%d3;',
             'mov.f32 %f1,0f00000000; mov.f32 %f2,0f00000000;',
             '@%p0 ld.global.f32 %f1,[%d4]; @%p0 ld.global.f32 %f2,[%d5];',
             'sub.rn.f32 %f3,%f1,%f2; mul.rn.f32 %f3,%f3,%f3;']
    # All lanes participate. Lane 0 receives product[i] using XOR(i), in
    # increasing index order, with the same FP32 add sequence as scalar MSE.
    for lane in range(32):
        lines += [f'shfl.sync.bfly.b32 %f4,%f3,{lane},31,0xffffffff;',
                  'add.rn.f32 %f0,%f0,%f4;']
    lines += ['add.u32 %r1,%r1,32;', f'setp.lt.u32 %p1,%r1,{count}; @%p1 bra loop;',
              f'mul.rn.f32 %f0,%f0,{_float(1/count)};',
              'setp.eq.u32 %p2,%r0,0; @%p2 st.global.f32 [%d2],%f0;', 'ret;', '}']
    return replace(baseline, name=name, ptx=chr(10).join(lines)+chr(10), grid=(1,1,1), threads=32)
