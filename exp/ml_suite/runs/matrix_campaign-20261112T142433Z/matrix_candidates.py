"""FP32 matrix candidates: pointer induction and guarded shared-memory tiling."""
from dataclasses import replace
import hashlib


def matrix_dims(kernel):
    a,b=kernel.inputs[:2]
    if kernel.op=='linear':
        m,k=a;n=b[0];return m,n,k,k,1,k,1
    if kernel.op=='linear_dx':
        m,n=a;k=b[1];return m,k,n,n,1,1,k
    if kernel.op=='linear_dw':
        m,n=a;k=b[1];return n,k,m,1,n,1,k
    raise ValueError('not a linear product')


def pointer(kernel, threads=128):
    m,n,k,ar,ak,bc,bk=matrix_dims(kernel)
    name='oa_ptr_'+hashlib.sha256((kernel.ptx+str(threads)).encode()).hexdigest()[:20]
    params=[f'.param .u64 a{i}' for i in range(len(kernel.inputs))]+['.param .u64 output']
    lines=[f'.visible .entry {name}({",".join(params)}) {{',
        '.reg .b32 %r<16>; .reg .b64 %d<16>; .reg .f32 %f<8>; .reg .pred %p<3>;']
    for i in range(len(kernel.inputs)):lines.append(f'ld.param.u64 %d{i},[a{i}];')
    lines+=['ld.param.u64 %d3,[output]; mov.u32 %r0,%tid.x; mov.u32 %r1,%ctaid.x;',
        f'mad.lo.u32 %r2,%r1,{threads},%r0; mov.u32 %r3,%ctaid.y;',
        f'setp.ge.u32 %p0,%r2,{n}; @%p0 bra done;',
        f'mul.lo.u32 %r4,%r3,{ar}; mul.lo.u32 %r5,%r2,{bc};',
        'mul.wide.u32 %d4,%r4,4; add.u64 %d5,%d0,%d4;',
        'mul.wide.u32 %d6,%r5,4; add.u64 %d7,%d1,%d6;',
        'mov.f32 %f0,0f00000000; mov.u32 %r6,0;',
        'loop:', 'ld.global.f32 %f1,[%d5]; ld.global.f32 %f2,[%d7];',
        'mul.rn.f32 %f3,%f1,%f2; add.rn.f32 %f0,%f0,%f3;',
        f'add.u64 %d5,%d5,{ak*4}; add.u64 %d7,%d7,{bk*4};',
        f'add.u32 %r6,%r6,1; setp.lt.u32 %p1,%r6,{k}; @%p1 bra loop;']
    if len(kernel.inputs)==3:
        lines+=['mul.wide.u32 %d8,%r2,4; add.u64 %d9,%d2,%d8;',
                'ld.global.f32 %f1,[%d9]; add.rn.f32 %f0,%f0,%f1;']
    lines += [f'mad.lo.u32 %r7,%r3,{n},%r2; mul.wide.u32 %d10,%r7,4;',
              'add.u64 %d11,%d3,%d10; st.global.f32 [%d11],%f0; done: ret; }']
    return replace(kernel,name=name,ptx=chr(10).join(lines),grid=((n+threads-1)//threads,m,1),threads=threads)


def tiled(kernel):
    m,n,k,ar,ak,bc,bk=matrix_dims(kernel)
    name='oa_tiled_'+hashlib.sha256(kernel.ptx.encode()).hexdigest()[:20]
    # 4x32 outputs, K blocks of 32, 128 threads. B rows are padded to 33
    # shared elements to avoid the column-read bank conflict.
    params=[f'.param .u64 a{i}' for i in range(len(kernel.inputs))]+['.param .u64 output']
    lines=[f'.visible .entry {name}({",".join(params)}) {{',
        '.reg .b32 %r<32>; .reg .b64 %d<16>; .reg .f32 %f<8>; .reg .pred %p<6>;',
        '.shared .align 4 .b8 smem[4736];']
    for i in range(len(kernel.inputs)):lines.append(f'ld.param.u64 %d{i},[a{i}];')
    lines+=['ld.param.u64 %d3,[output]; mov.u32 %r0,%tid.x;',
        'mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ctaid.y;',
        'and.b32 %r3,%r0,31; shr.u32 %r4,%r0,5;',
        'mad.lo.u32 %r5,%r1,32,%r3; mad.lo.u32 %r6,%r2,4,%r4;',
        'mov.u32 %r7,0; mov.f32 %f0,0f00000000; mov.u32 %r30,smem;', 'loop:']
    # A tile: four rows by 32.
    lines += ['add.u32 %r8,%r7,%r3;', f'setp.lt.u32 %p0,%r6,{m}; setp.lt.u32 %p1,%r8,{k};',
        'and.pred %p2,%p0,%p1;', f'mul.lo.u32 %r9,%r6,{ar}; mad.lo.u32 %r9,%r8,{ak},%r9;',
        'mul.wide.u32 %d4,%r9,4; add.u64 %d5,%d0,%d4;',
        'mov.f32 %f1,0f00000000; @%p2 ld.global.f32 %f1,[%d5];',
        'mul.lo.u32 %r10,%r0,4; add.u32 %r10,%r30,%r10; st.shared.f32 [%r10],%f1;']
    for j in range(8):
        lines += [f'add.u32 %r11,%r0,{j*128};']
        if bc==1:
            lines += ['and.b32 %r12,%r11,31; shr.u32 %r13,%r11,5;']
        else:
            lines += ['and.b32 %r13,%r11,31; shr.u32 %r12,%r11,5;']
        lines += ['mad.lo.u32 %r14,%r1,32,%r12; add.u32 %r15,%r7,%r13;',
            f'setp.lt.u32 %p0,%r14,{n}; setp.lt.u32 %p1,%r15,{k}; and.pred %p2,%p0,%p1;',
            f'mul.lo.u32 %r16,%r14,{bc}; mad.lo.u32 %r16,%r15,{bk},%r16;',
            'mul.wide.u32 %d6,%r16,4; add.u64 %d7,%d1,%d6;',
            'mov.f32 %f1,0f00000000; @%p2 ld.global.f32 %f1,[%d7];',
            'mad.lo.u32 %r17,%r12,33,%r13; mad.lo.u32 %r17,%r17,4,512;',
            'add.u32 %r17,%r30,%r17; st.shared.f32 [%r17],%f1;']
    lines+=['bar.sync 0;', 'mul.lo.u32 %r18,%r4,128; add.u32 %r18,%r30,%r18;',
        'mad.lo.u32 %r19,%r3,132,512; add.u32 %r19,%r30,%r19;']
    for q in range(32):
        lines += [f'ld.shared.f32 %f1,[%r18+{q*4}]; ld.shared.f32 %f2,[%r19+{q*4}];',
                  'mul.rn.f32 %f3,%f1,%f2; add.rn.f32 %f0,%f0,%f3;']
    lines += ['bar.sync 0;', f'add.u32 %r7,%r7,32; setp.lt.u32 %p3,%r7,{k}; @%p3 bra loop;',
        f'setp.lt.u32 %p0,%r6,{m}; setp.lt.u32 %p1,%r5,{n}; and.pred %p2,%p0,%p1; @!%p2 bra done;']
    if len(kernel.inputs)==3:
        lines+=['mul.wide.u32 %d8,%r5,4; add.u64 %d9,%d2,%d8;',
                'ld.global.f32 %f1,[%d9]; add.rn.f32 %f0,%f0,%f1;']
    lines += [f'mad.lo.u32 %r20,%r6,{n},%r5; mul.wide.u32 %d10,%r20,4;',
        'add.u64 %d11,%d3,%d10; st.global.f32 [%d11],%f0; done: ret; }']
    return replace(kernel,name=name,ptx=chr(10).join(lines),grid=((n+31)//32,(m+3)//4,1),threads=128)
