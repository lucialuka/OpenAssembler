"""Bounded FP32 neural operators for the OA-native graph frontend.

Softmax uses cooperative full-warp reductions. GELU implements the tanh
approximation and its stable analytic derivative with approximate exp/reciprocal.
Inputs must be finite with finite FP32 intermediate sums/products; this is not
an exceptional-value or exact-erf contract.
"""
from __future__ import annotations

import hashlib
import math

NN_OPS = frozenset({"softmax", "softmax_backward", "gelu", "gelu_backward",
                    "layernorm", "layernorm_backward", "channel_affine", "channel_mul", "channel_grad", "channel_sum"})


def emit_nn_kernel(op, inputs, output, attrs=None):
    if op.startswith("layernorm") or op.startswith("channel_"):
        return _emit_norm_kernel(op,inputs,output,attrs)
    from .kernels import KernelSpec, _float, canonical, shape_of
    attrs = dict(attrs or {})
    shapes = tuple(shape_of(x) for x in inputs)
    out = shape_of(output)
    if op not in NN_OPS or len(shapes) != (2 if op.endswith("_backward") else 1) or any(x != out for x in shapes):
        raise ValueError("neural operator requires matching FP32 tensor shapes and exact input count")
    if attrs:
        raise ValueError(f"unsupported {op} attributes: {sorted(attrs)}")
    if op.startswith("softmax") and (not out or out[-1] > 1024):
        raise ValueError("native softmax requires last-dimension width in [1,1024]")
    signature = {"op":op,"inputs":shapes,"output":out,"attrs":attrs}
    name = "oa_model_" + hashlib.sha256(canonical(signature)).hexdigest()[:20]
    params = [f".param .u64 a{i}" for i in range(len(shapes))]+[".param .u64 output"]
    lines = [f".visible .entry {name}({','.join(params)}) {{",
             ".reg .b32 %r<16>; .reg .b64 %d<12>; .reg .f32 %f<120>; .reg .pred %p<4>;"]
    for i in range(len(shapes)):
        lines += [f"ld.param.u64 %d{i},[a{i}];"]
    lines += ["ld.param.u64 %d2,[output]; mov.u32 %r0,%tid.x; mov.u32 %r1,%ctaid.x;"]
    if op.startswith("softmax"):
        width = out[-1];rows = math.prod(out[:-1]);chunks=(width+31)//32;tail=width%32
        grid,threads=((rows+3)//4,1,1),128
        lines += ["and.b32 %r2,%r0,31; shr.u32 %r3,%r0,5; mad.lo.u32 %r4,%r1,4,%r3;",
                  f"setp.ge.u32 %p0,%r4,{rows}; @%p0 bra done;",
                  f"mad.lo.u32 %r5,%r4,{width},%r2; mul.wide.u32 %d3,%r5,4;",
                  "add.u64 %d4,%d0,%d3; add.u64 %d6,%d2,%d3;"]
        if len(shapes)==2:
            lines += ["add.u64 %d5,%d1,%d3;"]
        if tail:
            lines += [f"setp.lt.u32 %p1,%r2,{tail};"]
        for j in range(chunks):
            guard = "@%p1 " if tail and j==chunks-1 else ""
            if guard:
                lines += [f"mov.f32 %f{j},{'0fff800000' if op=='softmax' else '0f00000000'};"]
            lines += [f"{guard}ld.global.f32 %f{j},[%d4+{j*128}];"]
            if op.endswith("_backward"):
                if guard:lines += [f"mov.f32 %f{32+j},0f00000000;"]
                lines += [f"{guard}ld.global.f32 %f{32+j},[%d5+{j*128}];"]
        def reduce(kind,values,target):
            lines.append(f"mov.f32 %f{target},%f{values[0]};")
            for value in values[1:]:lines.append(f"{kind}.f32 %f{target},%f{target},%f{value};")
            for delta in (16,8,4,2,1):
                lines.extend([f"shfl.sync.bfly.b32 %f119,%f{target},{delta},31,0xffffffff;",
                              f"{kind}.f32 %f{target},%f{target},%f119;"])
        if op=="softmax":
            reduce("max",range(chunks),96)
            for j in range(chunks):
                lines += [f"sub.rn.f32 %f{j},%f{j},%f96; mul.rn.f32 %f{j},%f{j},{_float(math.log2(math.e))};",
                          f"ex2.approx.ftz.f32 %f{j},%f{j};"]
            reduce("add",range(chunks),97)
            lines += ["rcp.approx.ftz.f32 %f98,%f97;"]
            for j in range(chunks):lines += [f"mul.rn.f32 %f{j},%f{j},%f98;"]
        else:
            # Arguments are (upstream, saved softmax output), not pre-softmax x.
            for j in range(chunks):lines += [f"mul.rn.f32 %f{64+j},%f{j},%f{32+j};"]
            reduce("add",range(64,64+chunks),96)
            for j in range(chunks):
                lines += [f"sub.rn.f32 %f{j},%f{j},%f96; mul.rn.f32 %f{j},%f{j},%f{32+j};"]
        for j in range(chunks):
            guard="@%p1 " if tail and j==chunks-1 else ""
            lines += [f"{guard}st.global.f32 [%d6+{j*128}],%f{j};"]
    else:
        count=math.prod(out);grid,threads=((count+127)//128,1,1),128
        lines += [f"mad.lo.u32 %r2,%r1,128,%r0; setp.ge.u32 %p0,%r2,{count}; @%p0 bra done;",
                  "mul.wide.u32 %d3,%r2,4; add.u64 %d4,%d0,%d3; add.u64 %d6,%d2,%d3;",
                  "ld.global.f32 %f0,[%d4];"]
        if op.endswith("_backward"):
            lines += ["add.u64 %d5,%d1,%d3; ld.global.f32 %f1,[%d5];"]
            x="%f1"
        else:x="%f0"
        # sigmoid(2u) with sign-dependent stable reciprocal avoids cancellation
        # in both tails. sech(u)^2 comes directly from exp(-2*abs(u)).
        lines += [f"mul.rn.f32 %f2,{x},{x}; mul.rn.f32 %f3,%f2,{_float(.044715)};",
                  f"add.rn.f32 %f3,%f3,{_float(1)}; mul.rn.f32 %f3,%f3,{x};",
                  f"mul.rn.f32 %f3,%f3,{_float(.7978845608028654)}; abs.f32 %f4,%f3;",
                  f"mul.rn.f32 %f4,%f4,{_float(-2.8853900817779268)}; ex2.approx.ftz.f32 %f4,%f4;",
                  f"add.rn.f32 %f5,%f4,{_float(1)}; rcp.approx.ftz.f32 %f5,%f5;",
                  "mul.rn.f32 %f6,%f4,%f5; setp.ge.f32 %p1,%f3,0f00000000;",
                  "selp.f32 %f6,%f5,%f6,%p1;"]
        if op=="gelu":
            lines += ["mul.rn.f32 %f0,%f0,%f6;"]
        else:
            lines += [f"mul.rn.f32 %f2,%f2,{_float(3*.044715)}; add.rn.f32 %f2,%f2,{_float(1)};",
                      f"mul.rn.f32 %f2,%f2,{_float(.7978845608028654)}; mul.rn.f32 %f7,%f5,%f5;",
                      f"mul.rn.f32 %f7,%f7,%f4; mul.rn.f32 %f7,%f7,{x};",
                      f"mul.rn.f32 %f7,%f7,{_float(2)}; mul.rn.f32 %f7,%f7,%f2;",
                      "add.rn.f32 %f7,%f7,%f6; mul.rn.f32 %f0,%f0,%f7;"]
        lines += ["st.global.f32 [%d6],%f0;"]
    lines += ["done:","ret;","}"]
    return KernelSpec(name,op,shapes,out,attrs,grid,threads,"\n".join(lines)+"\n")


def _emit_norm_kernel(op,inputs,output,attrs=None):
    from .kernels import KernelSpec,_float,canonical,shape_of
    shapes=tuple(shape_of(x) for x in inputs);out=shape_of(output);attrs=dict(attrs or {})
    if not shapes or not shapes[0] or shapes[0][-1]>1024:
        raise ValueError("native normalization needs a last dimension of width 1..1024")
    width=shapes[0][-1];rows=math.prod(shapes[0][:-1]);chunks=(width+31)//32;tail=width%32
    if op.startswith("layernorm"):
        if set(attrs)-{"epsilon"}:raise ValueError("unsupported LayerNorm attributes")
        epsilon=attrs.get("epsilon",1e-5)
        _float(epsilon)
        if epsilon<2**-126:raise ValueError("LayerNorm epsilon must be positive normal FP32")
        if len(shapes)!=(2 if op.endswith("backward") else 1) or any(s!=out for s in shapes):
            raise ValueError("LayerNorm inputs/output must have identical shape")
    else:
        if attrs:raise ValueError("channel operators accept no attributes")
        if op=="channel_sum":valid=len(shapes)==1 and out==(width,)
        elif op=="channel_grad":valid=len(shapes)==2 and shapes[0]==shapes[1] and out==(width,)
        elif op=="channel_mul":valid=len(shapes)==2 and out==shapes[0] and shapes[1]==(width,)
        else:valid=len(shapes) in (2,3) and out==shapes[0] and all(s==(width,) for s in shapes[1:])
        if not valid:raise ValueError(f"{op} has invalid tensor shapes")
    name="oa_model_"+hashlib.sha256(canonical({"op":op,"inputs":shapes,"output":out,"attrs":attrs})).hexdigest()[:20]
    params=[f".param .u64 a{i}" for i in range(len(shapes))]+[".param .u64 output"]
    lines=[f".visible .entry {name}({','.join(params)}) {{",
           ".reg .b32 %r<16>; .reg .b64 %d<16>; .reg .f32 %f<150>; .reg .pred %p<4>;"]
    for i in range(len(shapes)):lines+=[f"ld.param.u64 %d{i},[a{i}];"]
    lines+=["ld.param.u64 %d3,[output]; mov.u32 %r0,%tid.x; mov.u32 %r1,%ctaid.x;"]
    if op in ("channel_sum","channel_grad"):
        grid,threads=((width+127)//128,1,1),128
        lines += [f"mad.lo.u32 %r2,%r1,128,%r0; setp.ge.u32 %p0,%r2,{width}; @%p0 bra done;",
                  "mov.u32 %r3,0; mov.f32 %f0,0f00000000;", "loop:",
                  f"mad.lo.u32 %r4,%r3,{width},%r2; mul.wide.u32 %d4,%r4,4; add.u64 %d5,%d0,%d4;",
                  "ld.global.f32 %f1,[%d5];"]
        if op=="channel_grad":lines += ["add.u64 %d6,%d1,%d4; ld.global.f32 %f2,[%d6]; mul.rn.f32 %f1,%f1,%f2;"]
        lines += ["add.rn.f32 %f0,%f0,%f1;",f"add.u32 %r3,%r3,1; setp.lt.u32 %p1,%r3,{rows}; @%p1 bra loop;",
                  "mul.wide.u32 %d4,%r2,4; add.u64 %d5,%d3,%d4; st.global.f32 [%d5],%f0;"]
    elif op in ("channel_affine","channel_mul"):
        if rows>65535:raise ValueError("channel rows exceed CUDA grid-y")
        grid,threads=((width+127)//128,rows,1),128
        lines += [f"mad.lo.u32 %r2,%r1,128,%r0; setp.ge.u32 %p0,%r2,{width}; @%p0 bra done;",
                  f"mov.u32 %r3,%ctaid.y; mad.lo.u32 %r4,%r3,{width},%r2;",
                  "mul.wide.u32 %d4,%r4,4; add.u64 %d5,%d0,%d4; add.u64 %d6,%d3,%d4;",
                  "mul.wide.u32 %d7,%r2,4; add.u64 %d8,%d1,%d7;",
                  "ld.global.f32 %f0,[%d5]; ld.global.f32 %f1,[%d8]; mul.rn.f32 %f0,%f0,%f1;"]
        if len(shapes)==3:lines += ["add.u64 %d9,%d2,%d7; ld.global.f32 %f2,[%d9]; add.rn.f32 %f0,%f0,%f2;"]
        lines += ["st.global.f32 [%d6],%f0;"]
    else:
        grid,threads=((rows+3)//4,1,1),128
        lines += ["and.b32 %r2,%r0,31; shr.u32 %r3,%r0,5; mad.lo.u32 %r4,%r1,4,%r3;",
                  f"setp.ge.u32 %p0,%r4,{rows}; @%p0 bra done;",
                  f"mad.lo.u32 %r5,%r4,{width},%r2; mul.wide.u32 %d4,%r5,4;",
                  "add.u64 %d5,%d0,%d4; add.u64 %d7,%d3,%d4;"]
        if op.endswith("backward"):lines += ["add.u64 %d6,%d1,%d4;"]
        # Shift by first x before calculating mean, protecting small variance
        # around large offsets. Padding contributes neutral zero to reductions.
        xbase="%d1" if op.endswith("backward") else "%d0"
        lines += [f"mul.lo.u32 %r6,%r4,{width}; mul.wide.u32 %d8,%r6,4; add.u64 %d9,{xbase},%d8; ld.global.f32 %f140,[%d9];"]
        if tail:lines += [f"setp.lt.u32 %p1,%r2,{tail};"]
        for j in range(chunks):
            guard="@%p1 " if tail and j==chunks-1 else ""
            if guard:lines += [f"mov.f32 %f{j},0f00000000;"]
            ptr="%d6" if op.endswith("backward") else "%d5"
            lines += [f"{guard}ld.global.f32 %f{j},[{ptr}+{128*j}]; sub.rn.f32 %f{j},%f{j},%f140;"]
            if guard:lines += [f"@!%p1 mov.f32 %f{j},0f00000000;"]
            if op.endswith("backward"):
                if guard:lines += [f"mov.f32 %f{64+j},0f00000000;"]
                lines += [f"{guard}ld.global.f32 %f{64+j},[%d5+{128*j}];"]
        def reduce(values,target):
            lines.append(f"mov.f32 %f{target},%f{values[0]};")
            for v in values[1:]:lines.append(f"add.rn.f32 %f{target},%f{target},%f{v};")
            for delta in (16,8,4,2,1):lines.extend([f"shfl.sync.bfly.b32 %f149,%f{target},{delta},31,0xffffffff;",f"add.rn.f32 %f{target},%f{target},%f149;"])
        reduce(list(range(chunks)),141)
        lines += [f"mul.rn.f32 %f141,%f141,{_float(1/width)};"]
        for j in range(chunks):
            lines += [f"sub.rn.f32 %f{j},%f{j},%f141;"]
            if tail and j==chunks-1:lines += [f"@!%p1 mov.f32 %f{j},0f00000000;"]
            lines += [f"mul.rn.f32 %f{32+j},%f{j},%f{j};"]
        reduce(list(range(32,32+chunks)),142)
        lines += [f"mul.rn.f32 %f142,%f142,{_float(1/width)}; add.rn.f32 %f142,%f142,{_float(epsilon)}; rsqrt.approx.ftz.f32 %f143,%f142;"]
        for j in range(chunks):lines += [f"mul.rn.f32 %f{j},%f{j},%f143;"]
        if op.endswith("backward"):
            for j in range(chunks):lines += [f"mul.rn.f32 %f{32+j},%f{j},%f{64+j};"]
            reduce(list(range(64,64+chunks)),144);reduce(list(range(32,32+chunks)),145)
            lines += [f"mul.rn.f32 %f144,%f144,{_float(1/width)}; mul.rn.f32 %f145,%f145,{_float(1/width)};"]
            for j in range(chunks):
                lines += [f"mul.rn.f32 %f{j},%f{j},%f145; add.rn.f32 %f{j},%f{j},%f144;",
                          f"sub.rn.f32 %f{j},%f{64+j},%f{j}; mul.rn.f32 %f{j},%f{j},%f143;"]
        for j in range(chunks):
            guard="@%p1 " if tail and j==chunks-1 else ""
            lines += [f"{guard}st.global.f32 [%d7+{128*j}],%f{j};"]
    lines += ["done:","ret;","}"]
    return KernelSpec(name,op,shapes,out,attrs,grid,threads,"\n".join(lines)+"\n")
