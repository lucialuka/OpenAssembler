"""Direct FP32 NCHW convolution kernels for exact-shape model graphs."""
from __future__ import annotations

import hashlib
import math

CONV_OPS = frozenset({"conv2d", "conv2d_dx", "conv2d_dw"})


def emit_conv_kernel(op, inputs, output, attrs=None):
    from .kernels import KernelSpec, canonical, shape_of

    shapes = tuple(shape_of(spec) for spec in inputs)
    out = shape_of(output)
    attrs = dict(attrs or {})
    if op not in CONV_OPS or set(attrs) != {"stride", "padding", "dilation"}:
        raise ValueError("convolution requires a supported operator and static attributes")
    if any(not isinstance(attrs[key], (tuple, list)) for key in ("stride", "padding", "dilation")):
        raise ValueError("convolution attributes must be integer pairs")
    stride, padding, dilation = (tuple(attrs[key]) for key in ("stride", "padding", "dilation"))
    pairs = (stride, padding, dilation)
    if any(len(pair) != 2 or any(type(value) is not int for value in pair) for pair in pairs):
        raise ValueError("convolution attributes must be integer pairs")
    if any(value <= 0 for value in stride + dilation) or any(value < 0 for value in padding):
        raise ValueError("convolution stride and dilation must be positive and padding nonnegative")

    if op == "conv2d":
        if len(shapes) not in (2, 3) or len(shapes[0]) != 4 or len(shapes[1]) != 4 or len(out) != 4:
            raise ValueError("conv2d requires NCHW input and OIHW weights")
        batch, channels, height, width = shapes[0]
        output_channels, weight_channels, kh, kw = shapes[1]
        if channels != weight_channels:
            raise ValueError("conv2d input and weight channel counts disagree")
        oh = (height + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
        ow = (width + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
        if min(oh, ow) <= 0 or out != (batch, output_channels, oh, ow):
            raise ValueError("conv2d output shape disagrees with its static parameters")
        if len(shapes) == 3 and shapes[2] != (output_channels,):
            raise ValueError("conv2d bias must match the output-channel count")
    elif op == "conv2d_dx":
        if len(shapes) != 2 or len(shapes[0]) != 4 or len(shapes[1]) != 4 or len(out) != 4:
            raise ValueError("conv2d_dx requires rank-four gradient and weight tensors")
        batch, output_channels, oh, ow = shapes[0]
        weight_outputs, channels, kh, kw = shapes[1]
        height, width = out[-2:]
        if weight_outputs != output_channels or out[:2] != (batch, channels):
            raise ValueError("conv2d_dx tensor channels disagree")
        expected = ((height + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1,
                    (width + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1)
        if (oh, ow) != expected:
            raise ValueError("conv2d_dx spatial shapes disagree")
    else:
        if len(shapes) != 2 or len(shapes[0]) != 4 or len(shapes[1]) != 4 or len(out) != 4:
            raise ValueError("conv2d_dw requires rank-four gradient and input tensors")
        batch, output_channels, oh, ow = shapes[0]
        input_batch, channels, height, width = shapes[1]
        weight_outputs, weight_channels, kh, kw = out
        expected = ((height + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1,
                    (width + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1)
        if (batch, output_channels, oh, ow) != (input_batch, weight_outputs, *expected) or channels != weight_channels:
            raise ValueError("conv2d_dw tensor shapes disagree")

    # Spatial arithmetic uses signed int32 coordinates before the bounds checks.
    # Bound intermediate products too, including coordinates in padded regions.
    for length, kernel_size, output_length, step, pad, spacing in zip(
            (height, width), (kh, kw), (oh, ow), stride, padding, dilation):
        if max(length + pad, (output_length - 1) * step + (kernel_size - 1) * spacing, pad) >= 2**31:
            raise ValueError("convolution spatial arithmetic exceeds the int32 coordinate domain")

    signature = {"op": op, "inputs": shapes, "output": out, "attrs": attrs}
    name = "oa_model_" + hashlib.sha256(canonical(signature)).hexdigest()[:20]
    params = [f".param .u64 a{index}" for index in range(len(shapes))] + [".param .u64 output"]
    lines = [f".visible .entry {name}({','.join(params)}) {{",
             ".reg .b32 %r<32>; .reg .b64 %d<12>; .reg .f32 %f<8>; .reg .pred %p<8>;"]
    for index in range(len(shapes)):
        lines.append(f"ld.param.u64 %d{index},[a{index}];")
    output_pointer = len(shapes)
    lines += [f"ld.param.u64 %d{output_pointer},[output];",
              "mov.u32 %r0,%tid.x; mov.u32 %r1,%ctaid.x; mad.lo.u32 %r2,%r1,128,%r0;"]

    if op == "conv2d":
        batch, channels, height, width = shapes[0]
        output_channels, _, kh, kw = shapes[1]
        oh, ow = out[-2:]
        count = math.prod(out)
        lines += [f"setp.ge.u32 %p0,%r2,{count}; @%p0 bra done;",
                  f"div.u32 %r3,%r2,{ow}; rem.u32 %r4,%r2,{ow};",
                  f"div.u32 %r5,%r3,{oh}; rem.u32 %r6,%r3,{oh};",
                  f"div.u32 %r7,%r5,{output_channels}; rem.u32 %r8,%r5,{output_channels};",
                  "mov.f32 %f0,0f00000000;"]
        if len(shapes) == 3:
            lines += ["mul.wide.u32 %d4,%r8,4; add.u64 %d5,%d2,%d4; ld.global.f32 %f0,[%d5];"]
        loop = channels * kh * kw
        lines += ["mov.u32 %r9,0;", "conv_loop:",
                  f"div.u32 %r10,%r9,{kw}; rem.u32 %r11,%r9,{kw};",
                  f"div.u32 %r12,%r10,{kh}; rem.u32 %r13,%r10,{kh};",
                  f"mul.lo.u32 %r14,%r13,{dilation[0]}; mad.lo.u32 %r14,%r6,{stride[0]},%r14; sub.s32 %r14,%r14,{padding[0]};",
                  f"mul.lo.u32 %r15,%r11,{dilation[1]}; mad.lo.u32 %r15,%r4,{stride[1]},%r15; sub.s32 %r15,%r15,{padding[1]};",
                  f"setp.ge.s32 %p1,%r14,0; setp.lt.s32 %p2,%r14,{height}; @!%p1 bra conv_skip; @!%p2 bra conv_skip;",
                  f"setp.ge.s32 %p1,%r15,0; setp.lt.s32 %p2,%r15,{width}; @!%p1 bra conv_skip; @!%p2 bra conv_skip;",
                  f"mad.lo.u32 %r16,%r8,{channels},%r12; mad.lo.u32 %r16,%r16,{kh},%r13; mad.lo.u32 %r16,%r16,{kw},%r11;",
                  f"mad.lo.u32 %r17,%r7,{channels},%r12; mad.lo.u32 %r17,%r17,{height},%r14; mad.lo.u32 %r17,%r17,{width},%r15;",
                  "mul.wide.u32 %d6,%r16,4; add.u64 %d7,%d1,%d6; ld.global.f32 %f1,[%d7];",
                  "mul.wide.u32 %d6,%r17,4; add.u64 %d7,%d0,%d6; ld.global.f32 %f2,[%d7];",
                  "mul.rn.f32 %f3,%f1,%f2; add.rn.f32 %f0,%f0,%f3;", "conv_skip:",
                  "add.u32 %r9,%r9,1;", f"setp.lt.u32 %p1,%r9,{loop}; @%p1 bra conv_loop;",
                  f"mul.wide.u32 %d6,%r2,4; add.u64 %d7,%d{output_pointer},%d6; st.global.f32 [%d7],%f0;", "bra done;"]
    elif op == "conv2d_dx":
        batch, channels, height, width = out
        _, output_channels, oh, ow = shapes[0]
        kh, kw = shapes[1][-2:]
        count = math.prod(out)
        lines += [f"setp.ge.u32 %p0,%r2,{count}; @%p0 bra done;",
                  f"div.u32 %r3,%r2,{channels * height * width};",
                  f"rem.u32 %r4,%r2,{channels * height * width}; div.u32 %r5,%r4,{height * width};",
                  f"rem.u32 %r6,%r4,{height * width}; div.u32 %r7,%r6,{width}; rem.u32 %r8,%r6,{width};",
                  "mov.u32 %r9,0; mov.f32 %f0,0f00000000;", "dx_loop:",
                  f"div.u32 %r10,%r9,{kh * kw}; rem.u32 %r11,%r9,{kh * kw};",
                  f"div.u32 %r12,%r11,{kw}; rem.u32 %r13,%r11,{kw};",
                  f"mul.lo.u32 %r14,%r12,{dilation[0]}; sub.s32 %r14,%r7,%r14; add.s32 %r14,%r14,{padding[0]};",
                  f"mul.lo.u32 %r15,%r13,{dilation[1]}; sub.s32 %r15,%r8,%r15; add.s32 %r15,%r15,{padding[1]};",
                  "setp.lt.s32 %p1,%r14,0; @%p1 bra dx_skip; setp.lt.s32 %p1,%r15,0; @%p1 bra dx_skip;",
                  f"rem.u32 %r16,%r14,{stride[0]}; setp.ne.u32 %p1,%r16,0; @%p1 bra dx_skip;",
                  f"rem.u32 %r16,%r15,{stride[1]}; setp.ne.u32 %p1,%r16,0; @%p1 bra dx_skip;",
                  f"div.u32 %r17,%r14,{stride[0]}; div.u32 %r18,%r15,{stride[1]};",
                  f"setp.ge.u32 %p1,%r17,{oh}; @%p1 bra dx_skip; setp.ge.u32 %p1,%r18,{ow}; @%p1 bra dx_skip;",
                  f"mad.lo.u32 %r19,%r10,{channels},%r5; mad.lo.u32 %r19,%r19,{kh},%r12; mad.lo.u32 %r19,%r19,{kw},%r13;",
                  f"mad.lo.u32 %r20,%r3,{output_channels},%r10; mad.lo.u32 %r20,%r20,{oh},%r17; mad.lo.u32 %r20,%r20,{ow},%r18;",
                  "mul.wide.u32 %d6,%r19,4; add.u64 %d7,%d1,%d6; ld.global.f32 %f1,[%d7];",
                  "mul.wide.u32 %d6,%r20,4; add.u64 %d7,%d0,%d6; ld.global.f32 %f2,[%d7];",
                  "mul.rn.f32 %f3,%f1,%f2; add.rn.f32 %f0,%f0,%f3;", "dx_skip:",
                  "add.u32 %r9,%r9,1;", f"setp.lt.u32 %p1,%r9,{output_channels * kh * kw}; @%p1 bra dx_loop;",
                  f"mul.wide.u32 %d6,%r2,4; add.u64 %d7,%d{output_pointer},%d6; st.global.f32 [%d7],%f0;", "bra done;"]
    else:
        _, channels, height, width = shapes[1]
        batch, output_channels, oh, ow = shapes[0]
        _, _, kh, kw = out
        count = math.prod(out)
        lines += [f"setp.ge.u32 %p0,%r2,{count}; @%p0 bra done;",
                  f"div.u32 %r3,%r2,{channels * kh * kw};",
                  f"rem.u32 %r4,%r2,{channels * kh * kw}; div.u32 %r5,%r4,{kh * kw};",
                  f"rem.u32 %r6,%r4,{kh * kw}; div.u32 %r7,%r6,{kw}; rem.u32 %r8,%r6,{kw};",
                  "mov.u32 %r9,0; mov.f32 %f0,0f00000000;", "dw_loop:",
                  f"div.u32 %r10,%r9,{oh * ow}; rem.u32 %r11,%r9,{oh * ow};",
                  f"div.u32 %r12,%r11,{ow}; rem.u32 %r13,%r11,{ow};",
                  f"mul.lo.u32 %r14,%r12,{stride[0]}; mul.lo.u32 %r15,%r7,{dilation[0]}; add.u32 %r14,%r14,%r15; sub.s32 %r14,%r14,{padding[0]};",
                  f"mul.lo.u32 %r15,%r13,{stride[1]}; mul.lo.u32 %r16,%r8,{dilation[1]}; add.u32 %r15,%r15,%r16; sub.s32 %r15,%r15,{padding[1]};",
                  f"setp.ge.s32 %p1,%r14,0; @!%p1 bra dw_skip; setp.lt.s32 %p1,%r14,{height}; @!%p1 bra dw_skip;",
                  f"setp.ge.s32 %p1,%r15,0; @!%p1 bra dw_skip; setp.lt.s32 %p1,%r15,{width}; @!%p1 bra dw_skip;",
                  f"mad.lo.u32 %r17,%r10,{channels},%r5; mad.lo.u32 %r17,%r17,{height},%r14; mad.lo.u32 %r17,%r17,{width},%r15;",
                  f"mad.lo.u32 %r18,%r10,{output_channels},%r3; mad.lo.u32 %r18,%r18,{oh},%r12; mad.lo.u32 %r18,%r18,{ow},%r13;",
                  "mul.wide.u32 %d6,%r17,4; add.u64 %d7,%d1,%d6; ld.global.f32 %f1,[%d7];",
                  "mul.wide.u32 %d6,%r18,4; add.u64 %d7,%d0,%d6; ld.global.f32 %f2,[%d7];",
                  "mul.rn.f32 %f3,%f1,%f2; add.rn.f32 %f0,%f0,%f3;", "dw_skip:",
                  "add.u32 %r9,%r9,1;", f"setp.lt.u32 %p1,%r9,{batch * oh * ow}; @%p1 bra dw_loop;",
                  f"mul.wide.u32 %d6,%r2,4; add.u64 %d7,%d{output_pointer},%d6; st.global.f32 [%d7],%f0;", "bra done;"]
    lines += ["done:", "ret;", "}", ""]
    count = math.prod(out)
    return KernelSpec(name, op, shapes, out, attrs, ((count + 127) // 128, 1, 1), 128, "\n".join(lines))
