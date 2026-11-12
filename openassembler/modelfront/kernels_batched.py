"""Explicit FP32 batched products and contiguous permutation baselines.

These kernels establish native graph coverage, not tuned attention performance.
Batch/row coordinates are launch dimensions; any rank-four permutation only
uses independently validated constant integer division to decode its batch.
"""
from __future__ import annotations

import hashlib
import math

BATCHED_OPS = frozenset({"bmm", "bmm_dx", "bmm_dw", "permute"})


def emit_batched_kernel(op, inputs, output, attrs=None):
    from .kernels import KernelSpec, canonical, shape_of

    shapes, out, attrs = tuple(shape_of(x) for x in inputs), shape_of(output), dict(attrs or {})
    if op == "permute":
        if len(shapes) != 1 or not 2 <= len(out) <= 4 or len(shapes[0]) != len(out):
            raise ValueError("native permutation requires one rank-two to rank-four tensor")
        axes = attrs.get("axes")
        if (set(attrs) != {"axes"} or not isinstance(axes, (list, tuple))
                or any(type(x) is not int for x in axes) or sorted(axes) != list(range(len(out)))
                or tuple(shapes[0][i] for i in axes) != out):
            raise ValueError("permutation axes must exactly match the output shape")
        attrs = {"axes": list(axes)}
    elif op in ("bmm", "bmm_dx", "bmm_dw"):
        if attrs or len(shapes) != 2 or any(len(s) != 3 for s in (*shapes, out)):
            raise ValueError("native batched products require two rank-three matrices and no attributes")
        if op == "bmm":
            batch, m, k = shapes[0]; br, kb, n = shapes[1]
            expected, length = (batch, m, n), k
            valid = br == batch and kb == k
        elif op == "bmm_dx":
            batch, m, n = shapes[0]; br, k, nb = shapes[1]
            expected, length = (batch, m, k), n
            valid = br == batch and nb == n
        else:
            batch, m, n = shapes[0]; br, mb, k = shapes[1]
            expected, length = (batch, k, n), m
            valid = br == batch and mb == m
        if not valid or out != expected:
            raise ValueError("batched product shapes disagree")
    else:
        raise ValueError("unsupported batched native operator")
    rows, cols, batches = out[-2], out[-1], math.prod(out[:-2])
    if rows > 65535 or batches > 65535:
        raise ValueError("batched operation exceeds grid-y/grid-z launch bounds")
    signature = {"op": op, "inputs": shapes, "output": out, "attrs": attrs}
    name = "oa_model_" + hashlib.sha256(canonical(signature)).hexdigest()[:20]
    params = [f".param .u64 a{i}" for i in range(len(shapes))] + [".param .u64 output"]
    lines = [f".visible .entry {name}({','.join(params)}) {{",
             ".reg .b32 %r<28>; .reg .b64 %d<10>; .reg .f32 %f<5>; .reg .pred %p<3>;"]
    for i in range(len(shapes)):
        lines.append(f"ld.param.u64 %d{i},[a{i}];")
    lines += ["ld.param.u64 %d2,[output]; mov.u32 %r0,%tid.x; mov.u32 %r1,%ctaid.x;",
              "mad.lo.u32 %r2,%r1,128,%r0; mov.u32 %r3,%ctaid.y; mov.u32 %r4,%ctaid.z;",
              f"setp.ge.u32 %p0,%r2,{cols}; @%p0 bra done;",
              f"mad.lo.u32 %r5,%r4,{rows},%r3; mad.lo.u32 %r5,%r5,{cols},%r2;"]
    if op == "permute":
        rank = len(out)
        if rank == 2:
            coords = ["%r3", "%r2"]
        elif rank == 3:
            coords = ["%r4", "%r3", "%r2"]
        else:
            lines += [f"div.u32 %r10,%r4,{out[1]}; rem.u32 %r11,%r4,{out[1]};"]
            coords = ["%r10", "%r11", "%r3", "%r2"]
        strides = [math.prod(shapes[0][i+1:]) for i in range(rank)]
        lines.append("mov.u32 %r6,0;")
        for output_axis, input_axis in enumerate(attrs["axes"]):
            lines.append(f"mad.lo.u32 %r6,{coords[output_axis]},{strides[input_axis]},%r6;")
        lines += ["mul.wide.u32 %d3,%r6,4; add.u64 %d4,%d0,%d3; ld.global.f32 %f0,[%d4];"]
    else:
        lines += ["mov.u32 %r6,0; mov.f32 %f0,0f00000000;", "loop:"]
        if op == "bmm":
            lines += [f"mad.lo.u32 %r7,%r4,{m},%r3; mad.lo.u32 %r7,%r7,{k},%r6;",
                      f"mad.lo.u32 %r8,%r4,{k},%r6; mad.lo.u32 %r8,%r8,{n},%r2;"]
        elif op == "bmm_dx":
            lines += [f"mad.lo.u32 %r7,%r4,{m},%r3; mad.lo.u32 %r7,%r7,{n},%r6;",
                      f"mad.lo.u32 %r8,%r4,{k},%r2; mad.lo.u32 %r8,%r8,{n},%r6;"]
        else:
            lines += [f"mad.lo.u32 %r7,%r4,{m},%r6; mad.lo.u32 %r7,%r7,{n},%r2;",
                      f"mad.lo.u32 %r8,%r4,{m},%r6; mad.lo.u32 %r8,%r8,{k},%r3;"]
        lines += ["mul.wide.u32 %d3,%r7,4; add.u64 %d4,%d0,%d3; ld.global.f32 %f1,[%d4];",
                  "mul.wide.u32 %d3,%r8,4; add.u64 %d4,%d1,%d3; ld.global.f32 %f2,[%d4];",
                  "mul.rn.f32 %f3,%f1,%f2; add.rn.f32 %f0,%f0,%f3; add.u32 %r6,%r6,1;",
                  f"setp.lt.u32 %p1,%r6,{length}; @%p1 bra loop;"]
    lines += ["mul.wide.u32 %d3,%r5,4; add.u64 %d4,%d2,%d3; st.global.f32 [%d4],%f0;",
              "done:", "ret;", "}", ""]
    return KernelSpec(name, op, shapes, out, attrs, ((cols+127)//128, rows, batches), 128, "\n".join(lines))
