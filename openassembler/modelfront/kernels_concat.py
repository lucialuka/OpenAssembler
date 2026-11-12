"""Exact-shape contiguous concatenation and slice derivatives for cache buckets."""
from __future__ import annotations

import hashlib
import math

CONCAT_OPS = frozenset({"concat", "slice_axis"})


def emit_concat_kernel(op, inputs, output, attrs=None):
    from .kernels import KernelSpec, canonical, shape_of

    shapes, out, attrs = tuple(shape_of(x) for x in inputs), shape_of(output), dict(attrs or {})
    rank, axis = len(out), attrs.get("axis")
    if (op not in CONCAT_OPS or not 2 <= rank <= 4 or type(axis) is not int or not 0 <= axis < rank
            or any(len(x) != rank for x in shapes)):
        raise ValueError("concat/slice requires rank-two through rank-four tensors and a valid axis")
    if op == "concat":
        if len(shapes) < 2 or set(attrs) != {"axis"}:
            raise ValueError("concat needs at least two tensors and an axis")
        expected = list(shapes[0]); expected[axis] = sum(s[axis] for s in shapes)
        if tuple(expected) != out or any(any(s[i] != out[i] for i in range(rank) if i != axis) for s in shapes):
            raise ValueError("concatenation shapes disagree")
    else:
        start = attrs.get("start")
        if (len(shapes) != 1 or set(attrs) != {"axis", "start"} or type(start) is not int
                or start < 0 or start + out[axis] > shapes[0][axis]
                or any(shapes[0][i] != out[i] for i in range(rank) if i != axis)):
            raise ValueError("slice axis bounds/shapes disagree")
    outer, inner = math.prod(out[:axis]), math.prod(out[axis+1:])
    width = out[axis] * inner
    if outer > 65535:
        raise ValueError("concatenation outer groups exceed grid-y bounds")
    signature = {"op": op, "inputs": shapes, "output": out, "attrs": attrs}
    name = "oa_model_" + hashlib.sha256(canonical(signature)).hexdigest()[:20]
    params = [f".param .u64 a{i}" for i in range(len(shapes))] + [".param .u64 output"]
    lines = [f".visible .entry {name}({','.join(params)}) {{",
             f".reg .b32 %r<12>; .reg .b64 %d<{len(shapes)+5}>; .reg .f32 %f; .reg .pred %p<2>;"]
    for i in range(len(shapes)):
        lines.append(f"ld.param.u64 %d{i},[a{i}];")
    output_reg = len(shapes)
    offset_reg, address_reg = output_reg+1, output_reg+2
    lines += [f"ld.param.u64 %d{output_reg},[output];",
              "mov.u32 %r0,%tid.x; mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ctaid.y;",
              "mad.lo.u32 %r3,%r1,128,%r0;",
              f"setp.ge.u32 %p0,%r3,{width}; @%p0 bra done;"]
    if op == "concat":
        begin = 0
        for i, shape in enumerate(shapes):
            span = shape[axis] * inner
            if i < len(shapes)-1:
                lines += [f"setp.lt.u32 %p1,%r3,{begin+span}; @!%p1 bra next_{i};"]
            lines += [f"sub.u32 %r4,%r3,{begin}; mad.lo.u32 %r5,%r2,{span},%r4;",
                      f"mul.wide.u32 %d{offset_reg},%r5,4; add.u64 %d{address_reg},%d{i},%d{offset_reg};",
                      f"ld.global.f32 %f,[%d{address_reg}]; bra store;", f"next_{i}:"]
            begin += span
    else:
        lines += [f"add.u32 %r4,%r3,{attrs['start']*inner};",
                  f"mad.lo.u32 %r5,%r2,{shapes[0][axis]*inner},%r4;",
                  f"mul.wide.u32 %d{offset_reg},%r5,4; add.u64 %d{address_reg},%d0,%d{offset_reg};",
                  f"ld.global.f32 %f,[%d{address_reg}];"]
    lines += ["store:", f"mad.lo.u32 %r6,%r2,{width},%r3;",
              f"mul.wide.u32 %d{offset_reg},%r6,4; add.u64 %d{address_reg},%d{output_reg},%d{offset_reg};",
              f"st.global.f32 [%d{address_reg}],%f;", "done:", "ret;", "}", ""]
    return KernelSpec(name, op, shapes, out, attrs, ((width+127)//128, outer, 1), 128, "\n".join(lines))
