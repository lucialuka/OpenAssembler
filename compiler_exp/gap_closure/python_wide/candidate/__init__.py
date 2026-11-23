"""openassembler.pyfront: a from-scratch, minimal Python-subset compiler to PTX.

Not a general Python compiler -- a small, explicitly scoped numeric subset
(Numba @cuda.jit-lite), enough to write elementwise ops, reductions, softmax-style
kernels and tiled matmuls with Python syntax instead of raw PTX text:

    @kernel
    def add(a: Array(f32), b: Array(f32), out: Array(f32), n: i32):
        i = tid_x() + ctaid_x() * ntid_x()
        if i < n:
            out[i] = a[i] + b[i]

`compile_file(path)` compiles every top-level @kernel of a file into one PTX module
WITHOUT importing or executing the file; `compile_kernel(fn)` does one kernel (a live
function or a `find_kernel()` result) and produces the identical PTX. The PTX then goes
through the exact same openassembler.backend.ra / openassembler.container.cubin
pipeline as any other PTX in this repo.

Supported: typed params (Array(f32|i32|u32), i32/u32/f32 scalars), local int/float
variables, + - * / // % (Python semantics: / is true division to f32, // and % floor
on ints), comparisons, if/elif/else with and/or/not, for-range and while loops with
break/continue, float()/int(), min/max/abs, sqrt (exact) and approximate
exp/log/tanh/rsqrt (also as math.sqrt/exp/log/tanh), same-file pure typed scalar
@device helpers (scalar structured control flow and a terminal returned expression), shared_array + syncthreads, array
reads/writes, and tid_x/ctaid_x/ntid_x (and _y/_z). Anything else raises NotSupported
with the line. Every place GPU semantics differ from Python's is listed in the
openassembler.pyfront.compile module docstring.
"""

from __future__ import annotations

from openassembler.pyfront.compile import (KernelRef, NotSupported, compile_file, compile_kernel,
                                           device, find_kernel, kernel)
from openassembler.pyfront.intrinsics import (ctaid_x, ctaid_y, ctaid_z, exp, log, ntid_x, ntid_y,
                                              ntid_z, rsqrt, shared_array, sqrt, syncthreads, tanh,
                                              tid_x, tid_y, tid_z)
from openassembler.pyfront.types import Array, f32, i32, i64, u32, u64

__all__ = ["Array", "f32", "i32", "u32", "i64", "u64", "NotSupported", "KernelRef", "compile_kernel",
           "compile_file", "find_kernel", "kernel", "device", "tid_x", "tid_y", "tid_z", "ctaid_x",
           "ctaid_y", "ctaid_z", "ntid_x", "ntid_y", "ntid_z", "sqrt", "rsqrt", "exp", "log",
           "tanh", "shared_array", "syncthreads"]
