"""GPU intrinsics importable from openassembler.pyfront. They exist so `from
openassembler.pyfront import tid_x, sqrt, shared_array, syncthreads` works and linters see the
names; inside a @kernel the compiler recognizes them statically (it never calls these). Called
as ordinary Python they raise, the same way a @kernel function does -- there is no GPU here to
answer tid_x(). The bare names (tid_x() without an import) keep working inside kernels too.
"""

from __future__ import annotations

from openassembler.pyfront.compile import NotSupported


def _stub(name: str, doc: str):
    def f(*_a, **_kw):
        raise NotSupported(f"{name}() is a GPU intrinsic, only valid inside a @kernel function "
                           "compiled by openassembler.pyfront")

    f.__name__ = f.__qualname__ = name
    f.__doc__ = doc
    return f


tid_x = _stub("tid_x", "Thread index in the block, x (%tid.x).")
tid_y = _stub("tid_y", "Thread index in the block, y (%tid.y).")
tid_z = _stub("tid_z", "Thread index in the block, z (%tid.z).")
ctaid_x = _stub("ctaid_x", "Block index in the grid, x (%ctaid.x).")
ctaid_y = _stub("ctaid_y", "Block index in the grid, y (%ctaid.y).")
ctaid_z = _stub("ctaid_z", "Block index in the grid, z (%ctaid.z).")
ntid_x = _stub("ntid_x", "Threads per block, x (%ntid.x).")
ntid_y = _stub("ntid_y", "Threads per block, y (%ntid.y).")
ntid_z = _stub("ntid_z", "Threads per block, z (%ntid.z).")
sqrt = _stub("sqrt", "Correctly rounded f32 square root (sqrt.rn.f32); NaN for x < 0.")
rsqrt = _stub("rsqrt", "Approximate 1/sqrt(x) (rsqrt.approx.f32).")
exp = _stub("exp", "Approximate e**x: ex2.approx.f32(x * log2(e)), like CUDA's __expf.")
log = _stub("log", "Approximate ln(x): lg2.approx.f32(x) * ln(2), like CUDA's __logf.")
tanh = _stub("tanh", "Approximate tanh(x) (tanh.approx.f32, sm_75+).")
shared_array = _stub("shared_array", "s = shared_array(f32, N) at kernel top level: a .shared array.")
syncthreads = _stub("syncthreads", "Block-wide barrier (bar.sync 0), like CUDA's __syncthreads().")

__all__ = ["tid_x", "tid_y", "tid_z", "ctaid_x", "ctaid_y", "ctaid_z", "ntid_x", "ntid_y",
           "ntid_z", "sqrt", "rsqrt", "exp", "log", "tanh", "shared_array", "syncthreads"]
