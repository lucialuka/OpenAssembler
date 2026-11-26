from openassembler.pyfront import *
@kernel
def u64_literal(a: Array(u64), out: Array(u64), delta: u64, n: i32):
    shared = shared_array(u64, 128)
    t = tid_x()
    i = ctaid_x() * 128 + t
    if t == 127:
        shared[127] = a[i]
    syncthreads()
    v = shared[127]
    if i < n:
        out[i * 4] = v
