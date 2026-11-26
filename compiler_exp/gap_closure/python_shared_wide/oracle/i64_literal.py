from openassembler.pyfront import *
@kernel
def i64_literal(a: Array(i64), out: Array(i64), delta: i64, n: i32):
    shared = shared_array(i64, 128)
    t = tid_x()
    i = ctaid_x() * 128 + t
    if t == 127:
        shared[127] = a[i]
    syncthreads()
    v = shared[127]
    if i < n:
        out[i * 4] = v
