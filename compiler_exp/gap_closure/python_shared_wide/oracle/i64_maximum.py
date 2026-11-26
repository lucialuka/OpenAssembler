from openassembler.pyfront import *
@kernel
def i64_maximum(a: Array(i64), out: Array(i64), delta: i64, n: i32):
    shared = shared_array(i64, 6144)
    t = tid_x()
    i = ctaid_x() * 128 + t
    shared[t + 6016] = a[i]
    syncthreads()
    v = shared[(t ^ 64) + 6016]
    if i < n:
        out[i * 4] = v
