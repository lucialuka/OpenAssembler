from openassembler.pyfront import *
@kernel
def u64_maximum(a: Array(u64), out: Array(u64), delta: u64, n: i32):
    shared = shared_array(u64, 6144)
    t = tid_x()
    i = ctaid_x() * 128 + t
    shared[t + 6016] = a[i]
    syncthreads()
    v = shared[(t ^ 64) + 6016]
    if i < n:
        out[i * 4] = v
