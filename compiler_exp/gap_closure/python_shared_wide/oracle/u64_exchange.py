from openassembler.pyfront import *
@kernel
def u64_exchange(a: Array(u64), out: Array(u64), delta: u64, n: i32):
    shared = shared_array(u64, 128)
    t = tid_x()
    i = ctaid_x() * 128 + t
    shared[t] = a[i]
    syncthreads()
    v = shared[t ^ 64]
    if i < n:
        out[i * 4] = v
