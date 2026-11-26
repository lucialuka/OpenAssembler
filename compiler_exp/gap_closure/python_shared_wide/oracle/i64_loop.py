from openassembler.pyfront import *
@kernel
def i64_loop(a: Array(i64), out: Array(i64), delta: i64, n: i32):
    shared = shared_array(i64, 128)
    t = tid_x()
    i = ctaid_x() * 128 + t
    shared[t] = a[i]
    syncthreads()
    v = a[i]
    for step in range(3):
        v = shared[t ^ 64] + delta
        syncthreads()
        shared[t] = v
        syncthreads()
    if i < n:
        out[i * 4] = v
