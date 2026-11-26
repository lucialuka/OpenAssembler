from openassembler.pyfront import *
@kernel
def i64_padded_maximum(a: Array(i64), out: Array(i64), delta: i64, n: i32):
    prefix = shared_array(i32, 1)
    shared = shared_array(i64, 6143)
    t = tid_x()
    i = ctaid_x() * 128 + t
    if t == 0:
        prefix[0] = 0x12345678
    shared[t + 6015] = a[i]
    syncthreads()
    v = shared[(t ^ 64) + 6015]
    if i < n:
        out[i * 4] = v
        out[i * 4 + 1] = i64(prefix[0])
