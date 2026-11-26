from openassembler.pyfront import *
@kernel
def u64_padded_maximum(a: Array(u64), out: Array(u64), delta: u64, n: i32):
    prefix = shared_array(i32, 1)
    shared = shared_array(u64, 6143)
    t = tid_x()
    i = ctaid_x() * 128 + t
    if t == 0:
        prefix[0] = 0x12345678
    shared[t + 6015] = a[i]
    syncthreads()
    v = shared[(t ^ 64) + 6015]
    if i < n:
        out[i * 4] = v
        out[i * 4 + 1] = u64(prefix[0])
