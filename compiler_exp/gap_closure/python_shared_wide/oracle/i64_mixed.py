from openassembler.pyfront import *
@kernel
def i64_mixed(a: Array(i64), out: Array(i64), delta: i64, n: i32):
    prefix = shared_array(i32, 1)
    shared = shared_array(i64, 128)
    suffix = shared_array(i32, 3)
    t = tid_x()
    i = ctaid_x() * 128 + t
    if t == 0:
        prefix[0] = 0x12345678
    if t == 0:
        suffix[0] = 0x76543210
        suffix[1] = -1
        suffix[2] = -2147483648
    shared[t] = a[i]
    syncthreads()
    v = shared[t ^ 64]
    if i < n:
        out[i * 4] = v
        out[i * 4 + 1] = i64(prefix[0])
        out[i * 4 + 2] = i64(suffix[0])
        out[i * 4 + 3] = i64(suffix[2])
