from openassembler.pyfront import Array, i32, i64, u64, kernel, tid_x

@kernel
def verified_signed(a: Array(i64), out: Array(i64), delta: i64, n: i32):
    i = tid_x()
    if i < n:
        out[i] = a[i] + delta

@kernel
def verified_unsigned(a: Array(u64), out: Array(u64), delta: u64, n: i32):
    i = tid_x()
    if i < n:
        out[i] = a[i] + delta
