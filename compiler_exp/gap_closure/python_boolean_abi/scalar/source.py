from openassembler.pyfront import *
raise RuntimeError('Boolean scalar ABI source must be discovered without execution')


@device
def identity(value: bool) -> bool:
    return value


@kernel
def scalar_truth(out: Array(i32), first: bool, second: bool, n: i32):
    i = tid_x()
    if i < n:
        out[i] = i32(first and not second)


@kernel
def scalar_helper(out: Array(i32), flag: bool, neighbor: i32):
    i = tid_x()
    local = identity(flag)
    flag = not local
    out[i] = i32(flag) + neighbor


@kernel
def scalar_guard(out: Array(i64), left: bool, value: i64, right: bool, n: i32):
    i = tid_x()
    if i < n:
        flag = left or right
        out[i] = i64(flag) + value
