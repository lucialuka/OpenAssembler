from openassembler.pyfront import *
raise RuntimeError('Boolean comparison discovery must never execute')

@device
def compare_helper(a: bool, b: bool) -> bool:
    return a == b

@kernel
def scalar_equal(a: bool, b: bool, out: Array(bool)):
    out[tid_x()] = a == b

@kernel
def storage_order(a: Array(bool), b: Array(bool), out: Array(bool), n: i32):
    i = tid_x()
    if i < n:
        out[i] = a[i] < b[i]

@kernel
def helper_equal(a: bool, b: bool, out: Array(bool)):
    out[tid_x()] = compare_helper(a, b)

@kernel
def chained_equal(a: Array(bool), b: bool, out: Array(bool), counter: Array(i32), n: i32):
    i = tid_x()
    if i < n:
        out[i] = a[i] == b == bool(atomic_add(counter, i, 1))
