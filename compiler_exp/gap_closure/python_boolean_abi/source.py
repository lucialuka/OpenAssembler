from openassembler.pyfront import *
raise RuntimeError('Boolean ABI discovery must never execute user code')

@device
def invert(value: bool) -> bool:
    return not value

@kernel
def scalar_flag(out: Array(i32), flag: bool):
    out[0] = i32(flag)

@kernel
def boolean_arrays(a: Array(bool), out: Array(bool)):
    i = tid_x()
    out[i] = not a[i]

@kernel
def boolean_helper(a: Array(i32), out: Array(i32)):
    i = tid_x()
    out[i] = i32(invert(bool(a[i])))
