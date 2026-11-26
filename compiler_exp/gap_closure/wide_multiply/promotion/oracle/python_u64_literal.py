from openassembler.pyfront import *
@kernel
def python_u64_literal(out:Array(u64),a:Array(u64),n:i32,choice:u64):
    i = ctaid_x()*128 + tid_x()
    if i < n:
        v = a[i*2] * u64(-17)
        out[i] = v
