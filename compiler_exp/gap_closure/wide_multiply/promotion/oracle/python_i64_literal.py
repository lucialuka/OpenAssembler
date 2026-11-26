from openassembler.pyfront import *
@kernel
def python_i64_literal(out:Array(i64),a:Array(i64),n:i32,choice:i64):
    i = ctaid_x()*128 + tid_x()
    if i < n:
        v = a[i*2] * i64(-17)
        out[i] = v
