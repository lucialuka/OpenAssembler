from openassembler.pyfront import *
@kernel
def python_i64_square_register(out:Array(i64),a:Array(i64),n:i32,choice:i64):
    i = ctaid_x()*128 + tid_x()
    if i < n:
        x = a[i*2]
        out[i] = x*x
