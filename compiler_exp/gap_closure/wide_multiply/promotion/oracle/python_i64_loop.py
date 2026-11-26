from openassembler.pyfront import *
@kernel
def python_i64_loop(out:Array(i64),a:Array(i64),n:i32,choice:i64):
    i = ctaid_x()*128 + tid_x()
    if i < n:
        v = a[i*2]
        for step in range(3):
            v *= a[i*2+1]
        out[i] = v
