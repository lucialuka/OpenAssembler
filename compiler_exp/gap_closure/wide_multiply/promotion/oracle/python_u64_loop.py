from openassembler.pyfront import *
@kernel
def python_u64_loop(out:Array(u64),a:Array(u64),n:i32,choice:u64):
    i = ctaid_x()*128 + tid_x()
    if i < n:
        v = a[i*2]
        for step in range(3):
            v *= a[i*2+1]
        out[i] = v
