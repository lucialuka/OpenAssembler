from openassembler.pyfront import *
@kernel
def python_u64_guarded(out:Array(u64),a:Array(u64),n:i32,choice:u64):
    i = ctaid_x()*128 + tid_x()
    if i < n:
        v = a[i*2]
        if i % 2 == 0:
            v *= choice
        out[i] = v
