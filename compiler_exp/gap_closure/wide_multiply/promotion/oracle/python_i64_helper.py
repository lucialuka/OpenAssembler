from openassembler.pyfront import *
@device
def product(x:i64,y:i64)->i64:
    return x*y

@kernel
def python_i64_helper(out:Array(i64),a:Array(i64),n:i32,choice:i64):
    i = ctaid_x()*128 + tid_x()
    if i < n:
        v = product(a[i*2],a[i*2+1])
        out[i] = v
