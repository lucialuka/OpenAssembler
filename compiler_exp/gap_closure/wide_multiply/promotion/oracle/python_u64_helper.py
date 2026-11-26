from openassembler.pyfront import *
@device
def product(x:u64,y:u64)->u64:
    return x*y

@kernel
def python_u64_helper(out:Array(u64),a:Array(u64),n:i32,choice:u64):
    i = ctaid_x()*128 + tid_x()
    if i < n:
        v = product(a[i*2],a[i*2+1])
        out[i] = v
