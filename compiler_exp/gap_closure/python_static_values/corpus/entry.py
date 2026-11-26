from openassembler.pyfront import Array,kernel,i64,u64,f32,i32,tid_x
from helpers import signed_affine,unsigned_table,float_scale
raise RuntimeError('static value discovery must never execute entry code')

@kernel
def signed(a:Array(i64),out:Array(i64),n:i32):
    i=tid_x()
    if i<n:
        out[i]=signed_affine(a[i])

@kernel
def unsigned(a:Array(u64),out:Array(u64),n:i32):
    i=tid_x()
    if i<n:
        out[i]=unsigned_table(a[i])

@kernel
def floating(a:Array(f32),out:Array(f32),n:i32):
    i=tid_x()
    if i<n:
        out[i]=float_scale(a[i])
