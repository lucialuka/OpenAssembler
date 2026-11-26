from openassembler.pyfront import Array, i64, u64, f32, i32, kernel, tid_x
from arithmetic.transforms import signed_step as repeated_product
import arithmetic.transforms as transforms
from arithmetic.base import affine
raise RuntimeError('static compilation must not execute the entry module')

@kernel
def signed(a: Array(i64), b: Array(i64), out: Array(i64), n: i32):
    i = tid_x()
    if i < n:
        out[i] = repeated_product(a[i], b[i])

@kernel
def unsigned(a: Array(u64), b: Array(u64), out: Array(u64), n: i32):
    i = tid_x()
    if i < n:
        out[i] = transforms.unsigned_step(a[i], b[i])

@kernel
def floating(a: Array(f32), b: Array(f32), out: Array(f32), n: i32):
    i = tid_x()
    if i < n:
        out[i] = affine(a[i], 2.0, b[i])
