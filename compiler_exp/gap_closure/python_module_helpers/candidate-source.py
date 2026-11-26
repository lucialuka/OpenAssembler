from openassembler.pyfront import *
import math

@device
def __oa_module_0_signed_step(x: i64, y: i64) -> i64:
    result = x
    for step in range(3):
        result = __oa_module_1_signed_product(result, y)
    return result

@device
def __oa_module_1_signed_product(x: i64, y: i64) -> i64:
    return x * y

@device
def __oa_module_2_unsigned_step(x: u64, y: u64) -> u64:
    return __oa_module_3_unsigned_product(x, y) + x

@device
def __oa_module_3_unsigned_product(x: u64, y: u64) -> u64:
    return x * y

@device
def __oa_module_4_affine(x: f32, a: f32, b: f32) -> f32:
    return x * a + b

@kernel
def signed(a: Array(i64), b: Array(i64), out: Array(i64), n: i32):
    i = tid_x()
    if i < n:
        out[i] = __oa_module_0_signed_step(a[i], b[i])

@kernel
def unsigned(a: Array(u64), b: Array(u64), out: Array(u64), n: i32):
    i = tid_x()
    if i < n:
        out[i] = __oa_module_2_unsigned_step(a[i], b[i])

@kernel
def floating(a: Array(f32), b: Array(f32), out: Array(f32), n: i32):
    i = tid_x()
    if i < n:
        out[i] = __oa_module_4_affine(a[i], 2.0, b[i])
