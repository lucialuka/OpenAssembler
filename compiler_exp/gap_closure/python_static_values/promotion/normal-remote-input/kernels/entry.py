from openassembler.pyfront import *
from library.helpers import signed_affine,unsigned_table,nested_table,floating_scale,integer_expression
from library import constants as configuration
raise RuntimeError('entry must not execute')
@kernel
def signed(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i] = signed_affine(a[i])
@kernel
def unsigned(a:Array(u64),out:Array(u64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i] = unsigned_table(a[i])
@kernel
def nested(a:Array(u64),out:Array(u64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i] = nested_table(a[i])
@kernel
def floating(a:Array(f32),out:Array(f32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i] = floating_scale(a[i])
@kernel
def shifted(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i] = integer_expression(a[i])
@kernel
def shared_helper(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i] = signed_affine(a[i])
@kernel
def direct_constant(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i] = a[i]+i64(configuration.COEFFICIENTS[-1])
@kernel
def negative_zero(a:Array(f32),out:Array(f32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i] = configuration.NEGZERO
@kernel
def local_shadow(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        configuration = a[i]
        out[i] = signed_affine(configuration)
