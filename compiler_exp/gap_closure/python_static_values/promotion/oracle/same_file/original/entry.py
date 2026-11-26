from openassembler.pyfront import *
from typing import Final as Frozen
import typing as types
raise RuntimeError('static declarations are source-only')
GAIN: Frozen[int] = 3
BIAS: types.Final[int] = -(1 << 32) + 7
COEFFICIENTS: Frozen[tuple[int,int,int]] = (3,5,7)
GRID: Frozen[tuple[tuple[int,int],tuple[int,int]]] = ((11,13),(17,19))
SCALE: Frozen[float] = (1.0 / 8.0) * 4.0
NEGZERO: Frozen[float] = -0.0
SHIFT: Frozen[int] = (5 << 2) - 15 + (2**3 - 8)
COUNT: Frozen[int] = len(COEFFICIENTS)
raise RuntimeError('helper module must not execute')
@device
def signed_affine(x:i64)->i64:
    return x*GAIN + BIAS
@device
def unsigned_table(x:u64)->u64:
    return x*u64(COEFFICIENTS[1]) + u64(COEFFICIENTS[-1])
@device
def nested_table(x:u64)->u64:
    return x+u64(GRID[-1][0] + COEFFICIENTS[1:][0])
@device
def floating_scale(x:f32)->f32:
    return x*SCALE
@device
def integer_expression(x:i64)->i64:
    return (x << SHIFT) + i64(COUNT)
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
        out[i] = a[i]+i64(COEFFICIENTS[-1])
@kernel
def negative_zero(a:Array(f32),out:Array(f32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i] = NEGZERO
@kernel
def local_shadow(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        configuration = a[i]
        out[i] = signed_affine(configuration)
