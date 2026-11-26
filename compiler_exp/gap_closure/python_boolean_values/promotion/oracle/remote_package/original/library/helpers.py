from typing import Final
from openassembler.pyfront import *
from . import constants
raise RuntimeError("helper must never execute")
@device
def pick(x:i64)->i64:
    return x and 7
@device
def helper_operand_and(x:i32)->i32:
    return x and 7
@device
def helper_operand_or(x:i32)->i32:
    return x or 9
@device
def helper_operand_nested(x:i32)->i32:
    return (x and 7) or 9
@device
def helper_truth_flags(x:i32)->i32:
    flag=0<x<7
    inverse=not (flag or False)
    enabled=bool(x) and True
    return i32(flag)+i32(inverse)*2+i32(enabled)*4
@device
def helper_literal_left(x:i64)->i64:
    return 0 or x
@device
def helper_literal_right(x:i64)->i64:
    return x or 0
@device
def helper_signed_wide(x:i64)->i64:
    return x and -7
@device
def helper_unsigned_wide(x:u64)->u64:
    return x or 18446744073709551615
@device
def helper_bool_literals(x:i32)->i32:
    first=True
    second=False
    return i32(first and not second)+i32(second or first)*2
@device
def helper_not_wide(x:i64)->i64:
    return i64(not x)
@device
def helper_float_and(x:f32)->f32:
    return x and -0.0
@device
def helper_float_or(x:f32)->f32:
    return x or -0.0
@device
def helper_float_truth(x:f32)->f32:
    return float(i32(bool(x)))
@device
def helper_float_not(x:f32)->f32:
    flag=x!=0.0
    return float(i32(not flag))
@device
def helper_predicate_while(x:i32)->i32:
    flag=x<3
    step=0
    value=0
    again=flag and step<3
    while again:
        step+=1
        value+=1
        flag=step<2
        again=flag and step<3
    return value
@device
def helper_nested_helper(x:i64)->i64:
    return pick(x) or 9
