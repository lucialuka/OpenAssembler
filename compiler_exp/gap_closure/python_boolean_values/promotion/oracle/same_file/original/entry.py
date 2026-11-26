from typing import Final
from openassembler.pyfront import *
ZERO:Final=0
@device
def pick(x:i64)->i64:
    return x and 7
@device
def helper_operand_and(x:i32)->i32:
    return x and 7
@kernel
def operand_and(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        out[i]=helper_operand_and(a[i])
@device
def helper_operand_or(x:i32)->i32:
    return x or 9
@kernel
def operand_or(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        out[i]=helper_operand_or(a[i])
@device
def helper_operand_nested(x:i32)->i32:
    return (x and 7) or 9
@kernel
def operand_nested(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        out[i]=helper_operand_nested(a[i])
@device
def helper_truth_flags(x:i32)->i32:
    flag=0<x<7
    inverse=not (flag or False)
    enabled=bool(x) and True
    return i32(flag)+i32(inverse)*2+i32(enabled)*4
@kernel
def truth_flags(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        out[i]=helper_truth_flags(a[i])
@device
def helper_literal_left(x:i64)->i64:
    return 0 or x
@kernel
def literal_left(a:Array(i64),out:Array(i64),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        out[i]=helper_literal_left(a[i])
@device
def helper_literal_right(x:i64)->i64:
    return x or 0
@kernel
def literal_right(a:Array(i64),out:Array(i64),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        out[i]=helper_literal_right(a[i])
@device
def helper_signed_wide(x:i64)->i64:
    return x and -7
@kernel
def signed_wide(a:Array(i64),out:Array(i64),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        out[i]=helper_signed_wide(a[i])
@device
def helper_unsigned_wide(x:u64)->u64:
    return x or 18446744073709551615
@kernel
def unsigned_wide(a:Array(u64),out:Array(u64),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        out[i]=helper_unsigned_wide(a[i])
@device
def helper_bool_literals(x:i32)->i32:
    first=True
    second=False
    return i32(first and not second)+i32(second or first)*2
@kernel
def bool_literals(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        out[i]=helper_bool_literals(a[i])
@device
def helper_not_wide(x:i64)->i64:
    return i64(not x)
@kernel
def not_wide(a:Array(i64),out:Array(i64),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        out[i]=helper_not_wide(a[i])
@device
def helper_float_and(x:f32)->f32:
    return x and -0.0
@kernel
def float_and(a:Array(f32),out:Array(f32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        out[i]=helper_float_and(a[i])
@device
def helper_float_or(x:f32)->f32:
    return x or -0.0
@kernel
def float_or(a:Array(f32),out:Array(f32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        out[i]=helper_float_or(a[i])
@device
def helper_float_truth(x:f32)->f32:
    return float(i32(bool(x)))
@kernel
def float_truth(a:Array(f32),out:Array(f32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        out[i]=helper_float_truth(a[i])
@device
def helper_float_not(x:f32)->f32:
    flag=x!=0.0
    return float(i32(not flag))
@kernel
def float_not(a:Array(f32),out:Array(f32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        out[i]=helper_float_not(a[i])
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
@kernel
def predicate_while(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        out[i]=helper_predicate_while(a[i])
@device
def helper_nested_helper(x:i64)->i64:
    return pick(x) or 9
@kernel
def nested_helper(a:Array(i64),out:Array(i64),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        out[i]=helper_nested_helper(a[i])
@kernel
def atomic_and(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        x=a[i]
        result=x and atomic_add(count,i,1)
        out[i]=result
@kernel
def atomic_or(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        x=a[i]
        result=x or atomic_add(count,i,1)
        out[i]=result
@kernel
def atomic_nested(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        x=a[i]
        result=(x and atomic_add(count,i,1)) or atomic_add(count,i,1)
        out[i]=result
@kernel
def atomic_bool(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        x=a[i]
        flag=bool(atomic_add(count,i,1))
        second=not (flag or bool(atomic_add(count,i,1)))
        result=i32(flag)+i32(second)*2
        out[i]=result
