from typing import Final
from openassembler.pyfront import *
from library import helpers
from library import constants
raise RuntimeError("entry must never execute")
@kernel
def operand_and(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        out[i]=helpers.helper_operand_and(a[i])
@kernel
def operand_or(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        out[i]=helpers.helper_operand_or(a[i])
@kernel
def operand_nested(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        out[i]=helpers.helper_operand_nested(a[i])
@kernel
def truth_flags(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        out[i]=helpers.helper_truth_flags(a[i])
@kernel
def literal_left(a:Array(i64),out:Array(i64),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        out[i]=helpers.helper_literal_left(a[i])
@kernel
def literal_right(a:Array(i64),out:Array(i64),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        out[i]=helpers.helper_literal_right(a[i])
@kernel
def signed_wide(a:Array(i64),out:Array(i64),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        out[i]=helpers.helper_signed_wide(a[i])
@kernel
def unsigned_wide(a:Array(u64),out:Array(u64),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        out[i]=helpers.helper_unsigned_wide(a[i])
@kernel
def bool_literals(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        out[i]=helpers.helper_bool_literals(a[i])
@kernel
def not_wide(a:Array(i64),out:Array(i64),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        out[i]=helpers.helper_not_wide(a[i])
@kernel
def float_and(a:Array(f32),out:Array(f32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        out[i]=helpers.helper_float_and(a[i])
@kernel
def float_or(a:Array(f32),out:Array(f32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        out[i]=helpers.helper_float_or(a[i])
@kernel
def float_truth(a:Array(f32),out:Array(f32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        out[i]=helpers.helper_float_truth(a[i])
@kernel
def float_not(a:Array(f32),out:Array(f32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        out[i]=helpers.helper_float_not(a[i])
@kernel
def predicate_while(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        out[i]=helpers.helper_predicate_while(a[i])
@kernel
def nested_helper(a:Array(i64),out:Array(i64),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        out[i]=helpers.helper_nested_helper(a[i])
@kernel
def atomic_and(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=x and atomic_add(count,i,1)
        out[i]=result
@kernel
def atomic_or(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=x or atomic_add(count,i,1)
        out[i]=result
@kernel
def atomic_nested(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=(x and atomic_add(count,i,1)) or atomic_add(count,i,1)
        out[i]=result
@kernel
def atomic_bool(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        flag=bool(atomic_add(count,i,1))
        second=not (flag or bool(atomic_add(count,i,1)))
        result=i32(flag)+i32(second)*2
        out[i]=result
