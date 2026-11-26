from typing import Final
from openassembler.pyfront import *
from library import helpers
from library import constants
raise RuntimeError("entry must never execute")
@kernel
def positive_overflow(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_positive_overflow(a[i])+i64(constants.ZERO)
@kernel
def negative_overflow(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_negative_overflow(a[i])+i64(constants.ZERO)
@kernel
def large_positive_step(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_large_positive_step(a[i])+i64(constants.ZERO)
@kernel
def minimum_step(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_minimum_step(a[i])+i64(constants.ZERO)
@kernel
def runtime_positive(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_runtime_positive(a[i])+i64(constants.ZERO)
@kernel
def runtime_negative(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_runtime_negative(a[i])+i64(constants.ZERO)
@kernel
def overflow_else(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_overflow_else(a[i])+i64(constants.ZERO)
@kernel
def empty_else(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_empty_else(a[i])+i64(constants.ZERO)
@kernel
def break_else(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_break_else(a[i])+i64(constants.ZERO)
@kernel
def continue_else(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_continue_else(a[i])+i64(constants.ZERO)
@kernel
def mutated_target(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_mutated_target(a[i])+i64(constants.ZERO)
@kernel
def bound_snapshot(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_bound_snapshot(a[i])+i64(constants.ZERO)
@kernel
def nested_else(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_nested_else(a[i])+i64(constants.ZERO)
@kernel
def else_break_outer(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_else_break_outer(a[i])+i64(constants.ZERO)
@kernel
def while_else(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_while_else(a[i])+i64(constants.ZERO)
@kernel
def while_break(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_while_break(a[i])+i64(constants.ZERO)
@kernel
def while_continue(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_while_continue(a[i])+i64(constants.ZERO)
@kernel
def early_return(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_early_return(a[i])+i64(constants.ZERO)
@kernel
def while_else_return(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_while_else_return(a[i])+i64(constants.ZERO)
@kernel
def else_continue_outer(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_else_continue_outer(a[i])+i64(constants.ZERO)
