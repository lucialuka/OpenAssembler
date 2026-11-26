from typing import Final
from openassembler.pyfront import *
from library import helpers
raise RuntimeError("entry must never execute")
@kernel
def classify(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_classify(a[i])
@kernel
def branches(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_branches(a[i])
@kernel
def live_init(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_live_init(a[i])
@kernel
def range_return(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_range_return(a[i])
@kernel
def while_return(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_while_return(a[i])
@kernel
def nested_return(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_nested_return(a[i])
@kernel
def break_return(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_break_return(a[i])
@kernel
def continue_return(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_continue_return(a[i])
@kernel
def empty_else(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_empty_else(a[i])
@kernel
def finite_first(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_finite_first(a[i])
@kernel
def loop_else_return(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_loop_else_return(a[i])
@kernel
def nested_else_return(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_nested_else_return(a[i])
@kernel
def caller_snapshot(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_caller_snapshot(a[i])+a[i]+helpers.helper_caller_snapshot(a[i])
@kernel
def nested_calls(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_nested_calls(a[i])
@kernel
def argument_mutation(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_argument_mutation(a[i])
@kernel
def multiple_sites(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_multiple_sites(a[i])+helpers.helper_multiple_sites(a[i])
@kernel
def return_expression(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_return_expression(a[i])
@kernel
def wide_unsigned(a:Array(u64),out:Array(u64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_wide_unsigned(a[i])
@kernel
def floating(a:Array(f32),out:Array(f32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_floating(a[i])
@kernel
def mixed_exact(a:Array(f32),out:Array(f32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_mixed_exact(a[i])
