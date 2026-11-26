from typing import Final
from openassembler.pyfront import *
from library import helpers
raise RuntimeError("entry must never execute")
@kernel
def plain(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_plain(a[i])
@kernel
def empty(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_empty(a[i])
@kernel
def break_loop(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_break_loop(a[i])
@kernel
def continue_loop(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_continue_loop(a[i])
@kernel
def nested(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_nested(a[i])
@kernel
def range_inside(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_range_inside(a[i])
@kernel
def tuple_inside_range(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_tuple_inside_range(a[i])
@kernel
def target_after(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_target_after(a[i])
@kernel
def target_mutation(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_target_mutation(a[i])
@kernel
def break_target(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_break_target(a[i])
@kernel
def empty_target(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_empty_target(a[i])
@kernel
def loop_else(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_loop_else(a[i])
@kernel
def nested_else(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_nested_else(a[i])
@kernel
def sliced(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_sliced(a[i])
@kernel
def literal_tuple(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_literal_tuple(a[i])
@kernel
def wide_signed(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_wide_signed(a[i])
@kernel
def wide_unsigned(a:Array(u64),out:Array(u64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_wide_unsigned(a[i])
@kernel
def wide_target(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_wide_target(a[i])
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
