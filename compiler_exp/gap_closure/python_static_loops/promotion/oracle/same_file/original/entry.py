from typing import Final
from openassembler.pyfront import *
VALUES_PLAIN:Final=(3,-5,7)
VALUES_EMPTY:Final=()
VALUES_BREAK_LOOP:Final=(3,-5,7)
VALUES_CONTINUE_LOOP:Final=(3,-5,7)
VALUES_NESTED:Final=(3,-5,7)
VALUES_RANGE_INSIDE:Final=(3,-5,7)
VALUES_TUPLE_INSIDE_RANGE:Final=(3,-5,7)
VALUES_TARGET_AFTER:Final=(3,-5,7)
VALUES_TARGET_MUTATION:Final=(3,-5,7)
VALUES_BREAK_TARGET:Final=(3,-5,7)
VALUES_EMPTY_TARGET:Final=()
VALUES_LOOP_ELSE:Final=(3,-5,7)
VALUES_NESTED_ELSE:Final=(3,-5,7)
VALUES_SLICED:Final=(3,-5,7)
VALUES_LITERAL_TUPLE:Final=(3,-5,7)
VALUES_WIDE_SIGNED:Final=(-9223372036854775808,1099511627777,9223372036854775807)
VALUES_WIDE_UNSIGNED:Final=(0,9223372036854775808,18446744073709551615)
VALUES_WIDE_TARGET:Final=(3,-5,7)
VALUES_FLOATING:Final=(0.5,-0.25,2.0)
VALUES_MIXED_EXACT:Final=(2,0.5,-1)
@device
def helper_plain(x:i64)->i64:
    value=i64(0)
    for c in VALUES_PLAIN:
        value=value*x+i64(c)
    return value
@kernel
def plain(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_plain(a[i])
@device
def helper_empty(x:i64)->i64:
    value=x
    for c in VALUES_EMPTY:
        value+=i64(c)
    else:
        value+=i64(17)
    return value
@kernel
def empty(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_empty(a[i])
@device
def helper_break_loop(x:i64)->i64:
    value=i64(0)
    for c in VALUES_BREAK_LOOP:
        if x<i64(c):
            break
        value=value*x+i64(c)
    return value
@kernel
def break_loop(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_break_loop(a[i])
@device
def helper_continue_loop(x:i64)->i64:
    value=i64(0)
    for c in VALUES_CONTINUE_LOOP:
        if x<i64(c):
            continue
        value=value*x+i64(c)
    return value
@kernel
def continue_loop(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_continue_loop(a[i])
@device
def helper_nested(x:i64)->i64:
    value=x
    for c in VALUES_NESTED:
        for d in (1,2):
            if c<0 and d==2:
                break
            value+=i64(c*d)
    return value
@kernel
def nested(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_nested(a[i])
@device
def helper_range_inside(x:i64)->i64:
    value=x
    for c in VALUES_RANGE_INSIDE:
        for d in range(3):
            if d==1:
                continue
            value+=i64(c*d)
    return value
@kernel
def range_inside(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_range_inside(a[i])
@device
def helper_tuple_inside_range(x:i64)->i64:
    value=x
    for step in range(2):
        for c in VALUES_TUPLE_INSIDE_RANGE:
            value+=i64(c+step)
    return value
@kernel
def tuple_inside_range(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_tuple_inside_range(a[i])
@device
def helper_target_after(x:i64)->i64:
    value=x
    for c in VALUES_TARGET_AFTER:
        value+=i64(c)
    value+=i64(c)
    return value
@kernel
def target_after(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_target_after(a[i])
@device
def helper_target_mutation(x:i64)->i64:
    value=x
    for c in VALUES_TARGET_MUTATION:
        value+=i64(c)
        c=99
    value+=i64(c)
    return value
@kernel
def target_mutation(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_target_mutation(a[i])
@device
def helper_break_target(x:i64)->i64:
    value=x
    for c in VALUES_BREAK_TARGET:
        if x<i64(c):
            break
        value+=i64(c)
    value+=i64(c)
    return value
@kernel
def break_target(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_break_target(a[i])
@device
def helper_empty_target(x:i64)->i64:
    value=x
    c=11
    for c in VALUES_EMPTY_TARGET:
        value+=i64(c)
    value+=i64(c)
    return value
@kernel
def empty_target(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_empty_target(a[i])
@device
def helper_loop_else(x:i64)->i64:
    value=x
    for c in VALUES_LOOP_ELSE:
        if x<i64(c):
            break
        value+=i64(c)
    else:
        value+=i64(17)
    return value
@kernel
def loop_else(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_loop_else(a[i])
@device
def helper_nested_else(x:i64)->i64:
    value=x
    for c in VALUES_NESTED_ELSE:
        for d in (1,2):
            if x<i64(c*d):
                break
            value+=i64(d)
        else:
            value+=i64(7)
    else:
        value+=i64(19)
    return value
@kernel
def nested_else(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_nested_else(a[i])
@device
def helper_sliced(x:i64)->i64:
    value=x
    for c in VALUES_SLICED[::-1]:
        value=value*i64(3)+i64(c)
    return value
@kernel
def sliced(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_sliced(a[i])
@device
def helper_literal_tuple(x:i64)->i64:
    value=x
    for c in (2,3,5):
        value*=i64(c)
    return value
@kernel
def literal_tuple(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_literal_tuple(a[i])
@device
def helper_wide_signed(x:i64)->i64:
    value=x
    for c in VALUES_WIDE_SIGNED:
        value+=c
    return value
@kernel
def wide_signed(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_wide_signed(a[i])
@device
def helper_wide_unsigned(x:u64)->u64:
    value=x
    for c in VALUES_WIDE_UNSIGNED:
        value+=c
    return value
@kernel
def wide_unsigned(a:Array(u64),out:Array(u64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_wide_unsigned(a[i])
@device
def helper_wide_target(x:i64)->i64:
    value=x
    c=i64(11)
    for c in VALUES_WIDE_TARGET:
        value+=c
    value+=c
    return value
@kernel
def wide_target(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_wide_target(a[i])
@device
def helper_floating(x:f32)->f32:
    value=x
    for c in VALUES_FLOATING:
        value=value*c+c
    return value
@kernel
def floating(a:Array(f32),out:Array(f32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_floating(a[i])
@device
def helper_mixed_exact(x:f32)->f32:
    value=x
    for c in VALUES_MIXED_EXACT:
        value=value*c+c
    return value
@kernel
def mixed_exact(a:Array(f32),out:Array(f32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_mixed_exact(a[i])
