from typing import Final
from openassembler.pyfront import *
VALUES:Final=(1,2,3)
@device
def choose(x:i64)->i64:
    x+=i64(7)
    if x<i64(0):
        return -x
    return x+i64(3)
@device
def helper_classify(x:i64)->i64:
    if x<i64(0):
        return -x
    if x==i64(0):
        return i64(17)
    return x+i64(3)
@kernel
def classify(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_classify(a[i])
@device
def helper_branches(x:i64)->i64:
    if x<i64(0):
        return -x
    else:
        return x+i64(3)
@kernel
def branches(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_branches(a[i])
@device
def helper_live_init(x:i64)->i64:
    if x<i64(0):
        return -x
    else:
        value=x+i64(7)
    return value
@kernel
def live_init(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_live_init(a[i])
@device
def helper_range_return(x:i64)->i64:
    value=i64(0)
    for step in range(5):
        value+=i64(step)
        if value>x:
            return value
    return value+i64(11)
@kernel
def range_return(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_range_return(a[i])
@device
def helper_while_return(x:i64)->i64:
    value=i64(0)
    step=0
    while step<5:
        value+=i64(step)
        if value>x:
            return value
        step+=1
    return value+i64(11)
@kernel
def while_return(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_while_return(a[i])
@device
def helper_nested_return(x:i64)->i64:
    value=i64(0)
    for outer in VALUES:
        for inner in (3,5):
            value+=i64(outer*inner)
            if value>x:
                return value
    return value+i64(19)
@kernel
def nested_return(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_nested_return(a[i])
@device
def helper_break_return(x:i64)->i64:
    value=x
    for step in (3,7):
        if x<i64(step):
            break
        value+=i64(step)
    else:
        return value+i64(17)
    return value
@kernel
def break_return(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_break_return(a[i])
@device
def helper_continue_return(x:i64)->i64:
    value=i64(0)
    for step in (1,2,3,4):
        if step==2:
            continue
        value+=i64(step)
        if x<value:
            return value
    return value+i64(23)
@kernel
def continue_return(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_continue_return(a[i])
@device
def helper_empty_else(x:i64)->i64:
    for step in ():
        return x
    else:
        return x+i64(11)
@kernel
def empty_else(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_empty_else(a[i])
@device
def helper_finite_first(x:i64)->i64:
    for step in (3,7):
        return x+i64(step)
@kernel
def finite_first(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_finite_first(a[i])
@device
def helper_loop_else_return(x:i64)->i64:
    value=x
    for step in VALUES:
        if x<i64(step):
            return value-i64(13)
        value+=i64(step)
    else:
        return value+i64(17)
@kernel
def loop_else_return(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_loop_else_return(a[i])
@device
def helper_nested_else_return(x:i64)->i64:
    value=i64(0)
    for outer in VALUES:
        for inner in (3,5):
            if x<i64(inner):
                break
            if x<i64(outer*inner):
                return value+i64(outer)
            value+=i64(inner)
        else:
            if x<i64(19):
                return value+i64(7)
    return value+i64(23)
@kernel
def nested_else_return(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_nested_else_return(a[i])
@device
def helper_caller_snapshot(x:i64)->i64:
    x+=i64(7)
    if x<i64(0):
        return -x
    return x+i64(3)
@kernel
def caller_snapshot(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_caller_snapshot(a[i])+a[i]+helper_caller_snapshot(a[i])
@device
def helper_nested_calls(x:i64)->i64:
    value=choose(x)
    if value<i64(0):
        return choose(value+i64(7))
    return choose(value)+choose(x)
@kernel
def nested_calls(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_nested_calls(a[i])
@device
def helper_argument_mutation(x:i64)->i64:
    value=choose(x)+choose(x+i64(1))
    if value<i64(0):
        return value-x
    return value+x
@kernel
def argument_mutation(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_argument_mutation(a[i])
@device
def helper_multiple_sites(x:i64)->i64:
    if x<i64(0):
        return choose(x)+x
    return choose(x)-x
@kernel
def multiple_sites(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_multiple_sites(a[i])+helper_multiple_sites(a[i])
@device
def helper_return_expression(x:i64)->i64:
    if x<i64(0):
        return choose(x)*i64(3)+choose(x+i64(1))
    return choose(x)*i64(5)-choose(x-i64(1))
@kernel
def return_expression(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_return_expression(a[i])
@device
def helper_wide_unsigned(x:u64)->u64:
    if x<u64(9223372036854775808):
        return x+u64(1099511627777)
    return x*u64(3)+u64(18446744073709551615)
@kernel
def wide_unsigned(a:Array(u64),out:Array(u64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_wide_unsigned(a[i])
@device
def helper_floating(x:f32)->f32:
    if x<0.0:
        return x*0.5-0.25
    return x*2.0+0.5
@kernel
def floating(a:Array(f32),out:Array(f32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_floating(a[i])
@device
def helper_mixed_exact(x:f32)->f32:
    value=x
    for coefficient in (2.0,0.5,-1.0):
        value=value*coefficient+coefficient
        if value<0.0:
            return value
    return value+0.25
@kernel
def mixed_exact(a:Array(f32),out:Array(f32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_mixed_exact(a[i])
