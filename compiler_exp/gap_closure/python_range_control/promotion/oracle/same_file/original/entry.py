from typing import Final
from openassembler.pyfront import *
ZERO:Final=0
@device
def helper_positive_overflow(x:i64)->i64:
    value=i64(0)
    target=17
    for target in range(2147483646,2147483647,2):
        value+=i64(target)
    return value+i64(target)
@kernel
def positive_overflow(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_positive_overflow(a[i])+i64(ZERO)
@device
def helper_negative_overflow(x:i64)->i64:
    value=i64(0)
    target=17
    for target in range(-2147483647,-2147483648,-2):
        value+=i64(target)
    return value+i64(target)
@kernel
def negative_overflow(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_negative_overflow(a[i])+i64(ZERO)
@device
def helper_large_positive_step(x:i64)->i64:
    value=i64(0)
    target=17
    for target in range(-2147483648,2147483647,2147483647):
        value+=i64(target)
    return value+i64(target)
@kernel
def large_positive_step(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_large_positive_step(a[i])+i64(ZERO)
@device
def helper_minimum_step(x:i64)->i64:
    value=i64(0)
    target=17
    for target in range(2147483647,-2147483648,-2147483648):
        value+=i64(target)
    return value+i64(target)
@kernel
def minimum_step(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_minimum_step(a[i])+i64(ZERO)
@device
def helper_runtime_positive(x:i64)->i64:
    start=2147483643+i32(x&i64(3))
    stop=2147483647
    target=17
    value=i64(0)
    for target in range(start,stop,3):
        value+=i64(target)
    return value+i64(target)
@kernel
def runtime_positive(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_runtime_positive(a[i])+i64(ZERO)
@device
def helper_runtime_negative(x:i64)->i64:
    start=-2147483644-i32(x&i64(3))
    stop=-2147483648
    target=17
    value=i64(0)
    for target in range(start,stop,-3):
        value+=i64(target)
    return value+i64(target)
@kernel
def runtime_negative(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_runtime_negative(a[i])+i64(ZERO)
@device
def helper_overflow_else(x:i64)->i64:
    value=x
    for target in range(2147483646,2147483647,2):
        value+=i64(target)
    else:
        value+=i64(19)
    return value
@kernel
def overflow_else(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_overflow_else(a[i])+i64(ZERO)
@device
def helper_empty_else(x:i64)->i64:
    target=17
    value=x
    for target in range(3,3,2):
        value+=i64(99)
    else:
        value+=i64(target)
    return value
@kernel
def empty_else(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_empty_else(a[i])+i64(ZERO)
@device
def helper_break_else(x:i64)->i64:
    value=x
    target=17
    for target in range(2147483643,2147483647,3):
        if x&i64(1):
            break
        value+=i64(target)
    else:
        value+=i64(19)
    return value+i64(target)
@kernel
def break_else(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_break_else(a[i])+i64(ZERO)
@device
def helper_continue_else(x:i64)->i64:
    value=x
    target=17
    for target in range(-2147483644,-2147483648,-3):
        if x&i64(1):
            continue
        value+=i64(target)
    else:
        value+=i64(19)
    return value+i64(target)
@kernel
def continue_else(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_continue_else(a[i])+i64(ZERO)
@device
def helper_mutated_target(x:i64)->i64:
    value=x
    target=17
    for target in range(2147483643,2147483647,3):
        value+=i64(target)
        target=11
    return value+i64(target)
@kernel
def mutated_target(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_mutated_target(a[i])+i64(ZERO)
@device
def helper_bound_snapshot(x:i64)->i64:
    start=2147483643
    stop=2147483647
    value=x
    for target in range(start,stop,3):
        value+=i64(target)
        start=0
        stop=0
    return value+i64(start+stop)
@kernel
def bound_snapshot(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_bound_snapshot(a[i])+i64(ZERO)
@device
def helper_nested_else(x:i64)->i64:
    value=x
    for outer in range(3):
        for target in range(2147483646,2147483647,2):
            if outer==1 and x&i64(1):
                break
            value+=i64(target)
        else:
            value+=i64(7)
    else:
        value+=i64(19)
    return value
@kernel
def nested_else(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_nested_else(a[i])+i64(ZERO)
@device
def helper_else_break_outer(x:i64)->i64:
    value=x
    for outer in range(3):
        for target in range(3,3):
            value+=i64(99)
        else:
            value+=i64(7)
            if x&i64(1):
                break
    else:
        value+=i64(19)
    return value
@kernel
def else_break_outer(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_else_break_outer(a[i])+i64(ZERO)
@device
def helper_while_else(x:i64)->i64:
    value=x
    step=0
    while step<3:
        value+=i64(step)
        step+=1
    else:
        value+=i64(19)
    return value
@kernel
def while_else(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_while_else(a[i])+i64(ZERO)
@device
def helper_while_break(x:i64)->i64:
    value=x
    step=0
    while step<3:
        if step==1 and x&i64(1):
            break
        value+=i64(step)
        step+=1
    else:
        value+=i64(19)
    return value
@kernel
def while_break(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_while_break(a[i])+i64(ZERO)
@device
def helper_while_continue(x:i64)->i64:
    value=x
    step=0
    while step<3:
        step+=1
        if step==2:
            continue
        value+=i64(step)
    else:
        value+=i64(19)
    return value
@kernel
def while_continue(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_while_continue(a[i])+i64(ZERO)
@device
def helper_early_return(x:i64)->i64:
    for target in range(2147483643,2147483647,3):
        if x&i64(1):
            return x+i64(target)
    else:
        return x+i64(19)
@kernel
def early_return(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_early_return(a[i])+i64(ZERO)
@device
def helper_while_else_return(x:i64)->i64:
    step=0
    while step<3:
        if x&i64(1):
            return x+i64(step)
        step+=1
    else:
        return x+i64(19)
@kernel
def while_else_return(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_while_else_return(a[i])+i64(ZERO)
@device
def helper_else_continue_outer(x:i64)->i64:
    value=x
    for outer in (1,2,3):
        step=0
        while step<0:
            step+=1
        else:
            if outer==2:
                continue
        value+=i64(outer)
    return value
@kernel
def else_continue_outer(a:Array(i64),out:Array(i64),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_else_continue_outer(a[i])+i64(ZERO)
