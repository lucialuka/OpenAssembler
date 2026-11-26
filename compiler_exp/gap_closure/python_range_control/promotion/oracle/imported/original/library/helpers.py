from typing import Final
from openassembler.pyfront import *
from . import constants
raise RuntimeError("helper must never execute")
@device
def helper_positive_overflow(x:i64)->i64:
    value=i64(0)
    target=17
    for target in range(2147483646,2147483647,2):
        value+=i64(target)
    return value+i64(target)
@device
def helper_negative_overflow(x:i64)->i64:
    value=i64(0)
    target=17
    for target in range(-2147483647,-2147483648,-2):
        value+=i64(target)
    return value+i64(target)
@device
def helper_large_positive_step(x:i64)->i64:
    value=i64(0)
    target=17
    for target in range(-2147483648,2147483647,2147483647):
        value+=i64(target)
    return value+i64(target)
@device
def helper_minimum_step(x:i64)->i64:
    value=i64(0)
    target=17
    for target in range(2147483647,-2147483648,-2147483648):
        value+=i64(target)
    return value+i64(target)
@device
def helper_runtime_positive(x:i64)->i64:
    start=2147483643+i32(x&i64(3))
    stop=2147483647
    target=17
    value=i64(0)
    for target in range(start,stop,3):
        value+=i64(target)
    return value+i64(target)
@device
def helper_runtime_negative(x:i64)->i64:
    start=-2147483644-i32(x&i64(3))
    stop=-2147483648
    target=17
    value=i64(0)
    for target in range(start,stop,-3):
        value+=i64(target)
    return value+i64(target)
@device
def helper_overflow_else(x:i64)->i64:
    value=x
    for target in range(2147483646,2147483647,2):
        value+=i64(target)
    else:
        value+=i64(19)
    return value
@device
def helper_empty_else(x:i64)->i64:
    target=17
    value=x
    for target in range(3,3,2):
        value+=i64(99)
    else:
        value+=i64(target)
    return value
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
@device
def helper_mutated_target(x:i64)->i64:
    value=x
    target=17
    for target in range(2147483643,2147483647,3):
        value+=i64(target)
        target=11
    return value+i64(target)
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
@device
def helper_early_return(x:i64)->i64:
    for target in range(2147483643,2147483647,3):
        if x&i64(1):
            return x+i64(target)
    else:
        return x+i64(19)
@device
def helper_while_else_return(x:i64)->i64:
    step=0
    while step<3:
        if x&i64(1):
            return x+i64(step)
        step+=1
    else:
        return x+i64(19)
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
