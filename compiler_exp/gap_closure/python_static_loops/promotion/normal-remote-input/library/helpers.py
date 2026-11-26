from typing import Final
from openassembler.pyfront import *
from . import constants as constants
raise RuntimeError("helper module must never execute")
@device
def helper_plain(x:i64)->i64:
    value=i64(0)
    for c in constants.VALUES_PLAIN:
        value=value*x+i64(c)
    return value
@device
def helper_empty(x:i64)->i64:
    value=x
    for c in constants.VALUES_EMPTY:
        value+=i64(c)
    else:
        value+=i64(17)
    return value
@device
def helper_break_loop(x:i64)->i64:
    value=i64(0)
    for c in constants.VALUES_BREAK_LOOP:
        if x<i64(c):
            break
        value=value*x+i64(c)
    return value
@device
def helper_continue_loop(x:i64)->i64:
    value=i64(0)
    for c in constants.VALUES_CONTINUE_LOOP:
        if x<i64(c):
            continue
        value=value*x+i64(c)
    return value
@device
def helper_nested(x:i64)->i64:
    value=x
    for c in constants.VALUES_NESTED:
        for d in (1,2):
            if c<0 and d==2:
                break
            value+=i64(c*d)
    return value
@device
def helper_range_inside(x:i64)->i64:
    value=x
    for c in constants.VALUES_RANGE_INSIDE:
        for d in range(3):
            if d==1:
                continue
            value+=i64(c*d)
    return value
@device
def helper_tuple_inside_range(x:i64)->i64:
    value=x
    for step in range(2):
        for c in constants.VALUES_TUPLE_INSIDE_RANGE:
            value+=i64(c+step)
    return value
@device
def helper_target_after(x:i64)->i64:
    value=x
    for c in constants.VALUES_TARGET_AFTER:
        value+=i64(c)
    value+=i64(c)
    return value
@device
def helper_target_mutation(x:i64)->i64:
    value=x
    for c in constants.VALUES_TARGET_MUTATION:
        value+=i64(c)
        c=99
    value+=i64(c)
    return value
@device
def helper_break_target(x:i64)->i64:
    value=x
    for c in constants.VALUES_BREAK_TARGET:
        if x<i64(c):
            break
        value+=i64(c)
    value+=i64(c)
    return value
@device
def helper_empty_target(x:i64)->i64:
    value=x
    c=11
    for c in constants.VALUES_EMPTY_TARGET:
        value+=i64(c)
    value+=i64(c)
    return value
@device
def helper_loop_else(x:i64)->i64:
    value=x
    for c in constants.VALUES_LOOP_ELSE:
        if x<i64(c):
            break
        value+=i64(c)
    else:
        value+=i64(17)
    return value
@device
def helper_nested_else(x:i64)->i64:
    value=x
    for c in constants.VALUES_NESTED_ELSE:
        for d in (1,2):
            if x<i64(c*d):
                break
            value+=i64(d)
        else:
            value+=i64(7)
    else:
        value+=i64(19)
    return value
@device
def helper_sliced(x:i64)->i64:
    value=x
    for c in constants.VALUES_SLICED[::-1]:
        value=value*i64(3)+i64(c)
    return value
@device
def helper_literal_tuple(x:i64)->i64:
    value=x
    for c in (2,3,5):
        value*=i64(c)
    return value
@device
def helper_wide_signed(x:i64)->i64:
    value=x
    for c in constants.VALUES_WIDE_SIGNED:
        value+=c
    return value
@device
def helper_wide_unsigned(x:u64)->u64:
    value=x
    for c in constants.VALUES_WIDE_UNSIGNED:
        value+=c
    return value
@device
def helper_wide_target(x:i64)->i64:
    value=x
    c=i64(11)
    for c in constants.VALUES_WIDE_TARGET:
        value+=c
    value+=c
    return value
@device
def helper_floating(x:f32)->f32:
    value=x
    for c in constants.VALUES_FLOATING:
        value=value*c+c
    return value
@device
def helper_mixed_exact(x:f32)->f32:
    value=x
    for c in constants.VALUES_MIXED_EXACT:
        value=value*c+c
    return value
