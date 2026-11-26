from typing import Final
from openassembler.pyfront import *
from . import constants
raise RuntimeError("helper must never execute")
@device
def bump(x:i64)->i64:
    x+=i64(1)
    return x
@device
def helper_ordinary(x:i32)->i32:
    value=11
    if -3<x<7:
        value+=1
    return value
@device
def helper_negated(x:i32)->i32:
    value=11
    if not (-3<=x<=7):
        value+=2
    return value
@device
def helper_or_chain(x:i32)->i32:
    value=11
    if -3<x<0 or 1<=x<=7:
        value+=3
    return value
@device
def helper_and_chain(x:i32)->i32:
    value=11
    if -3<x<7 and 0<=x<3:
        value+=4
    return value
@device
def helper_while_chain(x:i32)->i32:
    value=11
    step=0
    while 0<=step<3 and -3<x<7:
        value+=5
        step+=1
    return value
@device
def helper_mixed_ops(x:i32)->i32:
    value=11
    if -3<=x!=3<7:
        value+=6
    return value
@device
def helper_length_four(x:i32)->i32:
    value=11
    if -7<-3<x<3<7:
        value+=7
    return value
@device
def helper_context_bridge(x:i64)->i32:
    value=11
    if 0<1<x:
        value+=8
    return value
@device
def helper_unsigned(x:u32)->i32:
    value=11
    if u32(0)<=x<u32(2147483648):
        value+=9
    return value
@device
def helper_signed_wide(x:i64)->i32:
    value=11
    if i64(-1099511627776)<x<i64(1099511627776):
        value+=10
    return value
@device
def helper_unsigned_wide(x:u64)->i32:
    value=11
    if u64(0)<=x<u64(9223372036854775808):
        value+=11
    return value
@device
def helper_special_ordered(x:f32)->i32:
    value=11
    if -2.0<x<3.0:
        value+=12
    return value
@device
def helper_special_ne(x:f32)->i32:
    value=11
    if -2.0!=x!=3.0:
        value+=13
    return value
@device
def helper_special_negated(x:f32)->i32:
    value=11
    if not (-2.0<=x<=3.0):
        value+=14
    return value
@device
def helper_nested_helper(x:i64)->i32:
    value=11
    if i64(-3)<bump(x)<i64(7):
        value+=15
    return value
@device
def helper_return_helper(x:i32)->i32:
    value=11
    if -3<x<7:
        return value+16
    return value
