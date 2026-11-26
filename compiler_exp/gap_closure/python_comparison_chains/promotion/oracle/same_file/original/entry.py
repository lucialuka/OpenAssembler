from typing import Final
from openassembler.pyfront import *
ZERO:Final=0
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
@kernel
def ordinary(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_ordinary(a[i])+i32(ZERO)
@device
def helper_negated(x:i32)->i32:
    value=11
    if not (-3<=x<=7):
        value+=2
    return value
@kernel
def negated(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_negated(a[i])+i32(ZERO)
@device
def helper_or_chain(x:i32)->i32:
    value=11
    if -3<x<0 or 1<=x<=7:
        value+=3
    return value
@kernel
def or_chain(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_or_chain(a[i])+i32(ZERO)
@device
def helper_and_chain(x:i32)->i32:
    value=11
    if -3<x<7 and 0<=x<3:
        value+=4
    return value
@kernel
def and_chain(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_and_chain(a[i])+i32(ZERO)
@device
def helper_while_chain(x:i32)->i32:
    value=11
    step=0
    while 0<=step<3 and -3<x<7:
        value+=5
        step+=1
    return value
@kernel
def while_chain(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_while_chain(a[i])+i32(ZERO)
@device
def helper_mixed_ops(x:i32)->i32:
    value=11
    if -3<=x!=3<7:
        value+=6
    return value
@kernel
def mixed_ops(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_mixed_ops(a[i])+i32(ZERO)
@device
def helper_length_four(x:i32)->i32:
    value=11
    if -7<-3<x<3<7:
        value+=7
    return value
@kernel
def length_four(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_length_four(a[i])+i32(ZERO)
@device
def helper_context_bridge(x:i64)->i32:
    value=11
    if 0<1<x:
        value+=8
    return value
@kernel
def context_bridge(a:Array(i64),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_context_bridge(a[i])+i32(ZERO)
@device
def helper_unsigned(x:u32)->i32:
    value=11
    if u32(0)<=x<u32(2147483648):
        value+=9
    return value
@kernel
def unsigned(a:Array(u32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_unsigned(a[i])+i32(ZERO)
@device
def helper_signed_wide(x:i64)->i32:
    value=11
    if i64(-1099511627776)<x<i64(1099511627776):
        value+=10
    return value
@kernel
def signed_wide(a:Array(i64),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_signed_wide(a[i])+i32(ZERO)
@device
def helper_unsigned_wide(x:u64)->i32:
    value=11
    if u64(0)<=x<u64(9223372036854775808):
        value+=11
    return value
@kernel
def unsigned_wide(a:Array(u64),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_unsigned_wide(a[i])+i32(ZERO)
@device
def helper_special_ordered(x:f32)->i32:
    value=11
    if -2.0<x<3.0:
        value+=12
    return value
@kernel
def special_ordered(a:Array(f32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_special_ordered(a[i])+i32(ZERO)
@device
def helper_special_ne(x:f32)->i32:
    value=11
    if -2.0!=x!=3.0:
        value+=13
    return value
@kernel
def special_ne(a:Array(f32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_special_ne(a[i])+i32(ZERO)
@device
def helper_special_negated(x:f32)->i32:
    value=11
    if not (-2.0<=x<=3.0):
        value+=14
    return value
@kernel
def special_negated(a:Array(f32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_special_negated(a[i])+i32(ZERO)
@device
def helper_nested_helper(x:i64)->i32:
    value=11
    if i64(-3)<bump(x)<i64(7):
        value+=15
    return value
@kernel
def nested_helper(a:Array(i64),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_nested_helper(a[i])+i32(ZERO)
@device
def helper_return_helper(x:i32)->i32:
    value=11
    if -3<x<7:
        return value+16
    return value
@kernel
def return_helper(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helper_return_helper(a[i])+i32(ZERO)
@kernel
def atomic_middle(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        x=a[i]
        value=11+i32(ZERO)
        if -1<atomic_add(count,i,1)<1:
            value+=17
        out[i]=value
@kernel
def atomic_last(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        x=a[i]
        value=11+i32(ZERO)
        if x<0<atomic_add(count,i,1) or 0<=x<3:
            value+=18
        out[i]=value
@kernel
def atomic_multi(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        x=a[i]
        value=11+i32(ZERO)
        if -1<atomic_add(count,i,1)<atomic_add(count,i,1)<2:
            value+=19
        out[i]=value
@kernel
def load_snapshot(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        x=a[i]
        value=11+i32(ZERO)
        if -1<atomic_add(count,i,1)<count[i]<2:
            value+=20
        out[i]=value
