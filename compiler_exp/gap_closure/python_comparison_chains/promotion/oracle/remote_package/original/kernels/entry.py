from typing import Final
from openassembler.pyfront import *
from library import helpers
from library import constants
raise RuntimeError("entry must never execute")
@kernel
def ordinary(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_ordinary(a[i])+i32(constants.ZERO)
@kernel
def negated(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_negated(a[i])+i32(constants.ZERO)
@kernel
def or_chain(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_or_chain(a[i])+i32(constants.ZERO)
@kernel
def and_chain(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_and_chain(a[i])+i32(constants.ZERO)
@kernel
def while_chain(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_while_chain(a[i])+i32(constants.ZERO)
@kernel
def mixed_ops(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_mixed_ops(a[i])+i32(constants.ZERO)
@kernel
def length_four(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_length_four(a[i])+i32(constants.ZERO)
@kernel
def context_bridge(a:Array(i64),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_context_bridge(a[i])+i32(constants.ZERO)
@kernel
def unsigned(a:Array(u32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_unsigned(a[i])+i32(constants.ZERO)
@kernel
def signed_wide(a:Array(i64),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_signed_wide(a[i])+i32(constants.ZERO)
@kernel
def unsigned_wide(a:Array(u64),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_unsigned_wide(a[i])+i32(constants.ZERO)
@kernel
def special_ordered(a:Array(f32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_special_ordered(a[i])+i32(constants.ZERO)
@kernel
def special_ne(a:Array(f32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_special_ne(a[i])+i32(constants.ZERO)
@kernel
def special_negated(a:Array(f32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_special_negated(a[i])+i32(constants.ZERO)
@kernel
def nested_helper(a:Array(i64),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_nested_helper(a[i])+i32(constants.ZERO)
@kernel
def return_helper(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        out[i]=helpers.helper_return_helper(a[i])+i32(constants.ZERO)
@kernel
def atomic_middle(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        x=a[i]
        value=11+i32(constants.ZERO)
        if -1<atomic_add(count,i,1)<1:
            value+=17
        out[i]=value
@kernel
def atomic_last(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        x=a[i]
        value=11+i32(constants.ZERO)
        if x<0<atomic_add(count,i,1) or 0<=x<3:
            value+=18
        out[i]=value
@kernel
def atomic_multi(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        x=a[i]
        value=11+i32(constants.ZERO)
        if -1<atomic_add(count,i,1)<atomic_add(count,i,1)<2:
            value+=19
        out[i]=value
@kernel
def load_snapshot(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        x=a[i]
        value=11+i32(constants.ZERO)
        if -1<atomic_add(count,i,1)<count[i]<2:
            value+=20
        out[i]=value
