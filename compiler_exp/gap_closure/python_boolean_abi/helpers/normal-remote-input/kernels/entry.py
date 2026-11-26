from typing import Final
from openassembler.pyfront import *
from library import helpers
from library import constants
raise RuntimeError("entry must never execute")
@kernel
def identity_value(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i32(helpers.identity(bool(x)))
        out[i]=result
@kernel
def invert_value(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i32(helpers.flip(bool(x)))
        out[i]=result
@kernel
def and_values(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i32(helpers.both(bool(x),x<3))
        out[i]=result
@kernel
def or_values(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i32(helpers.either(bool(x),x<3))
        out[i]=result
@kernel
def early_value(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i32(helpers.early(x<0,x))
        out[i]=result
@kernel
def caller_copy(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        flag=bool(x)
        result=i32(helpers.assign(flag))+i32(flag)*2
        out[i]=result
@kernel
def predicate_loop(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i32(helpers.toggle(bool(x),3))
        out[i]=result
@kernel
def loop_return_value(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i32(helpers.loop_return(bool(x),x))
        out[i]=result
@kernel
def nested_value(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i32(helpers.nested(bool(x)))
        out[i]=result
@kernel
def wide_pick(a:Array(i64),out:Array(i64),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=helpers.pick(bool(x),x)
        out[i]=result
@kernel
def float_select(a:Array(f32),out:Array(f32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=helpers.float_pick(bool(x),x)
        out[i]=result
@kernel
def wide_boolean(a:Array(i64),out:Array(i64),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i64(helpers.wide_truth(x))
        out[i]=result
@kernel
def unsigned_boolean(a:Array(u64),out:Array(u64),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=u64(helpers.unsigned_truth(x))
        out[i]=result
@kernel
def chain_boolean(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i32(helpers.chain(x))
        out[i]=result
@kernel
def literal_boolean(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i32(helpers.identity(True))+2*i32(helpers.flip(False))
        out[i]=result
@kernel
def eager_arguments(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i32(helpers.either(bool(atomic_add(count,i,1)),bool(atomic_add(count,i,1))))
        out[i]=result
@kernel
def lazy_call(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i32(bool(x) and helpers.flip(bool(atomic_add(count,i,1))))
        out[i]=result
@kernel
def argument_order(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i32(helpers.either(bool(atomic_add(count,i,1)),bool(count[i])))
        out[i]=result
@kernel
def caller_loop(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        flag=bool(x)
        step=0
        while step<4:
            flag=helpers.flip(flag)
            step+=1
        result=i32(flag)
        out[i]=result
@kernel
def wide_conversion(a:Array(u64),out:Array(u64),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=u64(helpers.flip(bool(x)))
        out[i]=result
