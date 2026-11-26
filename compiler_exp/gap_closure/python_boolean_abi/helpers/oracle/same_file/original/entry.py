from typing import Final
from openassembler.pyfront import *
ZERO:Final=0
raise RuntimeError("same file must never execute")
@device
def identity(flag: bool) -> bool:
    return flag
@device
def flip(flag: "bool") -> "bool":
    return not flag
@device
def both(first: bool, second: bool) -> bool:
    return first and second
@device
def either(first: bool, second: bool) -> bool:
    return first or second
@device
def early(flag: bool, x: i32) -> bool:
    if flag:
        return True
    if x < 0:
        return False
    return x < 7
@device
def assign(flag: bool) -> bool:
    flag = not flag
    return flag
@device
def toggle(flag: bool, trips: i32) -> bool:
    step = 0
    while step < trips:
        flag = not flag
        step += 1
    return flag
@device
def loop_return(flag: bool, x: i32) -> bool:
    for step in range(3):
        if flag:
            return not flag
        if x < step:
            return True
    return flag
@device
def nested(flag: bool) -> bool:
    return flip(identity(flag))
@device
def pick(flag: bool, x: i64) -> i64:
    if flag:
        return x + i64(3)
    return i64(-7)
@device
def float_pick(flag: bool, x: f32) -> f32:
    if flag:
        return x
    return -0.0
@device
def wide_truth(x: i64) -> bool:
    return bool(x)
@device
def unsigned_truth(x: u64) -> bool:
    return not x
@device
def chain(x: i32) -> bool:
    return -3 < x < 7
@kernel
def identity_value(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        x=a[i]
        result=i32(identity(bool(x)))
        out[i]=result
@kernel
def invert_value(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        x=a[i]
        result=i32(flip(bool(x)))
        out[i]=result
@kernel
def and_values(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        x=a[i]
        result=i32(both(bool(x),x<3))
        out[i]=result
@kernel
def or_values(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        x=a[i]
        result=i32(either(bool(x),x<3))
        out[i]=result
@kernel
def early_value(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        x=a[i]
        result=i32(early(x<0,x))
        out[i]=result
@kernel
def caller_copy(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        x=a[i]
        flag=bool(x)
        result=i32(assign(flag))+i32(flag)*2
        out[i]=result
@kernel
def predicate_loop(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        x=a[i]
        result=i32(toggle(bool(x),3))
        out[i]=result
@kernel
def loop_return_value(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        x=a[i]
        result=i32(loop_return(bool(x),x))
        out[i]=result
@kernel
def nested_value(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        x=a[i]
        result=i32(nested(bool(x)))
        out[i]=result
@kernel
def wide_pick(a:Array(i64),out:Array(i64),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        x=a[i]
        result=pick(bool(x),x)
        out[i]=result
@kernel
def float_select(a:Array(f32),out:Array(f32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        x=a[i]
        result=float_pick(bool(x),x)
        out[i]=result
@kernel
def wide_boolean(a:Array(i64),out:Array(i64),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        x=a[i]
        result=i64(wide_truth(x))
        out[i]=result
@kernel
def unsigned_boolean(a:Array(u64),out:Array(u64),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        x=a[i]
        result=u64(unsigned_truth(x))
        out[i]=result
@kernel
def chain_boolean(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        x=a[i]
        result=i32(chain(x))
        out[i]=result
@kernel
def literal_boolean(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        x=a[i]
        result=i32(identity(True))+2*i32(flip(False))
        out[i]=result
@kernel
def eager_arguments(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        x=a[i]
        result=i32(either(bool(atomic_add(count,i,1)),bool(atomic_add(count,i,1))))
        out[i]=result
@kernel
def lazy_call(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        x=a[i]
        result=i32(bool(x) and flip(bool(atomic_add(count,i,1))))
        out[i]=result
@kernel
def argument_order(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        x=a[i]
        result=i32(either(bool(atomic_add(count,i,1)),bool(count[i])))
        out[i]=result
@kernel
def caller_loop(a:Array(i32),out:Array(i32),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        x=a[i]
        flag=bool(x)
        step=0
        while step<4:
            flag=flip(flag)
            step+=1
        result=i32(flag)
        out[i]=result
@kernel
def wide_conversion(a:Array(u64),out:Array(u64),count:Array(i32),n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(ZERO)
        x=a[i]
        result=u64(flip(bool(x)))
        out[i]=result
