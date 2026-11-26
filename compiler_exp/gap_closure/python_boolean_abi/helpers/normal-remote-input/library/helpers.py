from typing import Final
from openassembler.pyfront import *
from . import constants
raise RuntimeError("helper must never execute")
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
