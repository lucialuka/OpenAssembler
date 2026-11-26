from typing import Final
from openassembler.pyfront import *
raise RuntimeError('static tuple loops must never execute user module code')
COEFFICIENTS:Final[tuple[int,...]]=(3,-5,7)
@device
def polynomial(x:i64)->i64:
    result=i64(0)
    for coefficient in COEFFICIENTS:
        result=result*x+i64(coefficient)
    return result
@kernel
def evaluate(a:Array(i64),out:Array(i64),n:i32):
    i=tid_x()
    if i<n:
        out[i]=polynomial(a[i])
