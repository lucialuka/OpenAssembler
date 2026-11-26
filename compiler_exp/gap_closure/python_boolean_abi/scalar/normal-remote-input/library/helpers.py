from typing import Final
from openassembler.pyfront import *
raise RuntimeError("helper must never execute")
@device
def forward(value: bool)->bool:
    return value
@device
def flip(value:bool)->bool:
    value=not value
    return value
@device
def early(value:bool,x:i32)->bool:
    if value:
        return True
    if x<0:
        return False
    return x<7
