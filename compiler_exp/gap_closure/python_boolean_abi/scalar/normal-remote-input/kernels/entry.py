from typing import Final
from openassembler.pyfront import *
from library import helpers
from library import constants
raise RuntimeError("entry must never execute")
@kernel
def forward_flag(first:bool,second:bool,a:Array(i32),out:Array(i32),counter:Array(i32),neighbor:i32,wide:i64,n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i32(first)
        out[i]=result
@kernel
def second_flag(a:Array(i32),out:Array(i32),counter:Array(i32),first:bool,second:bool,neighbor:i32,wide:i64,n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i32(second)
        out[i]=result
@kernel
def both_flags(a:Array(i32),out:Array(i32),counter:Array(i32),first:bool,neighbor:i32,second:bool,wide:i64,n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i32(first and second)
        out[i]=result
@kernel
def either_flag(first:bool,second:bool,a:Array(i32),out:Array(i32),counter:Array(i32),neighbor:i32,wide:i64,n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i32(first or second)
        out[i]=result
@kernel
def not_flag(a:Array(i32),out:Array(i32),counter:Array(i32),first:bool,second:bool,neighbor:i32,wide:i64,n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i32(not first)
        out[i]=result
@kernel
def local_reassign(a:Array(i32),out:Array(i32),counter:Array(i32),first:bool,neighbor:i32,second:bool,wide:i64,n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        first=not first
        result=i32(first)+2*i32(second)
        out[i]=result
@kernel
def helper_copy(first:bool,second:bool,a:Array(i32),out:Array(i32),counter:Array(i32),neighbor:i32,wide:i64,n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i32(helpers.flip(first))+2*i32(first)
        out[i]=result
@kernel
def early_helper(a:Array(i32),out:Array(i32),counter:Array(i32),first:bool,second:bool,neighbor:i32,wide:i64,n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i32(helpers.early(first,x))
        out[i]=result
@kernel
def predicate_loop(a:Array(i32),out:Array(i32),counter:Array(i32),first:bool,neighbor:i32,second:bool,wide:i64,n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        flag=first
        step=0
        while step<3:
            flag=not flag
            step+=1
        result=i32(flag)
        out[i]=result
@kernel
def branch_neighbor(first:bool,second:bool,a:Array(i32),out:Array(i32),counter:Array(i32),neighbor:i32,wide:i64,n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=neighbor
        if first:
            result=x
        if second:
            result+=neighbor
        out[i]=result
@kernel
def wide_select(a:Array(i64),out:Array(i64),counter:Array(i32),first:bool,second:bool,neighbor:i32,wide:i64,n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=wide
        if first:
            result=x
        if second:
            result+=i64(neighbor)
        out[i]=result
@kernel
def wide_flags(a:Array(i64),out:Array(i64),counter:Array(i32),first:bool,neighbor:i32,second:bool,wide:i64,n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i64(first)+2*i64(second)+wide
        out[i]=result
@kernel
def unsigned_flags(first:bool,second:bool,a:Array(u64),out:Array(u64),counter:Array(i32),neighbor:i32,wide:i64,n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=u64(first)+2*u64(second)
        out[i]=result
@kernel
def float_select(a:Array(f32),out:Array(f32),counter:Array(i32),first:bool,second:bool,neighbor:i32,wide:i64,n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=-0.0
        if first:
            result=x
        out[i]=result
@kernel
def float_flags(a:Array(f32),out:Array(f32),counter:Array(i32),first:bool,neighbor:i32,second:bool,wide:i64,n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=float(i32(first and not second))
        out[i]=result
@kernel
def lazy_and(first:bool,second:bool,a:Array(i32),out:Array(i32),counter:Array(i32),neighbor:i32,wide:i64,n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i32(first and bool(atomic_add(counter,i,1)))
        out[i]=result
@kernel
def lazy_or(a:Array(i32),out:Array(i32),counter:Array(i32),first:bool,second:bool,neighbor:i32,wide:i64,n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i32(first or bool(atomic_add(counter,i,1)))
        out[i]=result
@kernel
def lazy_nested(a:Array(i32),out:Array(i32),counter:Array(i32),first:bool,neighbor:i32,second:bool,wide:i64,n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i32((first and bool(atomic_add(counter,i,1)))or(second and bool(atomic_add(counter,i,1))))
        out[i]=result
@kernel
def forward_helper(first:bool,second:bool,a:Array(i32),out:Array(i32),counter:Array(i32),neighbor:i32,wide:i64,n:i32):
    i=ctaid_x()*128+tid_x()
    if i<n:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i32(helpers.forward(second))
        out[i]=result
@kernel
def dimension_name(a:Array(i32),out:Array(i32),counter:Array(i32),first:bool,n:bool,second:bool,neighbor:i32,wide:i64,count:i32):
    i=ctaid_x()*128+tid_x()
    if i<count:
        unused=i32(constants.ZERO)
        x=a[i]
        result=i32(n)+i32(second)*2
        out[i]=result
