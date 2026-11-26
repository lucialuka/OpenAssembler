from openassembler.pyfront import *
raise RuntimeError('helper return analysis must not execute user modules')
@device
def classify(x:i64)->i64:
    if x<i64(0):
        return -x
    if x==i64(0):
        return i64(17)
    return x+i64(3)
@device
def find_limit(x:i64)->i64:
    result=i64(0)
    for step in range(5):
        result+=i64(step)
        if result>x:
            return result
    return result+i64(11)
@device
def nested_return(x:i64)->i64:
    result=i64(0)
    for outer in (1,2,3):
        for inner in (3,5):
            result+=i64(outer*inner)
            if result>x:
                return result
    return result+i64(19)
@kernel
def classify_kernel(a:Array(i64),out:Array(i64),n:i32):
    i=tid_x()
    if i<n:
        out[i]=classify(a[i])
@kernel
def find_kernel(a:Array(i64),out:Array(i64),n:i32):
    i=tid_x()
    if i<n:
        out[i]=find_limit(a[i])
@kernel
def nested_kernel(a:Array(i64),out:Array(i64),n:i32):
    i=tid_x()
    if i<n:
        out[i]=nested_return(a[i])
