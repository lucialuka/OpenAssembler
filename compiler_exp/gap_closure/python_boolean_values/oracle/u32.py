from openassembler.pyfront import *
@kernel
def k(a:Array(u32),b:Array(u32),left:Array(u32),right:Array(u32),flags:Array(i32)):
    i=tid_x()
    conjunction=a[i] and b[i]
    disjunction=a[i] or b[i]
    inverse=not bool(a[i])
    left[i]=conjunction
    right[i]=disjunction
    flags[i]=i32(inverse)+i32(bool(conjunction))*2+i32(bool(disjunction))*4
