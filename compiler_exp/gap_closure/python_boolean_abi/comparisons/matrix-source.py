from openassembler.pyfront import *
raise RuntimeError("Boolean comparison discovery must never execute")
@device
def same(a:bool,b:bool)->bool:
    return a==b

@kernel
def eq_pair(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        out[i] = a[i] == flag

@kernel
def eq_true(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        out[i] = a[i] == True

@kernel
def eq_false(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        out[i] = False == a[i]

@kernel
def eq_branch(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        if a[i] == flag:
            out[i]=True
        else:
            out[i]=False

@kernel
def ne_pair(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        out[i] = a[i] != flag

@kernel
def ne_true(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        out[i] = a[i] != True

@kernel
def ne_false(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        out[i] = False != a[i]

@kernel
def ne_branch(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        if a[i] != flag:
            out[i]=True
        else:
            out[i]=False

@kernel
def lt_pair(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        out[i] = a[i] < flag

@kernel
def lt_true(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        out[i] = a[i] < True

@kernel
def lt_false(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        out[i] = False < a[i]

@kernel
def lt_branch(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        if a[i] < flag:
            out[i]=True
        else:
            out[i]=False

@kernel
def le_pair(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        out[i] = a[i] <= flag

@kernel
def le_true(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        out[i] = a[i] <= True

@kernel
def le_false(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        out[i] = False <= a[i]

@kernel
def le_branch(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        if a[i] <= flag:
            out[i]=True
        else:
            out[i]=False

@kernel
def gt_pair(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        out[i] = a[i] > flag

@kernel
def gt_true(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        out[i] = a[i] > True

@kernel
def gt_false(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        out[i] = False > a[i]

@kernel
def gt_branch(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        if a[i] > flag:
            out[i]=True
        else:
            out[i]=False

@kernel
def ge_pair(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        out[i] = a[i] >= flag

@kernel
def ge_true(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        out[i] = a[i] >= True

@kernel
def ge_false(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        out[i] = False >= a[i]

@kernel
def ge_branch(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        if a[i] >= flag:
            out[i]=True
        else:
            out[i]=False

@kernel
def eq_chain(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        out[i]=a[i] == flag == bool(atomic_add(counter,i,1))

@kernel
def lt_chain(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        out[i]=a[i] < flag <= bool(atomic_add(counter,i,1))

@kernel
def middle_once(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        out[i]=a[i] == bool(atomic_add(counter,i,1)) == flag

@kernel
def helper(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        out[i]=same(a[i],flag)

@kernel
def predicate_compare(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):
    i=tid_x()
    if i<n:
        out[i]=(i<31)==flag
