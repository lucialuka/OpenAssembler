from library.helpers import choose

from openassembler.pyfront import *
raise RuntimeError("Boolean storage source discovery must not execute")

@kernel
def copy(a: Array(bool), out: Array(bool), counter: Array(i32), flag: bool, n: i32):
    t = tid_x()
    i = t + ctaid_x() * ntid_x()
    if i < n:
        out[i] = a[i]

@kernel
def invert(a: Array(bool), out: Array(bool), counter: Array(i32), flag: bool, n: i32):
    t = tid_x()
    i = t + ctaid_x() * ntid_x()
    if i < n:
        out[i] = not a[i]

@kernel
def lazy_and(a: Array(bool), out: Array(bool), counter: Array(i32), flag: bool, n: i32):
    t = tid_x()
    i = t + ctaid_x() * ntid_x()
    if i < n:
        out[i] = a[i] and bool(atomic_add(counter, i, 1))

@kernel
def lazy_or(a: Array(bool), out: Array(bool), counter: Array(i32), flag: bool, n: i32):
    t = tid_x()
    i = t + ctaid_x() * ntid_x()
    if i < n:
        out[i] = a[i] or bool(atomic_add(counter, i, 1))

@kernel
def helper(a: Array(bool), out: Array(bool), counter: Array(i32), flag: bool, n: i32):
    t = tid_x()
    i = t + ctaid_x() * ntid_x()
    if i < n:
        out[i] = choose(a[i], flag)

@kernel
def equal_flag(a: Array(bool), out: Array(bool), counter: Array(i32), flag: bool, n: i32):
    t = tid_x()
    i = t + ctaid_x() * ntid_x()
    if i < n:
        out[i] = i32(a[i]) == i32(flag)

@kernel
def atomic_truth(a: Array(bool), out: Array(bool), counter: Array(i32), flag: bool, n: i32):
    t = tid_x()
    i = t + ctaid_x() * ntid_x()
    if i < n:
        out[i] = bool(atomic_add(counter, i, 1))

@kernel
def constants(a: Array(bool), out: Array(bool), counter: Array(i32), flag: bool, n: i32):
    t = tid_x()
    i = t + ctaid_x() * ntid_x()
    if i < n:
        if i % 2 == 0:
            out[i] = True
        else:
            out[i] = False

@kernel
def predicate_loop(a: Array(bool), out: Array(bool), counter: Array(i32), flag: bool, n: i32):
    t = tid_x()
    i = t + ctaid_x() * ntid_x()
    if i < n:
        value = a[i]
        for j in range(3):
            value = not value
        out[i] = value

@kernel
def alias_invert(a: Array(bool), out: Array(bool), counter: Array(i32), flag: bool, n: i32):
    t = tid_x()
    i = t + ctaid_x() * ntid_x()
    if i < n:
        out[i] = not a[i]

@kernel
def buffer_padded(a: Buffer(bool, (4, 7), strides=(11, 1), access='read'), out: Buffer(bool, (4, 7), strides=(9, 1)), counter: Array(i32), flag: bool, n: i32):
    t = tid_x()
    i = t + ctaid_x() * ntid_x()
    if i < n:
        row = i // 7
        col = i % 7
        out[row, col] = a[row, col] or flag

@kernel
def buffer_transpose(a: Buffer(bool, (4, 7), strides=(1, 5), access='read'), out: Buffer(bool, (4, 7)), counter: Array(i32), flag: bool, n: i32):
    t = tid_x()
    i = t + ctaid_x() * ntid_x()
    if i < n:
        row = i // 7
        col = i % 7
        out[row, col] = a[row, col] and flag

@kernel
def buffer_broadcast(a: Buffer(bool, (4, 7), strides=(0, 1), access='read'), out: Buffer(bool, (4, 7)), counter: Array(i32), flag: bool, n: i32):
    t = tid_x()
    i = t + ctaid_x() * ntid_x()
    if i < n:
        row = i // 7
        col = i % 7
        out[row, col] = not a[row, col]

@kernel
def buffer_bounds(a: Buffer(bool, (4, 7), access='read'), out: Buffer(bool, (4, 7)), counter: Array(i32), flag: bool, n: i32):
    t = tid_x()
    i = t + ctaid_x() * ntid_x()
    if i < n:
        row = i // 7 - 1
        col = i % 7
        out[row, col] = a[row, col] or flag

@kernel
def shared_reverse(a: Array(bool), out: Array(bool), counter: Array(i32), flag: bool, n: i32):
    t = tid_x()
    i = t + ctaid_x() * ntid_x()
    s = shared_array(bool, 128)
    s[t] = a[i]
    syncthreads()
    if i < n:
        out[i] = s[127 - t]

@kernel
def shared_mixed(a: Array(bool), out: Array(bool), counter: Array(i32), flag: bool, n: i32):
    t = tid_x()
    i = t + ctaid_x() * ntid_x()
    pad = shared_array(bool, 3)
    wide = shared_array(i64, 128)
    s = shared_array(bool, 128)
    if t < 3:
        pad[t] = flag
    wide[t] = i64(t)
    s[t] = a[i]
    syncthreads()
    if i < n:
        out[i] = s[127 - t] and (wide[127 - t] >= 0)

@kernel
def shared_broadcast(a: Array(bool), out: Array(bool), counter: Array(i32), flag: bool, n: i32):
    t = tid_x()
    i = t + ctaid_x() * ntid_x()
    s = shared_array(bool, 1)
    if t == 0:
        s[0] = flag
    syncthreads()
    if i < n:
        out[i] = s[0]

@kernel
def shared_loop(a: Array(bool), out: Array(bool), counter: Array(i32), flag: bool, n: i32):
    t = tid_x()
    i = t + ctaid_x() * ntid_x()
    s = shared_array(bool, 128)
    s[t] = a[i]
    syncthreads()
    value = s[(t + 65) % 128]
    for j in range(3):
        value = not value
    if i < n:
        out[i] = value
