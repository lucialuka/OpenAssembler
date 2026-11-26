from openassembler.pyfront import *
raise RuntimeError('Boolean storage discovery must never execute the user module')


@kernel
def raw_array(a: Array(bool), out: Array(bool), flag: bool, n: i32):
    i = tid_x()
    if i < n:
        out[i] = a[i] and flag


@kernel
def shaped_buffer(a: Buffer(bool, (4, 7), access='read'), out: Buffer(bool, (4, 7)), flag: bool):
    i = tid_x()
    if i < 7:
        out[0, i] = a[0, i] or flag


@kernel
def shared_flags(a: Array(i32), out: Array(i32), flag: bool):
    i = tid_x()
    shared = shared_array(bool, 128)
    shared[i] = bool(a[i]) and flag
    syncthreads()
    out[i] = i32(shared[127 - i])
