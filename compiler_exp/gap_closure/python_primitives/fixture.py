from __future__ import annotations
from openassembler.pyfront import Array, Buffer, i32, u32, kernel, tid_x, ctaid_x, ntid_x, atomic_add, atomic_exchange, atomic_compare_exchange

raise RuntimeError('must not execute module')

@kernel
def add(values: Array(i32), state: Array(i32), returned: Array(i32), n: i32):
    i = ctaid_x() * ntid_x() + tid_x()
    if i < n:
        returned[i] = atomic_add(state, 0, values[i])

@kernel
def exchange(state: Array(u32), returned: Array(u32), n: i32):
    i = ctaid_x() * ntid_x() + tid_x()
    if i < n:
        returned[i] = atomic_exchange(state, 0, u32(i + 1))

@kernel
def compare_exchange(state: Array(u32), returned: Array(u32), n: i32):
    i = ctaid_x() * ntid_x() + tid_x()
    if i < n:
        returned[i] = atomic_compare_exchange(state, 0, u32(0), u32(i + 1))

@kernel
def checked(state: Buffer(i32,(8,)), coords: Array(i32), returned: Array(i32)):
    i = tid_x()
    returned[i] = atomic_add(state, coords[i], 2)

@kernel
def evaluation_order(state: Array(u32), out: Array(u32)):
    if tid_x() == 0:
        out[0] = atomic_add(state, i32(atomic_exchange(state, 0, u32(0))), atomic_exchange(state, 1, u32(7)))
        atomic_add(state, 2, u32(1))

@kernel
def checked_exchange(state: Buffer(u32,(8,)), coords: Array(i32), returned: Array(u32)):
    i = tid_x()
    returned[i] = atomic_exchange(state, coords[i], 7)

@kernel
def checked_compare_exchange(state: Buffer(u32,(8,)), coords: Array(i32), returned: Array(u32)):
    i = tid_x()
    returned[i] = atomic_compare_exchange(state, coords[i], 10, 9)
