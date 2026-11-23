from __future__ import annotations
from openassembler.pyfront import Array, Buffer, i32, i64, f32, kernel, tid_x

raise RuntimeError('must not execute top-level code')

@kernel
def transpose(a: Buffer(f32,(3,4),strides=(6,1),access='read'),
              out: Buffer(f32,(4,3),strides=(1,6)), coords: Array(i32)):
    lane = tid_x()
    row = coords[lane * 2]
    col = coords[lane * 2 + 1]
    out[col,row] = a[row,col] + 0.5

@kernel
def checked_wide(a: Buffer(i64,(8,),access='read',alignment=16),
                 out: Buffer(i64,(8,),alignment=16), coords: Array(i32), loads: Array(i64)):
    lane = tid_x()
    index = coords[lane]
    value = a[index]
    loads[lane] = value
    out[index] = value + i64(1)

@kernel
def broadcast(a: Buffer(i32,(3,4),strides=(0,1),access='read'), out: Buffer(i32,(3,4))):
    lane = tid_x()
    row = lane // 4
    col = lane % 4
    out[row,col] = a[row,col]
