from __future__ import annotations
from openassembler.pyfront import Array, i32, u32, i64, u64, device, kernel, tid_x

raise RuntimeError('static discovery must not execute this module')

@device
def accumulate(x: i64, y: i64, count: i32) -> i64:
    value = x
    for j in range(count):
        if value < y:
            value += y
        else:
            value -= y
    return value

@kernel
def signed(a: Array(i64), b: Array(i64), out: Array(i64), flags: Array(i32), offset: i64, n: i32):
    i = tid_x()
    if i >= n:
        return
    x = a[i]
    y = b[i]
    base = i * 8
    out[base] = x + y
    out[base + 1] = x - y
    out[base + 2] = x & y
    out[base + 3] = x | y
    out[base + 4] = x ^ y
    out[base + 5] = ~x
    out[base + 6] = -x + offset
    out[base + 7] = accumulate(x, y, i % 4)
    mask = 0
    if x < y:
        mask += 1
    if x <= y:
        mask += 2
    if x > y:
        mask += 4
    if x >= y:
        mask += 8
    if x == y:
        mask += 16
    if x != y:
        mask += 32
    if x:
        mask += 64
    flags[i] = mask

@kernel
def unsigned(a: Array(u64), b: Array(u64), out: Array(u64), narrow: Array(i32), offset: u64, n: i32):
    i = tid_x()
    if i >= n:
        return
    x = a[i]
    y = b[i]
    base = i * 6
    out[base] = x + y + offset
    out[base + 1] = x - y
    out[base + 2] = x & y
    out[base + 3] = x | y
    out[base + 4] = x ^ y
    out[base + 5] = ~x
    mask = 0
    if x < y:
        mask += 1
    if x <= y:
        mask += 2
    if x > y:
        mask += 4
    if x >= y:
        mask += 8
    if x == y:
        mask += 16
    if x != y:
        mask += 32
    if x:
        mask += 64
    narrow[i * 2] = i32(x)
    narrow[i * 2 + 1] = mask

@kernel
def shifts(a: Array(i64), b: Array(u64), out: Array(i64), uout: Array(u64), n: i32):
    i = tid_x()
    if i >= n:
        return
    x = a[i]
    y = b[i]
    out[i * 8 + 0] = x >> 0
    out[i * 8 + 1] = x >> 1
    out[i * 8 + 2] = x >> 31
    out[i * 8 + 3] = x >> 32
    out[i * 8 + 4] = x >> 33
    out[i * 8 + 5] = x >> 63
    out[i * 8 + 6] = x << 32
    out[i * 8 + 7] = x << 63
    uout[i * 8 + 0] = y >> 0
    uout[i * 8 + 1] = y >> 1
    uout[i * 8 + 2] = y >> 31
    uout[i * 8 + 3] = y >> 32
    uout[i * 8 + 4] = y >> 33
    uout[i * 8 + 5] = y >> 63
    uout[i * 8 + 6] = y << 32
    uout[i * 8 + 7] = y << 63

@kernel
def casts(a: Array(i32), b: Array(u32), wide: Array(i64), uwide: Array(u64), n: i32):
    i = tid_x()
    if i >= n:
        return
    x = a[i]
    y = b[i]
    wide[i * 3] = i64(x)
    wide[i * 3 + 1] = i64(y)
    wide[i * 3 + 2] = i64(u64(-1))
    uwide[i * 3] = u64(x)
    uwide[i * 3 + 1] = u64(y)
    uwide[i * 3 + 2] = u64(0x123456789abcdef01)


@kernel
def conversion_snapshots(a: Array(i32), b: Array(i64), out: Array(i64), n: i32):
    i = tid_x()
    if i >= n:
        return
    x = a[i]
    before = i64(x)
    x += 1
    out[i * 3] = before
    z = b[i]
    narrow = i32(z)
    same = i64(z)
    z += 1
    out[i * 3 + 1] = i64(narrow)
    out[i * 3 + 2] = same
