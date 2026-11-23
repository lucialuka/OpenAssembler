from __future__ import annotations
from openassembler.pyfront import Array, i32, f32, device, kernel, tid_x

raise RuntimeError('must never execute module top-level code')

@device
def integer_control(x: i32, n: i32) -> i32:
    value = x
    total = 0
    for j in range(n, 0, -1):
        if j == 3:
            continue
        if j == 1:
            break
        total += j
    left = n
    while left > 0:
        left -= 1
        if left == 2:
            continue
        if left == 0:
            break
        total += left
    if value < 0:
        value = -value
    elif value == 0:
        value = 7
    else:
        value += 2
    return value + total

@device
def float_control(x: f32, n: i32) -> f32:
    value = x
    for j in range(n):
        if value < 0.0:
            value *= -0.5
        else:
            value += 0.25
    return value

@device
def nested(x: i32, n: i32) -> i32:
    y = integer_control(x, n)
    return integer_control(y, n) + x

@kernel
def evaluate(a: Array(i32), b: Array(f32), out: Array(i32), fo: Array(f32), n: i32):
    i = tid_x()
    if i >= n:
        return
    value = a[i]
    total = 99
    count = i % 6
    out[i] = nested(value, count) + integer_control(value, count) + value + total
    fo[i] = float_control(b[i], count) + b[i]
