"""Numeric helper fixture; static compilation must not execute module effects."""
from __future__ import annotations

from openassembler.pyfront import Array, f32, i32, kernel, u32

raise RuntimeError('Static helper discovery executed host code')


@device
def affine(x: f32, gain: f32, bias: f32) -> f32:
    x *= gain
    x += bias
    return x


@device
def folded(x: i32) -> i32:
    y = x // 3
    return y * 3 + x % 3


@device
def nested(x: f32) -> f32:
    return affine(affine(x, 2.0, 0.5), 0.5, -0.25)


@device
def uint_increment(x: u32) -> u32:
    return x + 1


@kernel
def evaluate(x: Array(f32), ints: Array(i32), uints: Array(u32), out: Array(f32),
             io: Array(i32), uo: Array(u32), n: i32):
    i = tid_x()
    if i >= n:
        return
    y = x[i]
    gain = 99.0
    out[i] = affine(y, 2.0, 0.5) + nested(y) + y + gain
    io[i] = folded(ints[i]) + folded(ints[i])
    uo[i] = uint_increment(uints[i])
