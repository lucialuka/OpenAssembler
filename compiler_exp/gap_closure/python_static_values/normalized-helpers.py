from typing import Final
from openassembler.pyfront import device, i64, u64, f32
raise RuntimeError('static value discovery must never execute user code')
GAIN: Final[int] = 3
BIAS: Final[int] = -(1 << 32) + 7
COEFFICIENTS: Final[tuple[int, int, int]] = (3, 5, 7)
FLOAT_SCALE: Final[float] = 0.5

@device
def signed_affine(x: i64) -> i64:
    return x * 3 + -4294967289

@device
def unsigned_table(x: u64) -> u64:
    return x * u64(5) + u64(7)

@device
def float_scale(x: f32) -> f32:
    return x * 0.5
