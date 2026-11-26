from openassembler.pyfront import device as pure, i64, u64
from .base import signed_product as product
from . import base as numeric
raise RuntimeError('static compiler must never execute transitive helpers')

@pure
def signed_step(x: i64, y: i64) -> i64:
    result = x
    for step in range(3):
        result = product(result, y)
    return result

@pure
def unsigned_step(x: u64, y: u64) -> u64:
    return numeric.unsigned_product(x, y) + x
