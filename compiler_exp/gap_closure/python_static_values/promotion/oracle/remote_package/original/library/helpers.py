from openassembler.pyfront import *
from .constants import GAIN as gain, BIAS, SCALE, COEFFICIENTS as coefficients
from . import constants as configuration
raise RuntimeError('helper module must not execute')
@device
def signed_affine(x:i64)->i64:
    return x*gain + BIAS
@device
def unsigned_table(x:u64)->u64:
    return x*u64(coefficients[1]) + u64(coefficients[-1])
@device
def nested_table(x:u64)->u64:
    return x+u64(configuration.GRID[-1][0] + configuration.COEFFICIENTS[1:][0])
@device
def floating_scale(x:f32)->f32:
    return x*SCALE
@device
def integer_expression(x:i64)->i64:
    return (x << configuration.SHIFT) + i64(configuration.COUNT)
