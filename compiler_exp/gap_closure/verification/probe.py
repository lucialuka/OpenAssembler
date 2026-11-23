"""G01 exact-reference and frozen-sequence diagnostic controls (no timing)."""
from fractions import Fraction
import random


COUNTEREXAMPLE = (0x8147EC48, 0x21DEA823, 0x2BAE9E99, 0xC5BAE70C)


def exact_division(a, b):
    """Independent integer-ratio binary32 RNE; no emulator/interpreter math."""
    sign = (a ^ b) & 0x80000000
    def ratio(x):
        exponent = (x >> 23) & 255
        mantissa = (x & 0x7fffff) | (0x800000 if exponent else 0)
        shift = (exponent - 150) if exponent else -149
        return Fraction(mantissa) * Fraction(2) ** shift
    if (a & 0x7fffffff) >= 0x7f800000 or (b & 0x7fffffff) >= 0x7f800000 or not (b & 0x7fffffff):
        raise ValueError('probe reference accepts finite operands and nonzero denominator')
    q = ratio(a) / ratio(b)
    if not q:
        return sign
    exponent = q.numerator.bit_length() - q.denominator.bit_length()
    if q < Fraction(2) ** exponent:
        exponent -= 1
    if exponent > 127:
        return sign | 0x7f800000
    scaled = q / Fraction(2) ** max(exponent - 23, -149)
    mantissa, remainder = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * remainder
    mantissa += twice > scaled.denominator or (twice == scaled.denominator and mantissa & 1)
    if exponent < -126:
        return sign | mantissa
    if mantissa == 0x1000000:
        exponent += 1
        mantissa >>= 1
    if exponent > 127:
        return sign | 0x7f800000
    return sign | ((exponent + 127) << 23) | (mantissa & 0x7fffff)


def inputs():
    """Frozen failure, adjacent bits, exponent sweep and reproducible finite fuzz."""
    a, b, c, d = COUNTEREXAMPLE
    rows = [COUNTEREXAMPLE]
    rows += [(a + da, b + db, c, d) for da in (-1, 0, 1) for db in (-1, 0, 1)]
    for ae in range(1, 255):
        for be in (1, 32, 64, 96, 127, 160, 192, 224, 254):
            rows.append(((a & 0x807fffff) | (ae << 23), (b & 0x807fffff) | (be << 23), c, d))
    rng = random.Random(0x473031)
    for _ in range(1024):
        aa, bb = rng.getrandbits(32), rng.getrandbits(32)
        if (aa & 0x7fffffff) < 0x7f800000 and 0 < (bb & 0x7fffffff) < 0x7f800000:
            rows.append((aa, bb, c, d))
    # Held-out small-dividend boundary controls, including both signs and the
    # minimum/maximum mantissas. Added after the initial isolation campaign.
    for exponent in (-104, -103, -102, -101, -100):
        for mantissa in (0, 1, 0x47ec48, 0x7ffffe, 0x7fffff):
            for sign in (0, 0x80000000):
                for denominator in (0x21dea823, 0x3fdea823):
                    rows.append((sign | ((exponent + 127) << 23) | mantissa, denominator, c, d))
    return rows
