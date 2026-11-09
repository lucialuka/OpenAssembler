"""Differential test for openassembler.pyfront: the SAME kernel body is run three ways and
must agree (see tests/pyfront_threeway.py) --
  1. as plain Python (one real thread per GPU thread, the language's real semantics),
  2. as the PTX pyfront emitted, on the reference PTX interpreter,
  3. as the SASS our backend generated, on the SASS emulator.

`compile --verify` only compares (2) with (3), so a bug in the Python -> PTX step (a dropped
`return`, a loop that runs the wrong number of times, truncating instead of floor division)
passes it. The first cases are the silent miscompiles found by auditing the docs; the rest
cover each construct in the module docstring of openassembler/pyfront/compile.py, with the
inputs where Python and naive GPU code disagree (negative operands, NaN, -0.0, denormals).

Leg (3) is skipped with "backend support pending" when our backend/encoder can't compile
the PTX yet (div.rn.f32, sqrt, cvt, ...); legs (1) == (2) must always pass."""

from __future__ import annotations

import math

import pytest
from pyfront_threeway import Pending, close, f32, run_ptx_and_sass, run_python, same

from openassembler.pyfront import (Array, exp, f32 as F32, i32, kernel, log, rsqrt, shared_array, sqrt,
                                   syncthreads, tanh, tid_x)

THREADS = 8
I32_MIN = -(1 << 31)
I32_MAX = (1 << 31) - 1


# ---- the original miscompile regressions -------------------------------------------------

@kernel
def early_return(a: Array(i32), n: i32):
    i = tid_x()
    if i >= n:
        return
    a[i] = i + 1


@kernel
def bound_snapshot(o: Array(i32), n: i32):
    i = tid_x()
    m = n
    s = 0
    for j in range(0, m):
        m = 100
        s += 1
    o[i] = s


@kernel
def neg_step(o: Array(i32), n: i32):
    i = tid_x()
    s = 0
    for j in range(n, 0, -2):
        s += j
    o[i] = s


@kernel
def pos_step(o: Array(i32), n: i32):
    i = tid_x()
    s = 0
    for j in range(1, n, 3):
        s += j
    o[i] = s


@kernel
def empty_loop(o: Array(i32), n: i32):
    i = tid_x()
    s = 7
    for j in range(0, n):
        s += 1
    o[i] = s


@kernel
def elif_chain(a: Array(i32), o: Array(i32), n: i32):
    i = tid_x()
    v = a[i]
    if v < 0:
        o[i] = 0 - 1
    elif v < 10:
        o[i] = v
    elif v < 100:
        o[i] = v * 2
    else:
        o[i] = 1000


@kernel
def and_or(a: Array(i32), o: Array(i32), n: i32):
    i = tid_x()
    if i > 1 and i < 6:
        o[i] = a[i]
    if i < 1 or i > 6:
        o[i] = 0 - 5


@kernel
def nested_loops(o: Array(i32), n: i32):
    i = tid_x()
    s = 0
    for x in range(0, n):
        for y in range(x, n):
            s += x * y + i
    o[i] = s


@kernel
def float_literals(a: Array(F32), o: Array(F32), n: i32):
    i = tid_x()
    o[i] = a[i] * 2.0 + 1.5
    if i < 4:
        o[i] = o[i] - 0.25


@kernel
def float_lhs_literal(a: Array(F32), o: Array(F32), n: i32):
    i = tid_x()
    o[i] = 0.5 - a[i] * 2.0


@kernel
def float_branch(a: Array(F32), o: Array(F32), n: i32):
    i = tid_x()
    v = a[i]
    if v > 0.5:
        o[i] = v * 4.0
    else:
        o[i] = v + 8.0


# ---- 1. / // % ---------------------------------------------------------------------------

@kernel
def truediv_int(a: Array(i32), b: Array(i32), o: Array(F32), n: i32):
    i = tid_x()
    o[i] = a[i] / b[i]                       # true division: f32 even for ints


@kernel
def truediv_f32(a: Array(F32), b: Array(F32), o: Array(F32), p: Array(F32), q: Array(F32), n: i32):
    i = tid_x()
    o[i] = a[i] / b[i]
    p[i] = a[i] / 4.0                        # power of two: becomes an exact multiply
    q[i] = a[i] / 3                          # int literal divisor, f32 result


@kernel
def floordiv_mod_reg(a: Array(i32), b: Array(i32), q: Array(i32), r: Array(i32), n: i32):
    i = tid_x()
    q[i] = a[i] // b[i]
    r[i] = a[i] % b[i]


@kernel
def floordiv_mod_const(a: Array(i32), q3: Array(i32), r3: Array(i32), qm3: Array(i32), rm3: Array(i32),
                       q4: Array(i32), r4: Array(i32), qm4: Array(i32), rm1: Array(i32), n: i32):
    i = tid_x()
    v = a[i]
    q3[i] = v // 3
    r3[i] = v % 3
    qm3[i] = v // -3
    rm3[i] = v % -3
    q4[i] = v // 4                           # power of two: shr.s32
    r4[i] = v % 4                            # power of two: and.b32
    qm4[i] = v // -4
    rm1[i] = v % -1
    v //= 7                                  # augmented
    a[i] = v


@kernel
def floordiv_one(a: Array(i32), q: Array(i32), m: Array(i32), n: i32):
    i = tid_x()
    q[i] = a[i] // 1 + a[i] // -1
    m[i] = a[i] % 1


# ---- 2. float() / int() ------------------------------------------------------------------

@kernel
def casts(a: Array(i32), x: Array(F32), fo: Array(F32), io: Array(i32), n: i32):
    i = tid_x()
    fo[i] = float(a[i]) * 0.5
    io[i] = int(x[i])
    if i == 0:
        fo[i] = float(7) + float(i)
        io[i] = int(-2.75)


# ---- 3. not ------------------------------------------------------------------------------

@kernel
def not_cond(a: Array(i32), o: Array(i32), f: Array(F32), n: i32):
    i = tid_x()
    v = a[i]
    r = 0
    if not v < 3:
        r += 1
    if not (v > 1 and v < 6):
        r += 10
    if not (v == 0 or not v != 7):
        r += 100
    if not f[i] != f[i]:                     # NaN != NaN is True in Python
        r += 1000
    if f[i]:                                 # truthiness: NaN is true, -0.0 is false
        r += 10000
    if not v:
        r += 100000
    o[i] = r


# ---- 4. min / max / abs ------------------------------------------------------------------

@kernel
def minmaxabs_i(a: Array(i32), b: Array(i32), mn: Array(i32), mx: Array(i32), ab: Array(i32), n: i32):
    i = tid_x()
    mn[i] = min(a[i], b[i])
    mx[i] = max(a[i], b[i], 0 - 3)
    ab[i] = abs(a[i]) + min(2, b[i])


@kernel
def minmaxabs_f(a: Array(F32), b: Array(F32), mn: Array(F32), mx: Array(F32), ab: Array(F32), n: i32):
    i = tid_x()
    mn[i] = min(a[i], b[i])
    mx[i] = max(a[i], b[i])
    ab[i] = abs(a[i])


# ---- 5. sqrt (exact) and the approximate intrinsics ----------------------------------------

@kernel
def sqrt_k(x: Array(F32), o: Array(F32), m: Array(F32), k: Array(F32), n: i32):
    i = tid_x()
    o[i] = sqrt(x[i])
    m[i] = math.sqrt(abs(x[i]))               # math.sqrt (import math)
    k[i] = sqrt(i)                            # an i32 argument is converted first
    # (not summed: a chain rounds to f32 after every step, Python's float64 does not)


@kernel
def approx_k(x: Array(F32), e: Array(F32), lg: Array(F32), th: Array(F32), rs: Array(F32),
             me: Array(F32), n: i32):
    i = tid_x()
    v = x[i]
    e[i] = exp(v)
    lg[i] = log(abs(v) + 0.5)
    th[i] = tanh(v)
    rs[i] = rsqrt(abs(v) + 0.25)
    me[i] = math.exp(v) + math.log(abs(v) + 1.0) + math.tanh(v)


# ---- 6. while / break / continue -----------------------------------------------------------

@kernel
def collatz(a: Array(i32), o: Array(i32), n: i32):
    i = tid_x()
    v = a[i]
    steps = 0
    while v != 1:
        if v % 2 == 0:
            v = v // 2
        else:
            v = 3 * v + 1
        steps += 1
        if steps >= 50:
            break
    o[i] = steps


@kernel
def while_true_continue(o: Array(i32), n: i32):
    i = tid_x()
    k = 0
    s = 0
    while True:
        k += 1
        if k > n + i:
            break
        if k % 3 == 0:
            continue
        s += k
    o[i] = s


@kernel
def for_break_continue(o: Array(i32), last: Array(i32), n: i32):
    i = tid_x()
    s = 0
    for j in range(0, n):
        if j == i:
            continue
        if j > i + 3:
            break
        for k in range(0, j):
            if k == 2:
                break
            s += k * 10
        s += j
    o[i] = s
    j = 0 - 1
    for j in range(i, n, 2):
        pass
    last[i] = j                       # Python: the last value taken (or -1 if empty), not `n`


@kernel
def loop_var_assigned(o: Array(i32), n: i32):
    i = tid_x()
    s = 0
    for j in range(0, n):
        j = j * 10                    # Python rebinds j next iteration: still n iterations
        s += j
    o[i] = s + j


# ---- 7. shared memory + syncthreads ---------------------------------------------------------

@kernel
def smem_reverse(a: Array(F32), o: Array(F32), n: i32):
    s = shared_array(F32, 32)
    t = tid_x()
    s[t] = a[t] * 2.0
    syncthreads()
    o[t] = s[n - 1 - t] + s[0]


@kernel
def smem_reduce(a: Array(F32), o: Array(F32), cnt: Array(i32), n: i32):
    s = shared_array(F32, 32)
    c = shared_array(i32, 32)
    t = tid_x()
    s[t] = a[t]
    c[t] = 1
    syncthreads()
    step = n // 2
    while step > 0:
        if t < step:
            s[t] += s[t + step]
            c[t] = c[t] + c[t + step]
        syncthreads()
        step //= 2
    if t == 0:
        o[0] = s[0]
        cnt[0] = c[0]


N = [-7, 7, -8, 8, 0, -1, I32_MIN + 1, 2147483647]
D = [2, -2, 3, -3, 5, 1, -7, 2]
FX = [0.0, -0.0, 1.5e-45, -4.0, 2.0, math.inf, 1e-40, 9.0]       # sqrt edge inputs
FA = [-0.0, 0.0, math.nan, 1.0, -2.5, math.nan, 3.0, -1.0]
FB = [0.0, -0.0, 1.0, math.nan, -2.5, math.nan, 2.0, -1.5]

CASES = [
    ("early_return", early_return, {"a": [-1] * THREADS}, {"n": 3}),
    ("early_return_none", early_return, {"a": [-1] * THREADS}, {"n": 8}),
    ("bound_snapshot", bound_snapshot, {"o": [0] * THREADS}, {"n": 5}),
    ("neg_step", neg_step, {"o": [0] * THREADS}, {"n": 9}),
    ("neg_step_empty", neg_step, {"o": [0] * THREADS}, {"n": 0}),
    ("pos_step", pos_step, {"o": [0] * THREADS}, {"n": 14}),
    ("empty_loop", empty_loop, {"o": [0] * THREADS}, {"n": 0}),
    ("elif_chain", elif_chain, {"a": [-4, 0, 9, 10, 55, 99, 100, 250], "o": [0] * THREADS}, {"n": 8}),
    ("and_or", and_or, {"a": [10, 11, 12, 13, 14, 15, 16, 17], "o": [0] * THREADS}, {"n": 8}),
    ("nested_loops", nested_loops, {"o": [0] * THREADS}, {"n": 5}),
    ("float_literals", float_literals, {"a": [0.0, 1.0, -2.5, 3.25, 100.0, -0.125, 0.5, 7.0], "o": [0.0] * THREADS}, {"n": 8}),
    ("float_lhs_literal", float_lhs_literal, {"a": [0.0, 1.0, -2.5, 3.25, 100.0, -0.125, 0.5, 7.0], "o": [0.0] * THREADS}, {"n": 8}),
    ("float_branch", float_branch, {"a": [0.0, 0.5, 0.75, -1.0, 2.0, 0.25, 9.0, -9.0], "o": [0.0] * THREADS}, {"n": 8}),
    ("truediv_int", truediv_int, {"a": [-7, 7, 1, 0, -1, 16777216, 10, 3], "b": [2, -2, 3, 5, -1, 3, 4, -7],
                                  "o": [0.0] * THREADS}, {"n": 8}),
    ("truediv_f32", truediv_f32, {"a": [1.0, -7.5, 0.0, -0.0, 1e-40, math.inf, 3.0, 1e38],
                                  "b": [3.0, 2.0, 7.0, 5.0, 3.0, 2.0, -3.0, 1e-5],
                                  "o": [0.0] * 8, "p": [0.0] * 8, "q": [0.0] * 8}, {"n": 8}),
    ("floordiv_mod_reg", floordiv_mod_reg, {"a": N, "b": D, "q": [0] * 8, "r": [0] * 8}, {"n": 8}),
    ("floordiv_mod_reg_signs", floordiv_mod_reg, {"a": [7, -7, 7, -7, 6, -6, 0, I32_MIN], "b": [2, 2, -2, -2, -3, -3, -5, 3],
                                                  "q": [0] * 8, "r": [0] * 8}, {"n": 8}),
    ("floordiv_mod_const", floordiv_mod_const, {"a": N, **{k: [0] * 8 for k in
                                                ("q3", "r3", "qm3", "rm3", "q4", "r4", "qm4", "rm1")}}, {"n": 8}),
    ("floordiv_one", floordiv_one, {"a": [-7, 7, 0, -1, 1, 100, -100, 5], "q": [0] * 8, "m": [0] * 8}, {"n": 8}),
    ("casts", casts, {"a": [0, -3, 5, 16777217, -16777216, 1, -1, 99], "x": [0.0, -2.7, 2.7, -0.5, 1e9, -1e9, 7.999, -0.0],
                      "fo": [0.0] * 8, "io": [0] * 8}, {"n": 8}),
    ("not_cond", not_cond, {"a": [0, 1, 2, 3, 5, 6, 7, -1], "o": [0] * 8,
                            "f": [0.0, -0.0, math.nan, 1.0, -2.0, math.inf, 1e-45, math.nan]}, {"n": 8}),
    ("minmaxabs_i", minmaxabs_i, {"a": [-5, 5, I32_MIN + 1, 0, -1, 7, -9, 3], "b": [3, -3, 0, 0, -1, -8, -9, 2],
                                  "mn": [0] * 8, "mx": [0] * 8, "ab": [0] * 8}, {"n": 8}),
    ("minmaxabs_f", minmaxabs_f, {"a": FA, "b": FB, "mn": [0.0] * 8, "mx": [0.0] * 8, "ab": [0.0] * 8}, {"n": 8}),
    ("sqrt", sqrt_k, {"x": FX, "o": [0.0] * 8, "m": [0.0] * 8, "k": [0.0] * 8}, {"n": 8}),
    ("collatz", collatz, {"a": [1, 2, 3, 6, 7, 27, 97, 12], "o": [0] * 8}, {"n": 8}),
    ("while_true_continue", while_true_continue, {"o": [0] * 8}, {"n": 5}),
    ("for_break_continue", for_break_continue, {"o": [0] * 8, "last": [0] * 8}, {"n": 6}),
    ("for_break_continue_empty", for_break_continue, {"o": [0] * 8, "last": [0] * 8}, {"n": 0}),
    ("loop_var_assigned", loop_var_assigned, {"o": [0] * 8}, {"n": 4}),
    ("smem_reverse", smem_reverse, {"a": [float(k) - 3.5 for k in range(8)], "o": [0.0] * 8}, {"n": 8}),
    ("smem_reduce", smem_reduce, {"a": [float(k * k) for k in range(8)], "o": [0.0] * 8, "cnt": [0] * 8}, {"n": 8}),
]


@pytest.mark.parametrize("label,fn,arrays,scalars", CASES, ids=[c[0] for c in CASES])
def test_python_interpreter_and_sass_agree(label, fn, arrays, scalars):
    py = run_python(fn, arrays, scalars, THREADS)
    interp, sass = run_ptx_and_sass(fn, arrays, scalars, THREADS)
    assert same(interp, py), f"PTX pyfront emitted != plain Python\n python {py}\n ptx    {interp}"
    if isinstance(sass, Pending):
        pytest.skip(str(sass))
    assert same(sass, py), f"our SASS != plain Python\n python {py}\n sass   {sass}"


def test_approximate_intrinsics_within_documented_error():
    """exp/log/tanh/rsqrt are .approx (module docstring): the interpreter computes the exact
    result of each .approx instruction, so only the documented error budget is compared."""
    xs = [0.0, -1.0, 0.5, 2.0, -3.25, 10.0, -20.0, 0.125]
    arrays = {"x": xs, **{k: [0.0] * 8 for k in ("e", "lg", "th", "rs", "me")}}
    py = run_python(approx_k, arrays, {"n": 8}, THREADS)
    interp, sass = run_ptx_and_sass(approx_k, arrays, {"n": 8}, THREADS)
    # exp: x*log2(e) rounds to f32 first, so the error grows with |x| (CUDA: 2 + 1.2|x| ulp)
    assert close({"e": py["e"]}, {"e": interp["e"]}, rel=(2 + 1.2 * 20) * 2 ** -23)
    assert close({k: py[k] for k in ("lg", "rs", "me")}, {k: interp[k] for k in ("lg", "rs", "me")},
                 rel=2 ** -19, abs_=2 ** -21)
    assert close({"th": py["th"]}, {"th": interp["th"]}, rel=2 ** -10.5, abs_=2 ** -21)
    if isinstance(sass, Pending):
        pytest.skip(str(sass))
    assert close(sass, interp, rel=2 ** -10.5, abs_=2 ** -21)


# ---- documented places where the GPU differs from Python (module docstring) -----------------
# Python raises here, so there is no plain-Python leg: the expected values are the IEEE /
# PTX results the docstring promises. The SASS leg (when the backend can compile it) must equal
# the interpreter bit for bit.

@kernel
def ieee_div(a: Array(F32), b: Array(F32), o: Array(F32), ia: Array(i32), ib: Array(i32), io: Array(F32), n: i32):
    i = tid_x()
    o[i] = a[i] / b[i]
    io[i] = ia[i] / ib[i]


@kernel
def cast_saturation(x: Array(F32), io: Array(i32), n: i32):
    i = tid_x()
    io[i] = int(x[i])


def _run_gpu_only(fn, arrays, scalars):
    interp, sass = run_ptx_and_sass(fn, arrays, scalars, THREADS)
    if not isinstance(sass, Pending):
        assert same(sass, interp), f"our SASS != PTX interpreter\n interp {interp}\n sass   {sass}"
    return interp, sass


def test_true_division_by_zero_is_ieee_not_an_exception():
    inf, nan = math.inf, math.nan
    arrays = {"a": [1.0, -1.0, 0.0, -0.0, 5.0, inf, 1.0, 0.0], "b": [0.0, 0.0, 0.0, 0.0, -0.0, inf, inf, 3.0],
              "o": [0.0] * 8, "ia": [1, -1, 0, 7, 5, 1, 2, 0], "ib": [0, 0, 0, 0, 0, 3, 3, 3], "io": [0.0] * 8}
    interp, sass = _run_gpu_only(ieee_div, arrays, {"n": 8})
    assert same({"o": interp["o"]}, {"o": [inf, -inf, nan, nan, -inf, nan, 0.0, 0.0]})
    assert same({"io": interp["io"]}, {"io": [inf, -inf, nan, inf, inf, f32(1 / 3), f32(2 / 3), 0.0]})
    if isinstance(sass, Pending):
        pytest.skip(str(sass))


def test_int_of_nan_inf_and_huge_saturates():
    xs = [math.nan, math.inf, -math.inf, 3e9, -3e9, 2147483520.0, -2147483648.0, 0.99999994]
    interp, sass = _run_gpu_only(cast_saturation, {"x": xs, "io": [9] * 8}, {"n": 8})
    assert interp["io"] == [0, I32_MAX, I32_MIN, I32_MAX, I32_MIN, 2147483520, I32_MIN, 0]
    if isinstance(sass, Pending):
        pytest.skip(str(sass))


def test_i32_multiplication_wraps():
    """i32 arithmetic wraps (Python ints don't): documented, so pin it."""
    @kernel
    def wrap(a: Array(i32), o: Array(i32), n: i32):
        i = tid_x()
        o[i] = a[i] * a[i]

    interp, _ = _run_gpu_only(wrap, {"a": [65536, 3, -65536, 46341, 0, 1, 2, 3], "o": [0] * 8}, {"n": 8})
    assert interp["o"][:5] == [0, 9, 0, (46341 * 46341 + 2**31) % 2**32 - 2**31, 0]   # 46341**2 > 2**31
