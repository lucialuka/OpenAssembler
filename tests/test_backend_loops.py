"""Regression coverage for nvcc's pointer-increment forms and address loops."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from openassembler import oracle
from tests.backend_harness import compile_and_diff, f32b, pack_f32

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / "data" / "backend_loops"
needs_nvcc = pytest.mark.skipif(not oracle.available("nvcc"), reason="needs CUDA toolkit nvcc")


def _ptx(name: str) -> str:
    return (DATA / f"{name}.ptx").read_text()


def test_nvcc_unrolled_dot_rows_pointer_increment_is_bit_exact():
    n, rows = 7, 4
    rng = random.Random(7)
    a = [rng.uniform(-1, 1) for _ in range(rows * n)]
    b = [rng.uniform(-1, 1) for _ in range(rows * n)]
    ptx = _ptx("dot_rows")
    _, _, bad = compile_and_diff(ptx, "dot_rows", {
        "dot_rows_param_0": pack_f32(a),
        "dot_rows_param_1": pack_f32(b),
        "dot_rows_param_2": b"\0" * (rows * 4),
    }, {"dot_rows_param_3": n}, block=(rows, 1, 1))
    assert not bad, bad[:5]


def test_grid_stride_saxpy_pointer_increment_loop_is_bit_exact():
    n, grid, block = 37, 3, 8
    x = [i / 13 for i in range(n)]
    y = [i / 7 for i in range(n)]
    ptx = _ptx("saxpy_gs")
    _, _, bad = compile_and_diff(ptx, "saxpy_gs", {
        "saxpy_gs_param_1": pack_f32(x),
        "saxpy_gs_param_2": pack_f32(y),
    }, {"saxpy_gs_param_0": f32b(0.75), "saxpy_gs_param_3": n},
        block=(block, 1, 1), grid=(grid, 1, 1))
    assert not bad, bad[:5]


@pytest.mark.parametrize("name", ["matmul_naive", "nested2"])
def test_nvcc_complex_loop_fixtures_compile_and_encode(name):
    ptx = _ptx(name)
    from tests.backend_harness import compile_ptx
    code, stats, words = compile_ptx(ptx, name)
    assert code and words and stats["regs"] <= 255


def test_simple_pointer_increment_dot_loop_is_bit_exact():
    """A direct PTX loop: two base pointers advance by four bytes each trip."""
    ptx = """.version 8.7
.target sm_80
.address_size 64
.visible .entry dot_simple(.param .u64 out, .param .u64 a, .param .u64 b, .param .u32 n)
{
 .reg .pred %p<2>; .reg .b32 %r<8>; .reg .f32 %f<5>; .reg .b64 %rd<8>;
 ld.param.u64 %rd1, [out]; ld.param.u64 %rd2, [a]; ld.param.u64 %rd3, [b];
 ld.param.u32 %r1, [n]; cvta.to.global.u64 %rd1, %rd1; cvta.to.global.u64 %rd2, %rd2; cvta.to.global.u64 %rd3, %rd3;
 mov.f32 %f1, 0f00000000; mov.u32 %r2, 0;
LOOP: setp.ge.u32 %p1, %r2, %r1; @%p1 bra DONE;
 ld.global.f32 %f2, [%rd2]; ld.global.f32 %f3, [%rd3]; fma.rn.f32 %f1, %f2, %f3, %f1;
 add.s64 %rd2, %rd2, 4; add.s64 %rd3, %rd3, 4; add.u32 %r2, %r2, 1; bra LOOP;
DONE: st.global.f32 [%rd1], %f1; ret;
}
"""
    n = 9
    a = [i / 7 for i in range(n)]
    b = [i / 5 for i in range(n)]
    out = b"\0" * 4
    _, _, bad = compile_and_diff(ptx, "dot_simple", {
        "out": out, "a": pack_f32(a), "b": pack_f32(b),
    }, {"n": n}, block=(1, 1, 1))
    assert not bad, bad[:5]


@needs_nvcc
@pytest.mark.parametrize("name", ["dot_rows", "saxpy_gs"])
def test_cuda_source_reproduces_stored_nvcc_ptx(name, tmp_path):
    from openassembler.cppfront import compile_to_ptx_module

    ptx, names = compile_to_ptx_module(DATA / f"{name}.cu", arch="sm_80")
    assert name in names
    assert ".entry" in ptx



def test_nested_loop_outputs_are_numerically_correct():
    from tests.backend_harness import pack_u32
    _, _, bad = compile_and_diff(_ptx('nested2'), 'nested2', {
        'nested2_param_0': pack_u32([i * 3 - 17 for i in range(64)]),
        'nested2_param_1': b'\0' * 32,
    }, {'nested2_param_2': 3, 'nested2_param_3': 7}, block=(8, 1, 1))
    assert not bad, bad[:5]


def test_matmul_outputs_are_numerically_correct():
    m, n, k = 3, 5, 7
    _, _, bad = compile_and_diff(_ptx('matmul_naive'), 'matmul_naive', {
        'matmul_naive_param_0': pack_f32([(i % 11 - 5) / 7 for i in range(m*k)]),
        'matmul_naive_param_1': pack_f32([(i % 13 - 6) / 5 for i in range(k*n)]),
        'matmul_naive_param_2': b'\0' * (m*n*4),
    }, {'matmul_naive_param_3': m, 'matmul_naive_param_4': n, 'matmul_naive_param_5': k},
        block=(n, m, 1))
    assert not bad, bad[:5]
