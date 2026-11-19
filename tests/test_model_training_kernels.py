"""Exact admission boundaries for A100 model matrix and reduction policies."""

import pytest
from openassembler.modelfront.kernels import emit_kernel
from openassembler.modelfront.kernels_tensor import emit_tensor_recipe, MATH_MODE
from openassembler.modelfront.kernels_training import select_tensor_plan
from openassembler.kernels.gemm import GemmPlan


def spec(*shape):
    return {"shape": list(shape), "dtype": "f32"}


@pytest.mark.parametrize(
    "op,a,b,out",
    [
        ("linear", (128, 128), (256, 128), (128, 256)),
        ("linear", (128, 256), (128, 256), (128, 128)),
        ("linear_dx", (128, 256), (256, 128), (128, 128)),
        ("linear_dw", (128, 256), (128, 128), (256, 128)),
        ("linear", (128, 128), (128, 128), (128, 128)),
    ],
)
def test_measured_half_model_products_choose_small_tiles(op, a, b, out):
    r = emit_tensor_recipe(
        op, [spec(*a), spec(*b)], spec(*out), {"math_mode": MATH_MODE}
    )
    assert r.signature["policy"] == "model_small_grid_tile64"
    stage = next(s for s in r.stages if s.gemm_plan is not None)
    assert stage.gemm_plan.tile == "64x64"
    assert stage.launch["grid"] == [out[0] // 64, out[1] // 64, 1]
    assert stage.launch["smem"] == 32768


def test_small_tile_policy_does_not_override_other_precision_shapes_or_selected_policy():
    for p, d in [
        (GemmPlan(256, 256, 128), {"policy": "baseline"}),
        (GemmPlan(128, 128, 128, "bf16"), {"policy": "baseline"}),
        (GemmPlan(128, 128, 128), {"policy": "pointer_direct"}),
    ]:
        got, decision = select_tensor_plan(p, d)
        assert got == p and decision == d


@pytest.mark.parametrize(
    "shape,selected",
    [
        ((128, 128), True),
        ((128, 256), True),
        ((127, 128), False),
        ((128, 129), False),
        ((256, 128), False),
    ],
)
def test_parallel_bias_reduction_is_exact_shape_only(shape, selected):
    k = emit_kernel("bias_grad", [spec(*shape)], spec(shape[1]))
    assert k.name.startswith("oa_bias_reduce_") is selected
    if selected:
        assert k.grid == ((shape[1] + 31) // 32, 1, 1) and k.threads == 128


def test_warp_mse_preserves_scalar_output_and_rejects_invalid_contracts():
    with pytest.raises(ValueError):
        emit_kernel("mse", [spec(128, 128), spec(128, 127)], spec())
    with pytest.raises(ValueError):
        emit_kernel("mse", [spec(128, 128)] * 2, spec(2))
    k = emit_kernel("mse", [spec(128, 128)] * 2, spec())
    assert k.output == () and k.threads == 1024 and k.grid == (1, 1, 1)
