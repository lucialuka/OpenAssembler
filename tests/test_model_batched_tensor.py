"""FP16 batched orientation, stride bounds, and exact small-path operand rounding."""

import math
import struct

import pytest

from openassembler.build import generate
from openassembler.emu.machine import Launch, Machine, Memory
from openassembler.emu.program import load_listing
from openassembler.modelfront.kernels_tensor import emit_tensor_recipe, MATH_MODE
from openassembler.modelfront.kernels_tensor_batched import rounded_operand
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing


def spec(shape):
    return {"shape": list(shape), "dtype": "f32"}


@pytest.mark.parametrize(
    "op,left,right,out",
    [
        ("bmm", (3, 17, 65), (3, 65, 129), (3, 17, 129)),
        ("bmm_dx", (3, 17, 129), (3, 65, 129), (3, 17, 65)),
        ("bmm_dw", (3, 17, 129), (3, 17, 65), (3, 65, 129)),
    ],
)
def test_batched_tensor_geometry_and_every_workspace_view_is_in_bounds(
    op, left, right, out
):
    recipe = emit_tensor_recipe(
        op, [spec(left), spec(right)], spec(out), {"math_mode": MATH_MODE}
    )
    gemm = next(s for s in recipe.stages if s.gemm_plan)
    assert gemm.launch["grid"][2] == 3
    assert "%ctaid.z" in gemm.ptx
    assert gemm.signature["batch_strides_bytes"] == [
        gemm.gemm_plan.m * gemm.gemm_plan.k * 2,
        gemm.gemm_plan.n * gemm.gemm_plan.k * 2,
        gemm.gemm_plan.m * gemm.gemm_plan.n * 4,
    ]
    for workspace in recipe.workspaces.values():
        if "base" in workspace:
            base = recipe.workspaces[workspace["base"]]
            assert workspace["offset"] + math.prod(workspace["shape"]) <= math.prod(
                base["shape"]
            )


@pytest.mark.parametrize(
    "left,right,out",
    [
        ((2, 3, 4), (1, 4, 5), (2, 3, 5)),
        ((33, 3, 4), (33, 4, 5), (33, 3, 5)),
        ((2, 513, 4), (2, 4, 5), (2, 513, 5)),
        ((2, 3, 4), (2, 4, 5), (2, 3, 6)),
    ],
)
def test_batched_half_rejects_broadcasts_unbounded_shapes_and_wrong_outputs(
    left, right, out
):
    with pytest.raises(ValueError):
        emit_tensor_recipe(
            "bmm", [spec(left), spec(right)], spec(out), {"math_mode": MATH_MODE}
        )


def test_small_path_rounds_half_boundaries_exactly_without_padded_gemm():
    bits = [
        0,
        0x80000000,
        1,
        0x80000001,
        0x33000000,
        0x33000001,
        0x33800000,
        0x387FFFFF,
        0x38800000,
        0x3F801000,
        0x3F803000,
        0xBF801000,
        0x477FE000,
        0xC77FEFFF,
        0x38000000,
        0xB8000000,
    ]
    source, destination = 0x100000, 0x200000
    stage = rounded_operand((len(bits),), "output")
    code, labels, *_ = generate(
        parse_module(stage.ptx).kernels[stage.name], "sm_80", "opt"
    )
    memory = Memory()
    memory.write_bytes(source, struct.pack("<16I", *bits))
    Machine(
        load_listing(listing(code, labels)),
        Launch(
            block=(128, 1, 1),
            grid=(1, 1, 1),
            params=struct.pack("<QQ", source, destination),
            poison=0xDEADBEEF,
        ),
        memory,
    ).run()
    expected = []
    for raw in bits:
        value = struct.unpack("<f", struct.pack("<I", raw))[0]
        rounded = struct.unpack("<e", struct.pack("<e", value))[0]
        expected.append(int.from_bytes(struct.pack("<f", rounded), "little"))
    assert [memory.read(destination + 4 * i, 4) for i in range(len(bits))] == expected


def test_half_bmm_capture_and_reference_round_all_three_products_independently():
    torch = pytest.importorskip("torch")
    from openassembler.modelfront.graph import capture_model
    from openassembler.modelfront.reference import CPUReference
    from openassembler.modelfront.kernels import prepare

    class Model(torch.nn.Module):
        def forward(self, a, b):
            return a @ b

    torch.manual_seed(81)
    a, b = torch.randn(2, 3, 5), torch.randn(2, 5, 7)
    plan = capture_model(Model(), [a, b], math_mode=MATH_MODE)
    reference = CPUReference(plan, {})
    output = plan["outputs"][0]
    result = reference.forward([a, b])[output]
    dy = torch.randn(2, 3, 7)
    gradients = reference.backward({output: dy})["inputs"]
    half = lambda x: x.half().float()
    torch.testing.assert_close(result, half(a) @ half(b), rtol=0, atol=0)
    torch.testing.assert_close(
        gradients[plan["inputs"][0]["name"]],
        half(dy) @ half(b).transpose(-2, -1),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        gradients[plan["inputs"][1]["name"]],
        half(a).transpose(-2, -1) @ half(dy),
        rtol=0,
        atol=0,
    )
    _, steps, _ = prepare(plan)
    assert all(s.get("tensor") for s in steps if s["kernel"].startswith("bmm"))


def test_small_dispatch_does_not_mistake_tall_low_volume_products_for_small_matrices():
    small = emit_tensor_recipe('bmm', [spec((3, 7, 5)), spec((3, 5, 9))], spec((3, 7, 9)), {'math_mode': MATH_MODE})
    tall = emit_tensor_recipe('bmm', [spec((1, 511, 17)), spec((1, 17, 3))], spec((1, 511, 3)), {'math_mode': MATH_MODE})
    assert small.signature['implementation'] == 'rounded_scalar'
    assert all(s.gemm_plan is None for s in small.stages)
    assert any(s.gemm_plan is not None for s in tall.stages)
