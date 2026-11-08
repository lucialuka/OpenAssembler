import pytest

from lowered_adapter import LoweredInstruction, schedule_lowered_block


def test_lowered_adapter_preserves_accumulator_chain_edges():
    body = [
        LoweredInstruction("HMMA.16816.F32 R8, R4, R4, R8", False, (4,), (8,)),
        LoweredInstruction("HMMA.16816.F32 R8, R4, R4, R8", False, (8, 4), (8,)),
        LoweredInstruction("STG.E [R0.64], R8", True, (0, 8), ()),
    ]
    order = schedule_lowered_block(body)
    assert order == [0, 1, 2]


def test_lowered_adapter_fails_closed_on_control_flow():
    with pytest.raises(ValueError, match="control flow"):
        schedule_lowered_block([LoweredInstruction("BRA label", False, (), ())])

