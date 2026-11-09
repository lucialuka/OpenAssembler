import pytest

from lowered_adapter import LoweredInstruction, schedule_lowered_block, to_tensor_instructions
from dependency_scheduler import EdgeKind, build_dependency_graph


def test_lowered_adapter_preserves_accumulator_chain_edges():
    body = [
        LoweredInstruction("HMMA.16816.F32 R8, R4, R4, R8", False, tuple(range(4, 12)), tuple(range(8, 12))),
        LoweredInstruction("HMMA.16816.F32 R8, R4, R4, R8", False, tuple(range(4, 12)), tuple(range(8, 12))),
        LoweredInstruction("MOV R20, R8", False, (8,), (20,)),
    ]
    order = schedule_lowered_block(body)
    assert order == [0, 1, 2]
    graph = build_dependency_graph(to_tensor_instructions(body))
    assert any(edge.kind == EdgeKind.OBSERVER and edge.consumer == 2 for edge in graph.edges)


def test_lowered_adapter_fails_closed_on_control_flow():
    with pytest.raises(ValueError, match="control flow"):
        schedule_lowered_block([LoweredInstruction("BRA label", False, (), ())])


@pytest.mark.parametrize("text", ("@P0 MOV R4, R5", "IADD3 R4, P0, R5, R6, RZ", "<label LOOP>"))
def test_lowered_adapter_refuses_unmodeled_predicate_or_label_state(text):
    with pytest.raises(ValueError, match="predicate|predication|control flow"):
        schedule_lowered_block([LoweredInstruction(text, False, (5, 6), (4,))])


@pytest.mark.parametrize("text", ("LDGSTS.E.BYPASS.128 [R4], [R8.64]", "LDGDEPBAR",
                                  "DEPBAR.LE SB0, 0x0", "BAR.SYNC 0x0", "STG.E [R0.64], R8"))
def test_lowered_adapter_does_not_accept_unmodeled_memory_or_visibility(text):
    with pytest.raises(ValueError, match="visibility|unsupported memory"):
        schedule_lowered_block([LoweredInstruction(text, False, (), ())])


def test_lowered_adapter_refuses_incomplete_hmma_wide_operand_metadata():
    with pytest.raises(ValueError, match="incomplete HMMA"):
        schedule_lowered_block([LoweredInstruction("HMMA.16816.F32 R8, R4, R4, R8", False, (4,), (8,))])


def test_ldsm_requires_known_visibility_even_when_register_metadata_is_complete():
    ldsm = LoweredInstruction("LDSM.16.M88.4 R8, [R0+0x800]", True, (0,), (8, 9, 10, 11))
    with pytest.raises(ValueError, match="shared visibility"):
        schedule_lowered_block([ldsm])
    assert schedule_lowered_block([ldsm], shared_visibility_known=True) == [0]


def test_scalar_overwrite_of_pending_tensor_output_fails_closed():
    body = [LoweredInstruction("HMMA.16816.F32 R8, R4, R4, R8", False,
                               tuple(range(4, 12)), tuple(range(8, 12))),
            LoweredInstruction("MOV R8, R20", False, (20,), (8,))]
    with pytest.raises(ValueError, match="overwrite"):
        schedule_lowered_block(body)


def test_scalar_special_operand_cannot_hide_implicit_state():
    with pytest.raises(ValueError, match="implicit or unsupported"):
        schedule_lowered_block([LoweredInstruction("MOV R8, CC", False, (), (8,))])

