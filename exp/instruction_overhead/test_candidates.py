import pytest
from exp.instruction_overhead.candidates import CANDIDATES, address_certificate, source, static_summary
from exp.shape_coverage.small_tile_gemm import SmallTilePlan, generate


def test_all_thread_chunk_addresses_are_equal_and_keep_pointer_carry():
    assert address_certificate()["copy_address_equivalences"] == 7680


@pytest.mark.parametrize("dtype", ["f16","bf16"])
@pytest.mark.parametrize("k", [128,256,512])
def test_candidate_stream_keeps_tensor_visibility_barrier_order(dtype,k):
    plan=SmallTilePlan(128,512,k,dtype)
    for candidate in CANDIDATES:
        text, cert=source(plan,candidate)
        assert cert["tensor_copy_visibility_barrier_stream_equal"]
        assert cert["dependent_accumulator_k_order_equal"]
    assert source(plan,CANDIDATES[0])[0] == generate(plan)


def test_native_encodings_preserve_tensor_controls_and_reduce_non_tensor_work():
    rows = {c.label: static_summary(SmallTilePlan(128,512,512),c) for c in CANDIDATES}
    assert rows["legacy"]["instructions"] == 723
    assert rows["direct"]["instructions"] == 659
    for row in rows.values():
        assert row["opcodes"]["HMMA.16816.F32"] == 128
        assert row["opcodes"]["LDSM.16.M88.4"] == 64
        assert row["hmma_control"] == [8]
        assert row["opcodes"]["NOP"] == 16
        assert row["registers"] <= 96
    assert rows["init_reuse_direct"]["instructions"] < rows["direct"]["instructions"]


def test_unknown_candidate_cannot_change_source():
    from exp.instruction_overhead.candidates import Candidate
    with pytest.raises(ValueError,match="inventory"):
        source(SmallTilePlan(128,512,512),Candidate("unknown"))
