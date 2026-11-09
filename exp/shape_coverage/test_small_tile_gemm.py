from collections import Counter
import hashlib

import pytest

from exp.shape_coverage.small_tile_gemm import (
    SmallTilePlan, copy_coordinates, generate, layout_certificate, ldsm_addresses,
    output_coordinates, source_for_small_tile, static_summary,
    source_for_tile64,
)


def test_output_map_covers_exactly_one_cta_and_every_thread_has_32_distinct_elements():
    ownership = Counter()
    for tid in range(128):
        coordinates = output_coordinates(tid)
        assert len(coordinates) == len(set(coordinates)) == 32
        ownership.update(coordinates)
    assert ownership == Counter({(m, n): 1 for m in range(64) for n in range(64)})


def test_copy_map_is_exact_and_folded_displacement_matches_all_threads():
    for operand in range(2):
        for stage in range(2):
            logical = Counter()
            shared = Counter()
            for tid in range(128):
                first = copy_coordinates(tid, 0, operand, stage)["shared_byte"]
                for chunk in range(4):
                    item = copy_coordinates(tid, chunk, operand, stage)
                    assert item["shared_byte"] == first + 2048 * chunk
                    assert item["shared_byte"] % 16 == 0
                    logical.update((item["row"], item["column"] + j) for j in range(8))
                    shared.update(range(item["shared_byte"], item["shared_byte"] + 16))
            assert logical == Counter({(m, k): 1 for m in range(64) for k in range(64)})
            base = stage * 16384 + operand * 8192
            assert shared == Counter({byte: 1 for byte in range(base, base + 8192)})


def test_ldsm_vectors_are_initialized_aligned_and_stay_in_their_operand_stage():
    for operand in range(2):
        for stage in range(2):
            base = stage * 16384 + operand * 8192
            for warp in range(4):
                for ik in range(4):
                    fragments = range(2) if operand == 0 else (0, 2)
                    for fragment in fragments:
                        rows = ldsm_addresses(operand, warp, ik, fragment, stage)
                        assert len(rows) == 32 and len(set(rows)) == 32
                        assert all(a % 16 == 0 and base <= a and a + 16 <= base + 8192 for a in rows)


@pytest.mark.parametrize("m,n,k,dtype", [
    (0,64,128,"f16"), (65,64,128,"f16"), (64,65,128,"f16"),
    (64,64,64,"f16"), (64,64,128,"tf32"), (True,64,128,"f16"),
    (64,4194304,128,"f16"), (131072,131072,128,"f16"),
])
def test_unsupported_contracts_fail_before_emission(m,n,k,dtype):
    with pytest.raises(ValueError):
        SmallTilePlan(m,n,k,dtype)


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_source_preserves_stage_envelope_and_exact_tensor_work(dtype):
    plan = SmallTilePlan(128,512,512,dtype)
    source, certificate = source_for_small_tile(plan)
    assert source == generate(plan)
    assert not source.startswith("// openassembler-gemm-v1")
    assert certificate["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert certificate["launch"] == {"grid":[2,8,1], "threads":128, "smem":32768}
    assert source.count("mma.sync.") == 128
    assert source.count("ldmatrix.sync.") == 64
    assert source.count("cp.async.cg.") == 32
    assert source.count("cp.async.commit_group") == 4
    assert source.count("cp.async.wait_group 0") == 4
    assert source.count("bar.sync 0") == 8
    assert source.count("st.global.v2.f32") == 16
    assert certificate["hardware_validated"] is False


def test_k128_has_no_speculative_third_tile_or_loop():
    source = generate(SmallTilePlan(64,64,128))
    assert "LOOP_STAGE_PAIR" not in source
    assert source.count("mma.sync.") == 64
    assert source.count("cp.async.cg.") == 16
    assert source.count("cp.async.commit_group") == 2


def test_certificate_and_native_encoding_remain_experiment_only():
    certificate = layout_certificate()
    assert certificate["output_ownership_exact_once"] and certificate["copy_coverage_exact_once"]
    result = static_summary(SmallTilePlan(128,512,512))
    assert result["static"]["opcodes"]["HMMA"] == 128
    assert result["static"]["opcodes"]["LDSM"] == 64
    assert result["static"]["registers"] <= 255
    assert result["certificate"]["production_promoted"] is False


def test_role_adapter_rejects_composing_unmeasured_policies():
    from openassembler.kernels.gemm import GemmPlan
    source, certificate = source_for_tile64(GemmPlan(128,512,512))
    assert source == generate(SmallTilePlan(128,512,512))
    assert certificate["manual_launch"] == {"grid":[2,8,1],"threads":128,"smem":32768}
    assert certificate["full_work_preserved"] and certificate["k_order_preserved"]
    with pytest.raises(ValueError, match="baseline"):
        source_for_tile64(GemmPlan(128,512,512,pointer_mode="direct"))
