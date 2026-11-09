import copy

import pytest

from openassembler.gemm import contract_from_source
from openassembler.kernels.gemm import GemmPlan

from exp.shape_coverage.gemm_policy import policy_source, verify_compile_record


def _record(plan, source):
    return {"backend": "openassembler", "gemm": contract_from_source(source),
            "tensor": {"schedule": "source-order-hmma8-epoch-drain15",
                       "pointer_mode": plan.pointer_mode,
                       "pointer_advance": plan.pointer_advance}}


def test_checked_pointer_lowering_is_retained_by_source_and_kernel_name():
    plan = GemmPlan(1024, 4096, 1024, pointer_mode="direct", pointer_advance="early")
    source, cert = policy_source(plan, "pointer_direct_early")
    result = verify_compile_record(plan, "pointer_direct_early", source, _record(plan, source), cert)
    assert result["checked_gemm_contract"]
    assert result["native_tensor_policy"]["pointer_mode"] == "direct"
    assert contract_from_source(source.replace(".entry gemm(", ".entry renamed(")) is None


@pytest.mark.parametrize("split", (2, 4, 6))
def test_fragment_source_requires_explicit_manual_launch_and_certificate(split):
    plan = GemmPlan(1024, 4096, 1024)
    policy = f"fragment_split{split}"
    source, cert = policy_source(plan, policy)
    assert contract_from_source(source) is None
    result = verify_compile_record(plan, policy, source, _record(plan, source), cert)
    assert result["checked_gemm_contract"] is False
    assert result["manual_launch"] == plan.launch
    assert result["requested_tensor_policy"]["hmma_stall"] == 8
    assert result["requested_tensor_policy"]["epoch_drain"] == 15
    bad = copy.deepcopy(cert)
    bad["k_order_preserved"] = False
    with pytest.raises(AssertionError, match="certificate"):
        verify_compile_record(plan, policy, source, _record(plan, source), bad)


def test_manifest_guard_rejects_fallback_or_silently_lost_pointer_policy():
    plan = GemmPlan(1024, 4096, 1024, pointer_mode="direct")
    source, cert = policy_source(plan, "pointer_direct")
    record = _record(plan, source)
    record["backend"] = "ptxas-fallback"
    with pytest.raises(AssertionError):
        verify_compile_record(plan, "pointer_direct", source, record, cert)
    record["backend"] = "openassembler"
    record["tensor"]["pointer_mode"] = "legacy"
    with pytest.raises(AssertionError):
        verify_compile_record(plan, "pointer_direct", source, record, cert)


def test_tile64_manual_launch_cannot_inherit_the_128_tile_launch():
    plan = GemmPlan(128, 512, 512)
    source, cert = policy_source(plan, "tile64")
    assert contract_from_source(source) is None
    result = verify_compile_record(plan, "tile64", source, _record(plan, source), cert)
    assert result["manual_launch"] == {"grid": [2, 8, 1], "threads": 128, "smem": 32768}
    cert["manual_launch"] = plan.launch
    with pytest.raises(AssertionError, match="tile64"):
        verify_compile_record(plan, "tile64", source, _record(plan, source), cert)
