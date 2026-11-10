from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import builtins
import importlib
import json
import sys
import zipfile

import pytest

from openassembler import gemm_dispatch
from openassembler import gemm_tune
from openassembler.kernels.gemm import GemmPlan, generate


def _record(plan=None, *, winner="tile64"):
    plan = plan or GemmPlan(128, 512, 512)
    candidates = {}
    for name, candidate in gemm_dispatch.candidate_plans(plan).items():
        checks = [{"input": family, "relative_l2_fp32": 0.0, **{gate: True for gate in (
            "full_output", "canaries", "inputs_unchanged", "finite_output",
            "ptxas_bitwise", "baseline_bitwise")}} for family in ("zero", "integer", "normal")]
        candidates[name] = {
            "plan": asdict(candidate),
            "ptx_body_sha256": hashlib.sha256(generate(candidate).encode()).hexdigest(),
            "checks": checks,
            "poisoned_graph": True,
            "disassembly_valid": True,
            "resources_valid": True,
            "artifacts": {kind: {"resources": {"num_regs": 88, "local_bytes": 0,
                "max_threads": 640, "shared_bytes": 0, "binary_version": 80}}
                for kind in ("native", "ptxas")},
        }
    names = list(candidates)
    timings = []
    for round_index in range(3):
        orders = [names[:], names[::-1], names[1:] + names[:1], names[2:] + names[:2],
                  names[::2] + names[1::2], names[::3] + names[1::3] + names[2::3],
                  names[::4] + names[1::4] + names[2::4] + names[3::4],
                  names[3:] + names[:3]]
        samples = {}
        for name in names:
            value = 10.0 if name == "baseline" else (7.0 if name == winner else 10.5)
            samples[name] = [value] * len(orders)
        timings.append({"round": round_index, "paired_orders": orders, "samples_ms": samples})
    return {"schema": 1, "compiler_identity": gemm_dispatch.compiler_identity(),
            "contract": gemm_dispatch.contract_key(plan),
            "device": {"name": "A100-SXM4-40GB", "cc": "8.0"},
            "candidates": candidates, "timings": timings}


def test_choose_policy_requires_complete_checks_and_repeatable_paired_speedup():
    assert gemm_tune.choose_policy(_record()) == "tile64"
    assert gemm_tune.choose_policy(_record(winner="pointer_direct")) == "pointer_direct"

    result = _record()
    result["timings"][1]["samples_ms"]["tile64"] = [10.0] * len(
        result["timings"][1]["paired_orders"])
    assert gemm_tune.choose_policy(result) == "baseline"


@pytest.mark.parametrize("minimum", [1, 0.9, float("nan"), float("inf"), "1.02"])
def test_choose_policy_rejects_invalid_threshold(minimum):
    with pytest.raises(ValueError, match="threshold"):
        gemm_tune.choose_policy(_record(), minimum_speedup=minimum)


@pytest.mark.parametrize("mutate,match", [
    (lambda r: r["timings"].pop(), "three paired rounds"),
    (lambda r: r["timings"][0]["samples_ms"].pop("tile64"), "incomplete paired"),
    (lambda r: r["timings"][0]["paired_orders"][0].append("tile64"), "cover every"),
    (lambda r: r["timings"][0]["samples_ms"]["baseline"].__setitem__(0, 0), "invalid"),
    (lambda r: r["timings"][0]["samples_ms"]["baseline"].__setitem__(0, float("nan")), "invalid"),
    (lambda r: r["candidates"]["tile64"]["checks"][0].__setitem__("canaries", False), "failed numerical"),
    (lambda r: r["candidates"]["tile64"].__setitem__("poisoned_graph", False), "graph"),
    (lambda r: r["candidates"]["tile64"].__setitem__("disassembly_valid", False), "graph"),
    (lambda r: r["candidates"]["tile64"].__setitem__("resources_valid", False), "graph"),
])
def test_choose_policy_rejects_incomplete_or_failed_evidence(mutate, match):
    result = _record()
    mutate(result)
    with pytest.raises(ValueError, match=match):
        gemm_tune.choose_policy(result)


def test_validate_record_checks_identity_contract_device_and_every_source_plan():
    plan = GemmPlan(128, 512, 512)
    result = _record(plan)
    chosen, decision = gemm_tune.validate_record(plan, result)
    assert chosen == gemm_dispatch.policy_plan(plan, "tile64")
    assert decision["policy"] == "tile64"
    assert decision["device"]["cc"] == "8.0"

    for mutate, match in [
        (lambda r: r.__setitem__("compiler_identity", "stale"), "different compiler"),
        (lambda r: r.__setitem__("contract", gemm_dispatch.contract_key(GemmPlan(128, 512, 640))), "different shape"),
        (lambda r: r["device"].__setitem__("cc", "9.0"), "not an A100"),
        (lambda r: r["device"].__setitem__("name", "H100"), "not an A100"),
        (lambda r: r["candidates"].pop("tile64"), "current admitted candidate set"),
        (lambda r: r["candidates"]["tile64"]["plan"].__setitem__("k", 640), "source changed"),
        (lambda r: r["candidates"]["tile64"].__setitem__("ptx_body_sha256", "0" * 64), "source changed"),
    ]:
        broken = deepcopy(result)
        mutate(broken)
        with pytest.raises(ValueError, match=match):
            gemm_tune.validate_record(plan, broken)


@pytest.mark.parametrize("mutate", [
    lambda r: r.__setitem__("device", []),
    lambda r: r["candidates"].__setitem__("tile64", None),
    lambda r: r["candidates"]["tile64"].__setitem__("checks", ["malformed"]),
])
def test_tuning_record_malformed_nested_data_fails_closed_with_value_error(mutate):
    plan = GemmPlan(128, 512, 512)
    broken = _record(plan)
    mutate(broken)
    with pytest.raises(ValueError):
        gemm_tune.validate_record(plan, broken)


@pytest.mark.parametrize("kwargs,match", [
    ({"backend": "bogus", "record": None}, "backend"),
    ({"backend": None, "record": None}, "record"),
    ({"backend": None, "record": "/definitely/missing.json"}, "No such file"),
    ({"backend": None, "record": None, "rounds": 2}, "at least three"),
    ({"backend": None, "record": None, "samples": 7}, "at least three"),
    ({"backend": None, "record": None, "rounds": 11}, "bounded"),
    ({"backend": None, "record": None, "samples": 101}, "bounded"),
])
def test_tune_or_load_rejects_non_gpu_or_unbounded_requests(kwargs, match):
    with pytest.raises((ValueError, OSError), match=match):
        gemm_tune.tune_or_load(GemmPlan(128, 512, 512), **kwargs)


def test_tune_or_load_record_path_requires_retained_archive_before_validation(tmp_path):
    record = tmp_path / "result.json"
    record.write_text("{}")
    with pytest.raises(ValueError, match="evidence archive"):
        gemm_tune.tune_or_load(GemmPlan(128, 512, 512), backend=None, record=str(record))


def test_record_reuse_verifies_source_and_binary_artifacts_before_selecting(tmp_path):
    from openassembler.gemm import source_for

    plan = GemmPlan(128, 512, 512)
    result = _record(plan)
    archive_path = tmp_path / "result.artifacts.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, candidate in result["candidates"].items():
            for kind in ("native", "ptxas"):
                cubin = f"{name}/{kind}.cubin".encode()
                sass = f"{name}/{kind}.sass".encode()
                candidate["artifacts"][kind] = {
                    "cubin_sha256": hashlib.sha256(cubin).hexdigest(),
                    "sass_sha256": hashlib.sha256(sass).hexdigest(),
                    "resources": {"num_regs": 88, "local_bytes": 0, "max_threads": 640,
                                  "shared_bytes": 0, "binary_version": 80},
                }
                archive.writestr(f"{name}/{kind}.cubin", cubin)
                archive.writestr(f"{name}/{kind}.sass", sass)
            archive.writestr(f"{name}/gemm.ptx", source_for(GemmPlan(**candidate["plan"])))
    result["artifact_archive"] = {
        "path": archive_path.name,
        "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    }
    record = tmp_path / "result.json"
    record.write_text(json.dumps(result))
    chosen, _ = gemm_tune.tune_or_load(plan, backend=None, record=str(record))
    assert chosen == gemm_dispatch.policy_plan(plan, "tile64")

    archive_path.write_bytes(archive_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="archive is missing or changed"):
        gemm_tune.tune_or_load(plan, backend=None, record=str(record))


def test_gemm_tuner_imports_without_touching_torch_or_modal(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("normal tuner import must not touch Torch")
        if name == "modal" or name.startswith("modal."):
            raise AssertionError("normal tuner import must not touch Modal")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    importlib.reload(gemm_tune)
    assert "torch" not in sys.modules
    assert "modal" not in sys.modules


def test_local_tuning_is_explicit_and_does_not_silently_fallback(monkeypatch):
    called = []
    def unavailable(*args, **kwargs):
        called.append((args, kwargs))
        raise ValueError("local GPU tuning requires the optional openassembler[torch] dependency")
    monkeypatch.setattr(gemm_tune, "benchmark", unavailable)
    with pytest.raises(ValueError, match=r"optional openassembler\[torch\]"):
        gemm_tune.tune_or_load(GemmPlan(128, 512, 512), backend="local", record=None)
    assert called


def test_cli_tuning_requires_no_manual_knobs_and_uses_supplied_tuning_result(monkeypatch, tmp_path):
    from openassembler import gemm
    from openassembler import gemm_tune as tuner
    from openassembler.cli import main

    calls = []
    selected = GemmPlan(128, 512, 512, tile="64x64")
    monkeypatch.setattr(tuner, "tune_or_load",
                        lambda plan, **kwargs: calls.append((plan, kwargs)) or
                        (selected, {"mode": "gpu-tuned", "policy": "tile64"}))
    monkeypatch.setattr(gemm, "compile_gemm", lambda plan, *args, **kwargs:
                        calls.append((plan, kwargs["dispatch"])) or 0)
    record = tmp_path / "tune.json"
    assert main(["gemm", "--m", "128", "--n", "512", "--k", "512",
                 "--autotune", "local", "--tuning-record", str(record),
                 "--tune-rounds", "3", "--tune-samples", "8",
                 "--out-dir", str(tmp_path)]) == 0
    tune_input, tune_kwargs = calls[0]
    assert tune_input.tile == "128x128"
    assert tune_kwargs == {"backend": "local", "record": str(record), "rounds": 3, "samples": 8}
    assert calls[1][0] is selected
    assert calls[1][1]["mode"] == "gpu-tuned"

    calls.clear()
    assert main(["gemm", "--m", "128", "--n", "512", "--k", "512",
                 "--autotune", "local", "--tile", "64x64", "--out-dir", str(tmp_path)]) == 1
    assert not calls


def test_record_source_and_device_are_shape_specific_not_a_generic_fastest_claim():
    plan = GemmPlan(1024, 4096, 1024)
    result = _record(plan, winner="baseline")
    selected, decision = gemm_tune.validate_record(plan, result)
    assert selected == gemm_dispatch.baseline(plan)
    assert "this shape" in decision["scope"]
    assert decision["contract"]["m"] == 1024
    assert decision["contract"]["dtype"] == "f16"


@pytest.mark.parametrize("error", [float("inf"), float("nan"), -1.0, 5e-5, True, None])
def test_record_cannot_claim_pass_when_recorded_numeric_error_is_invalid(error):
    result = _record()
    result["candidates"]["tile64"]["checks"][0]["relative_l2_fp32"] = error
    with pytest.raises(ValueError, match="numerical error"):
        gemm_tune.validate_record(GemmPlan(128, 512, 512), result)


@pytest.mark.parametrize("field,value", [
    ("local_bytes", 4), ("binary_version", 90), ("num_regs", True),
    ("num_regs", 256), ("shared_bytes", -1), ("max_threads", 64),
])
def test_record_revalidates_resource_numbers_instead_of_trusting_pass_flag(field, value):
    result = _record()
    result["candidates"]["tile64"]["artifacts"]["native"]["resources"][field] = value
    with pytest.raises(ValueError, match="resource report is invalid"):
        gemm_tune.validate_record(GemmPlan(128, 512, 512), result)


def test_three_copies_of_one_timing_round_do_not_establish_repeatability():
    result = _record()
    result["timings"][1]["round"] = result["timings"][0]["round"]
    with pytest.raises(ValueError, match="round"):
        gemm_tune.choose_policy(result)
