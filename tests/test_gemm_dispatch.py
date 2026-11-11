from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

import pytest

from openassembler import gemm_dispatch as dispatch
from openassembler.kernels.gemm import GemmPlan, generate


def _table(plan, policy="tile64", **changes):
    chosen = dispatch.policy_plan(plan, policy)
    entry = {
        "contract": dispatch.contract_key(plan),
        "status": "promoted",
        "gates_passed": True,
        "policy": policy,
        "ptx_body_sha256": hashlib.sha256(generate(chosen).encode()).hexdigest(),
        "evidence": ["a100-case.json"],
        "speedup_range": [1.8, 1.9],
    }
    entry.update(changes)
    return {"schema": 1, "compiler_identity": dispatch.compiler_identity(),
            "device_scope": {"name": "A100-SXM4-40GB", "cc": "8.0"}, "entries": [entry]}


def test_dispatch_uses_exact_measured_contract_and_recomputes_source_hash():
    plan = GemmPlan(128, 512, 512)
    chosen, decision = dispatch.select_plan(plan, table=_table(plan))
    assert chosen == dispatch.policy_plan(plan, "tile64")
    assert chosen.launch == {"grid": [2, 8, 1], "threads": 128, "smem": 32768}
    assert decision["policy"] == "tile64"
    assert decision["reason"] == "exact measured contract"
    assert decision["evidence"] == ["a100-case.json"]
    assert decision["device_scope"]["name"] == "A100-SXM4-40GB"

    table = _table(plan)
    table["entries"][0]["ptx_body_sha256"] = "0" * 64
    fallback, decision = dispatch.select_plan(plan, table=table)
    assert fallback == dispatch.baseline(plan)
    assert "source changed" in decision["reason"]


@pytest.mark.parametrize("entry_changes,reason", [
    ({"status": "validated-experiment-only"}, "promotion gates"),
    ({"status": "promoted", "gates_passed": False}, "promotion gates"),
])
def test_dispatch_falls_back_for_unpromoted_or_invalid_entries(entry_changes, reason):
    plan = GemmPlan(128, 512, 512)
    chosen, decision = dispatch.select_plan(plan, table=_table(plan, **entry_changes))
    assert chosen == dispatch.baseline(plan)
    assert reason in decision["reason"]


def test_dispatch_requires_unique_exact_contract_and_current_compiler_identity():
    plan = GemmPlan(128, 512, 512)
    table = _table(plan)
    table["entries"].append(deepcopy(table["entries"][0]))
    chosen, decision = dispatch.select_plan(plan, table=table)
    assert chosen == dispatch.baseline(plan)
    assert "unique" in decision["reason"]

    table = _table(plan)
    table["compiler_identity"] = "stale"
    chosen, decision = dispatch.select_plan(plan, table=table)
    assert chosen == dispatch.baseline(plan)
    assert "different compiler" in decision["reason"]


def test_dispatch_contract_covers_shape_dtype_layout_epilogue_precision_and_workspace():
    plan = GemmPlan(128, 512, 512, "f16")
    table = _table(plan)
    cases = [GemmPlan(128, 512, 512, "bf16"), GemmPlan(128, 512, 640),
             GemmPlan(128, 512, 512, epilogue="bias_residual")]
    for different in cases:
        chosen, decision = dispatch.select_plan(different, table=table)
        assert chosen == dispatch.baseline(different)
        assert "unique" in decision["reason"]
    contract = dispatch.contract_key(plan)
    assert contract["a_alignment"] == contract["b_alignment"] == 16
    assert contract["c_alignment"] == 8
    assert contract["workspace_bytes"] == 0
    assert contract["tails"] == "none"
    assert contract["alpha"] == 1 and contract["beta"] == 0
    assert contract["batch_count"] == 1


@pytest.mark.parametrize("table", [None, {"schema": 99}, {"schema": 1, "entries": {}},
                                    {"schema": 1, "entries": []}])
def test_dispatch_invalid_or_absent_table_falls_back(table, monkeypatch, tmp_path):
    plan = GemmPlan(128, 512, 512)
    actual = table
    if table is None:
        monkeypatch.setattr(dispatch, "TABLE", tmp_path / "missing.json")
    elif table.get("schema") == 1:
        actual = {**table, "compiler_identity": dispatch.compiler_identity()}
    chosen, decision = dispatch.select_plan(plan, table=actual)
    assert chosen == dispatch.baseline(plan)
    assert "reason" in decision


def test_candidate_search_keeps_prior_shape_losers_and_validates_each_plan():
    plan = GemmPlan(1024, 4096, 1024)
    candidates = dispatch.candidate_plans(plan)
    assert {"baseline", "combined", "morton8", "grouped_n8", "pointer_direct",
            "pointer_early", "pointer_direct_early", "load_next", "tile64"} <= set(candidates)
    assert candidates["combined"].schedule == "combined"
    assert candidates["tile64"].launch["grid"] == [16, 64, 1]
    assert len({json.dumps(asdict(candidate), sort_keys=True)
                for candidate in candidates.values()}) == len(candidates)

    # A syntactically available policy is admitted only when its complete
    # shape/pointer/layout contract survives GemmPlan validation.
    narrow = dispatch.candidate_plans(GemmPlan(128, 1024, 128))
    assert "morton8" not in narrow
    assert "grouped_n8" in narrow
    assert dispatch.policy_plan(GemmPlan(128, 512, 512), "tile64").tile == "64x64"


def test_64_multiple_shapes_use_baseline_tile_and_keep_invalid_policies_out():
    plan = GemmPlan(64, 192, 256, tile="64x64")
    assert dispatch.baseline(plan).tile == "64x64"
    candidates = dispatch.candidate_plans(plan)
    assert set(candidates) == {"baseline", "tile64_direct", "tile64_reuse",
                               "tile64_reuse_direct", "tile64_reuse_direct_early",
                               "tile64_direct_early", "pointer_early"}
    assert candidates["tile64_direct"].pointer_mode == "direct"
    assert dispatch.contract_key(plan) == dispatch.contract_key(dispatch.baseline(plan))


def test_cli_auto_selection_is_default_but_any_explicit_knob_has_precedence(monkeypatch, tmp_path):
    from openassembler import gemm
    from openassembler import gemm_dispatch as module
    from openassembler.cli import main

    calls = []
    selected = GemmPlan(128, 512, 512, tile="64x64")
    monkeypatch.setattr(module, "select_plan", lambda plan: (selected, {"mode": "measured", "policy": "tile64"}))
    monkeypatch.setattr(gemm, "compile_gemm", lambda plan, out, name, **kw:
                        calls.append((plan, kw["dispatch"])) or 0)
    assert main(["gemm", "--m", "128", "--n", "512", "--k", "512",
                 "--out-dir", str(tmp_path)]) == 0
    assert calls[-1][0] is selected

    def unexpected(plan):
        raise AssertionError("manual knob must bypass measured automatic selection")
    monkeypatch.setattr(module, "select_plan", unexpected)
    assert main(["gemm", "--m", "128", "--n", "512", "--k", "512",
                 "--tile", "64x64", "--out-dir", str(tmp_path)]) == 0
    manual, decision = calls[-1]
    assert manual.tile == "64x64"
    assert decision["mode"] == "manual"


def test_cli_auto_uses_64_tile_when_128_geometry_cannot_cover_shape(monkeypatch, tmp_path):
    from openassembler import gemm
    from openassembler import gemm_dispatch as module
    from openassembler.cli import main

    captured = []
    monkeypatch.setattr(module, "select_plan", lambda plan: (dispatch.baseline(plan), {"mode": "measured"}))
    monkeypatch.setattr(gemm, "compile_gemm", lambda plan, *args, **kwargs: captured.append(plan) or 0)
    assert main(["gemm", "--m", "64", "--n", "192", "--k", "128",
                 "--out-dir", str(tmp_path)]) == 0
    assert captured[-1].tile == "64x64"
    assert captured[-1].launch["grid"] == [1, 3, 1]


def test_runner_preserves_tile64_contract_and_launch_from_manifest(tmp_path):
    from openassembler import project
    from openassembler.runner import RunOptions, build_plan
    plan = GemmPlan(64, 128, 128, tile="64x64")
    source = tmp_path / "gemm.ptx"
    source.write_text("// checked source")
    (tmp_path / "ours.cubin").write_bytes(b"ours")
    (tmp_path / "ref.cubin").write_bytes(b"ref")
    contract = {**dispatch.contract_key(plan), **asdict(plan), "schema": 1,
                "launch": plan.launch, "input_layout": "A[M,K] and B[N,K], contiguous in K; C=A@B.T",
                "output_dtype": "f32", "accumulation": "f32"}
    project.save_manifest(tmp_path, tmp_path, "sm_80", "opt", [{
        "source": str(source), "cubin": "ours.cubin", "reference": "ref.cubin",
        "kernels": [{"name": "gemm", "params": [["u64", x] for x in ("A", "B", "C")],
                     "compiled_by": "openassembler", "gemm": contract}],
    }])
    run_plan, _ = build_plan(RunOptions(out_dir=tmp_path, n=1, reps=0))
    assert len(run_plan.jobs) == 2
    assert all(job["steps"][0][2:5] == ((1, 2, 1), (128, 1, 1), 32768)
               for job in run_plan.jobs)


def test_compiler_identity_covers_dispatch_policy_code(monkeypatch):
    identity = dispatch.compiler_identity()
    original = dispatch.Path.read_bytes
    for target in (dispatch.Path(dispatch.__file__).resolve(),):
        monkeypatch.setattr(dispatch.Path, "read_bytes",
                            lambda path: original(path) + (b"changed" if path.resolve() == target else b""))
        assert dispatch.compiler_identity() != identity
        monkeypatch.setattr(dispatch.Path, "read_bytes", original)
