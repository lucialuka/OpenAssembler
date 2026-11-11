"""Exact-shape rowwise CTA policy dispatch and runner launch contracts."""
from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from openassembler import project
from openassembler.cli import main
from openassembler.kernels.reductions import ReductionPlan, generate
from openassembler.rowwise_dispatch import TABLE, compiler_identity, select_plan
from openassembler.runner import RunOptions, build_plan
from openassembler.rowwise import contract_from_source, source_for


WINNERS = {
    ("softmax", 129, 1023): 2,
    ("softmax", 4096, 1): 8,
    ("softmax", 4096, 32): 8,
    ("layernorm", 4096, 1): 8,
    ("layernorm", 4096, 32): 8,
}


@pytest.mark.parametrize("shape,warps", WINNERS.items())
def test_auto_dispatch_is_exact_to_promoted_operation_shape(shape, warps):
    op, rows, width = shape
    plan, decision = select_plan(ReductionPlan(rows, width, op))
    assert plan.warps_per_cta == warps
    assert decision["mode"] == "measured"
    assert decision["policy"] == f"w{warps}"
    assert decision["evidence"]
    assert decision["measured_speedup"][0] >= 1.03
    assert generate(plan) == generate(ReductionPlan(rows, width, op, warps_per_cta=warps))


@pytest.mark.parametrize("shape", [("softmax", 129, 1024), ("layernorm", 4096, 513),
                                    ("softmax", 4096, 1024)])
def test_auto_dispatch_unknown_shape_keeps_four_warp_baseline(shape):
    op, rows, width = shape
    plan, decision = select_plan(ReductionPlan(rows, width, op))
    assert plan.warps_per_cta == 4
    assert decision["policy"] == "w4"
    assert decision["reason"] == "no unique exact rowwise entry"


def test_stale_or_corrupt_dispatch_table_fails_closed():
    plan = ReductionPlan(4096, 32, "softmax")
    table = json.loads(TABLE.read_text())
    stale = {**table, "compiler_identity": "0" * 64}
    selected, decision = select_plan(plan, table=stale)
    assert selected.warps_per_cta == 4
    assert decision["reason"] == "rowwise table belongs to a different compiler"
    corrupted = json.loads(json.dumps(table))
    matching = next(entry for entry in corrupted["entries"]
                    if entry["contract"]["op"] == "softmax"
                    and entry["contract"]["rows"] == 4096
                    and entry["contract"]["width"] == 32)
    matching["ptx_body_sha256"] = "0" * 64
    selected, decision = select_plan(plan, table=corrupted)
    assert selected.warps_per_cta == 4
    assert decision["reason"] == "invalid rowwise dispatch entry"
    duplicated = json.loads(json.dumps(table))
    original = next(entry for entry in duplicated["entries"]
                    if entry["contract"]["op"] == "softmax"
                    and entry["contract"]["rows"] == 4096
                    and entry["contract"]["width"] == 32)
    duplicated["entries"].append(dict(original))
    selected, decision = select_plan(plan, table=duplicated)
    assert selected.warps_per_cta == 4
    assert decision["reason"] == "duplicate exact rowwise entry"


def test_promoted_table_binds_current_rowwise_identity_and_body_hashes():
    table = json.loads(TABLE.read_text())
    assert table["compiler_identity"] == compiler_identity()
    assert len(table["entries"]) == 5
    for entry in table["entries"]:
        c = entry["contract"]
        plan = ReductionPlan(c["rows"], c["width"], c["op"], epsilon=c["epsilon_f32"],
                             arch=c["arch"], warps_per_cta=entry["warps_per_cta"])
        assert entry["ptx_body_sha256"] == entry["current_generator_body_sha256"]
        assert entry["status"] == "promoted" and entry["gates_passed"] is True
        assert len(entry["evidence"]) == 2
        assert all(min(e.get("heldout_speedup_w4_over_candidate", e.get("selection_speedup_w4_over_candidate", [0]))) >= 1.03
                   for e in entry["evidence"])


def _capture_compile(monkeypatch):
    captured = {}
    def fake_compile(plan, out_dir, name=None, **kwargs):
        captured["plan"] = plan
        captured["dispatch"] = kwargs["dispatch"]
        return 0
    monkeypatch.setattr("openassembler.rowwise.compile_rowwise", fake_compile)
    return captured


def test_cli_automatic_dispatch_and_manual_warp_override(monkeypatch, tmp_path):
    captured = _capture_compile(monkeypatch)
    assert main(["rowwise", "--op", "softmax", "--rows", "4096", "--width", "32",
                 "--out-dir", str(tmp_path)]) == 0
    assert captured["plan"].warps_per_cta == 8
    assert captured["dispatch"]["mode"] == "measured"

    assert main(["rowwise", "--op", "softmax", "--rows", "4096", "--width", "32",
                 "--dispatch", "auto", "--warps-per-cta", "2", "--out-dir", str(tmp_path)]) == 0
    assert captured["plan"].warps_per_cta == 2
    assert captured["dispatch"]["mode"] == "manual"
    assert captured["dispatch"]["policy"] == "w2"


def test_cli_emit_roundtrip_preserves_default_w4_body_and_exact_dispatch_metadata(tmp_path):
    args = ["rowwise", "--op", "softmax", "--rows", "4096", "--width", "32",
            "--emit-only", "--out-dir", str(tmp_path)]
    assert main(args) == 0
    source_path = tmp_path / ".build/generated/softmax.ptx"
    emitted = source_path.read_text()
    contract = contract_from_source(emitted)
    assert contract["warps_per_cta"] == 8
    assert contract["launch"] == {"grid": [512, 1, 1], "threads": 256, "smem": 0}
    assert emitted.split("\n", 1)[1] == generate(ReductionPlan(4096, 32, "softmax", warps_per_cta=8))
    decision = json.loads((tmp_path / "rowwise-dispatch.json").read_text())
    assert decision["policy"] == "w8"
    assert decision["plan"]["warps_per_cta"] == 8

    fallback = tmp_path / "fallback"
    assert main(["rowwise", "--op", "softmax", "--rows", "7", "--width", "33",
                 "--emit-only", "--out-dir", str(fallback)]) == 0
    fallback_body = (fallback / ".build/generated/softmax.ptx").read_text().split("\n", 1)[1]
    assert fallback_body == generate(ReductionPlan(7, 33, "softmax"))
    assert ReductionPlan(7, 33, "softmax").warps_per_cta == 4
    # Existing W4 source annotations do not grow a new field, and the actual
    # generated PTX body remains exactly the pre-dispatch generator output.
    assert '"warps_per_cta"' not in (fallback / ".build/generated/softmax.ptx").read_text().split("\n", 1)[0]


@pytest.mark.parametrize("warps", [1, 2, 8])
def test_runner_uses_rowwise_warp_geometry_in_launch_contract(tmp_path, warps):
    plan = ReductionPlan(7, 33, "softmax", warps_per_cta=warps)
    source = tmp_path / "softmax.ptx"
    source.write_text(source_for(plan))
    (tmp_path / "ours.cubin").write_bytes(b"OA cubin")
    (tmp_path / "ref.cubin").write_bytes(b"ptxas cubin")
    kernel = {"name": "softmax", "params": [["u64", "X"], ["u64", "Y"]],
              "compiled_by": "openassembler", "rowwise": contract_from_source(source.read_text())}
    project.save_manifest(tmp_path, tmp_path, "sm_80", "opt", [{
        "source": str(source), "cubin": "ours.cubin", "reference": "ref.cubin", "kernels": [kernel]
    }])
    run_plan = build_plan(RunOptions(out_dir=tmp_path, n=1, reps=0))[0]
    expected_grid = (7 + warps - 1) // warps
    assert len(run_plan.jobs) == 2
    assert all(job["steps"][0][2:5] == ((expected_grid, 1, 1), (warps*32, 1, 1), 0)
               for job in run_plan.jobs)
