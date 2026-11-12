"""Checked cooperative reduction source, CLI, launch and analytic smoke contracts."""
from __future__ import annotations

import json
import struct

import pytest
from tests.native_fixture import save_native_manifest

from openassembler import oracle, project
from openassembler.cli import main
from openassembler.kernels.reductions import ReductionPlan
from openassembler.rowwise import MARKER, contract_from_source, source_for
from openassembler.runner import RunError, RunOptions, build_plan, compare


@pytest.mark.parametrize("op", ["softmax", "layernorm"])
def test_emit_and_exact_body_contract(op, tmp_path):
    assert main(["rowwise", "--op", op, "--rows", "7", "--width", "33",
                 "--out-dir", str(tmp_path), "--emit-only"]) == 0
    source = (tmp_path / f".build/generated/{op}.ptx").read_text()
    contract = contract_from_source(source)
    assert contract["op"] == contract["kernel"] == op
    assert contract["launch"] == {"grid": [2, 1, 1], "threads": 128, "smem": 0}
    assert contract["input_dtype"] == contract["output_dtype"] == "f32"
    assert contract["affine"] is False
    assert contract_from_source(source.replace("shfl.sync.bfly.b32", "shfl.sync.down.b32", 1)) is None
    assert contract_from_source(source.split("\n", 1)[1]) is None
    assert contract_from_source(MARKER + "{bad}\n") is None
    assert contract_from_source(MARKER + "null\n") is None


@pytest.mark.parametrize("arguments", [
    ["--rows", "0", "--width", "128"], ["--rows", "1", "--width", "1025"],
    ["--rows", "1", "--width", "128", "--epsilon", "0"],
    ["--rows", "1", "--width", "128", "--epsilon", "1e-320"],
    ["--rows", "1", "--width", "128", "--epsilon", "1e-40"],
    ["--rows", "1", "--width", "128", "--epsilon", "1e308"],
    ["--rows", "4294967296", "--width", "1"],
    ["--rows", "1", "--width", "128", "--arch", "sm_90"],
    ["--rows", "1", "--width", "128", "--out", "../escape"],
])
def test_invalid_plan_fails_before_writing(arguments, tmp_path):
    assert main(["rowwise", "--op", "layernorm", *arguments, "--out-dir", str(tmp_path)]) == 1
    assert not list(tmp_path.iterdir())


def _rowwise_plan(tmp_path, op, mutate=None):
    plan = ReductionPlan(7, 33, op)
    source = tmp_path / "row.ptx"
    source.write_text(source_for(plan))
    for name in ("ours.cubin", "ref.cubin"):
        (tmp_path / name).write_bytes(b"test cubin")
    kernel = {"name": op, "params": [["u64", "X"], ["u64", "Y"]],
              "compiled_by": "openassembler", "rowwise": contract_from_source(source.read_text())}
    if mutate:
        mutate(kernel)
    save_native_manifest(tmp_path, tmp_path, "sm_80", "opt", [{
        "source": str(source), "cubin": "ours.cubin", "reference": "ref.cubin", "kernels": [kernel]}])
    return build_plan(RunOptions(out_dir=tmp_path, n=1, reps=0))[0]


@pytest.mark.parametrize("op", ["softmax", "layernorm"])
def test_run_has_exact_shape_zero_input_and_poisoned_output(op, tmp_path):
    plan = _rowwise_plan(tmp_path, op)
    assert len(plan.jobs) == 1
    for job in plan.jobs:
        assert job["buffers"] == {"X": 924, "Y": 924}
        assert job["init"] == {}
        assert job["init_rand"] == {}
        assert job["init_byte"] == {"Y": 255}
        assert job["steps"][0][2:5] == ((2, 1, 1), (128, 1, 1), 0)
        assert job["outputs"] == {"Y": [(0, 924)]}
    assert plan.cases[0].ref is None
    assert all("ptxas" not in name for job in plan.jobs for name in job["modules"])
    assert plan.cases[0].expected_f32_constant == {"Y": 1 / 33 if op == "softmax" else 0.0}


@pytest.mark.parametrize("op", ["softmax", "layernorm"])
def test_analytic_smoke_rejects_both_wrong_and_nonfinite(op, tmp_path):
    plan = _rowwise_plan(tmp_path, op)
    expected = 1 / 33 if op == "softmax" else 0.0
    def outputs(value):
        return {i: {"out": {"Y": [struct.pack("<f", value) * 231]}} for i in (0, 1)}
    assert compare(plan, outputs(expected))[0].status == "OK"
    assert compare(plan, outputs(1.0))[0].status == "MISMATCH"
    assert compare(plan, outputs(float("nan")))[0].status == "MISMATCH"
    matching = outputs(expected)
    matching[0]["out"]["Y"] = [struct.pack("<f", expected)]
    assert compare(plan, matching)[0].status == "MISMATCH"


def test_rowwise_manifest_cannot_change_launch_or_abi(tmp_path):
    def wrong_abi(kernel):
        kernel["params"][0][1] = "Other"
    with pytest.raises(RunError, match="invalid rowwise launch contract"):
        _rowwise_plan(tmp_path, "softmax", wrong_abi)


def test_saved_report_retains_analytic_smoke_scope(tmp_path, monkeypatch):
    from openassembler import runner
    _rowwise_plan(tmp_path, "layernorm")
    monkeypatch.setattr(runner, "execute", lambda plan, options: (
        {i: {"out": {"Y": [b"\0" * 924]}} for i in (0, 1)}, "test device"))
    assert runner.run_command(RunOptions(out_dir=tmp_path, reps=0)) == 0
    case = json.loads((tmp_path / "run.json").read_text())["cases"][0]
    assert "zero-input analytic smoke" in case["note"]
    assert case["analytic_smoke"] == {"expected_f32_constant": {"Y": 0.0},
                                       "rtol": 5e-5, "atol": 1e-6, "require_finite": True}


def test_device_byte_initialization_is_nan_and_rejects_invalid_values():
    from types import SimpleNamespace
    from openassembler.gpu.cuda_driver import run_plan_job
    class FakeDriver:
        cu = SimpleNamespace(cuCtxSynchronize=lambda: 0)
        def __init__(self):
            self.data = {}
            self.freed = []
        def alloc(self, size):
            self.data[4096] = bytes(size)
            return 4096
        def memset(self, pointer, value, size):
            self.data[pointer] = bytes([value]) * size
        def dtoh(self, pointer, size):
            return self.data[pointer][:size]
        def free(self, pointer):
            self.freed.append(pointer)
        def ck(self, status, label):
            assert status == 0, label
    job = {"modules": {}, "buffers": {"Y": 4}, "init_byte": {"Y": 255},
           "steps": [], "outputs": {"Y": [(0, 4)]}, "reps": 0}
    driver = FakeDriver()
    assert run_plan_job(driver, job)["out"]["Y"] == [b"\xff" * 4]
    assert driver.freed == [4096]
    for invalid in ({"Y": 256}, {"Y": True}, {"missing": 255}):
        with pytest.raises(ValueError, match="byte initialization"):
            run_plan_job(FakeDriver(), {**job, "init_byte": invalid})


@pytest.mark.skipif(not all(oracle.available(t) for t in ("ptxas", "nvdisasm")), reason=oracle.NEEDS_TOOLKIT)
@pytest.mark.parametrize("op", ["softmax", "layernorm"])
def test_native_cli_compile_and_plain_recompile_preserve_contract(op, tmp_path):
    out = tmp_path / "first"
    assert main(["rowwise", "--op", op, "--rows", "7", "--width", "33",
                 "--out-dir", str(out)]) == 0
    entry = json.loads((out / "manifest.json").read_text())["entries"][0]
    kernel = entry["kernels"][0]
    assert kernel["compiled_by"] == "openassembler"
    assert kernel["rowwise"]["rows"] == 7
    sass = (out / entry["sass"]).read_text()
    assert "SHFL.BFLY" in sass and "WARPSYNC" in sass
    again = tmp_path / "again"
    assert main(["compile", str(out / f".build/generated/{op}.ptx"), "--out-dir", str(again)]) == 0
    second = json.loads((again / "manifest.json").read_text())["entries"][0]
    assert second["kernels"][0]["rowwise"] == kernel["rowwise"]
    assert (out / entry["cubin"]).read_bytes() == (again / second["cubin"]).read_bytes()
