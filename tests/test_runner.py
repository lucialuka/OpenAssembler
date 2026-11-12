"""Runner planning and result handling can be tested without contacting a GPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.native_fixture import save_native_manifest

from openassembler.runner import (Case, Plan, RunError, RunOptions, _first_diff, compare, load_spec, report,
                                  run_command)


def test_first_diff_reports_word_index_and_handles_length_changes():
    assert _first_diff(b"abcd1234", b"abcd1234") is None
    assert _first_diff(b"abcd1234", b"abcd5678") == 1
    assert _first_diff(b"abcd", b"abcd1234") == 1


def test_compare_synthetic_results_and_faults():
    plan = Plan(cases=[Case("add", "k.ptx", 0, 1, ["out"]),
                       Case("bad", "bad.ptx", 2, 3, ["out"])])
    result = compare(plan, {
        0: {"out": {"out": [b"abcd"]}, "time_ms": {"median": 2.0}},
        1: {"out": {"out": [b"abcd"]}, "time_ms": {"median": 1.0}},
        2: {"error": "illegal instruction"},
        3: {"out": {"out": [b"abcd"]}},
    })
    assert [x.status for x in result] == ["OK", "FAULT"]
    assert result[0].speedup == 0.5
    assert "OUR cubin failed" in result[1].detail


def test_report_returns_failure_for_mismatch(capsys):
    plan = Plan(cases=[Case("add", "k.cu", 0, 1, ["out"])])
    outs = compare(plan, {0: {"out": {"out": [b"0000"]}}, 1: {"out": {"out": [b"1111"]}}})
    assert report(outs) == 1
    text = capsys.readouterr().out
    assert "MISMATCH" in text and "ours 0x30303030" in text


def test_load_spec_errors_are_user_facing(tmp_path):
    with pytest.raises(RunError, match="does not exist"):
        load_spec(tmp_path / "absent.json")
    malformed = tmp_path / "bad.json"
    malformed.write_text("[]")
    with pytest.raises(RunError, match="needs"):
        load_spec(malformed)


def test_modal_dry_run_builds_plan_without_credentials(tmp_path, monkeypatch, capsys):
    import openassembler.runner as runner
    from openassembler import project

    source = tmp_path / "kernel.ptx"
    source.write_text(".visible .entry k(.param .u64 out, .param .u64 in, .param .u32 n) { ret; }")
    out = tmp_path / "output"
    (out / "cubin").mkdir(parents=True)
    (out / ".build" / "reference").mkdir(parents=True)
    (out / "cubin/k.cubin").write_bytes(b"ours")
    (out / ".build/reference/k.cubin").write_bytes(b"reference")
    entry = {"source": str(source), "source_sha256": project.sha256_file(source), "language": "ptx",
             "cubin": "cubin/k.cubin", "sass": None, "ptx": ".build/ptx/k.ptx",
             "reference": ".build/reference/k.cubin",
             "kernels": [{"name": "k", "params": [["u64", "out"], ["u64", "in"], ["u32", "n"]],
                          "compiled_by": "openassembler"}]}
    save_native_manifest(out, tmp_path, "sm_80", "opt", [entry])
    monkeypatch.setattr(runner, "modal_ready", lambda: "no modal credentials")
    assert run_command(RunOptions(out_dir=out, backend="modal", dry_run=True, n=32, reps=0)) == 0
    text = capsys.readouterr().out
    assert "dry run" in text and "nothing was run" in text
    assert "no modal credentials" in text
    assert "no GPU" in text or "would run" in text


def test_run_reports_missing_manifest(tmp_path, capsys):
    assert run_command(RunOptions(out_dir=tmp_path / "missing", backend="modal", dry_run=True)) == 1
    assert "no output folder" in capsys.readouterr().err



def test_missing_outputs_cannot_pass_comparison():
    plan = Plan(cases=[Case('k', 'k.py', 0, 1, ['out'])])
    assert compare(plan, {0: {'out': {}}, 1: {'out': {}}})[0].status == 'MISMATCH'


def test_modal_command_preserves_watchdog():
    from openassembler.runner import modal_cmd
    cmd = modal_cmd(Path('jobs'), Path('results'), 'A100-40GB', 120, 7)
    assert cmd[cmd.index('--job-timeout') + 1] == '7'


@pytest.mark.parametrize('dimension', [True, [False], [1, 0], [], -1])
def test_invalid_launch_dimensions_are_rejected(dimension):
    from openassembler.runner import _dim
    with pytest.raises(RunError):
        _dim(dimension, 'block')


def test_manifest_cannot_read_artifacts_outside_output(tmp_path):
    from openassembler import project
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'manifest.json').write_text(json.dumps({'schema': 1, 'root': str(tmp_path), 'entries': [
        {'source': 'k.py', 'cubin': '../private.cubin', 'kernels': []}]}))
    with pytest.raises(project.ManifestError, match='path'):
        project.load_manifest(out)


def test_spec_rejects_fractional_integer_initializers(tmp_path):
    spec = tmp_path / "fractional.json"
    spec.write_text(json.dumps({"buffers": {"x": {"dtype": "i32", "count": 1, "init": [1.5]}},
                               "steps": [{"kernel": "k"}]}))
    with pytest.raises(RunError, match="initialization values must be integers"):
        import openassembler.runner as runner
        runner.spec_case(load_spec(spec), {"entries": []}, tmp_path, 0, Plan(), "fractional")
