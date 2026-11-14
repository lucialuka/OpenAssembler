"""Source discovery stays static, shorthand preserves dispatch, failures are reviewable."""

import json

from openassembler.cli import main
from openassembler import build


def test_inspect_mixed_repository_never_imports_host_code(tmp_path, capsys):
    (tmp_path / "host.py").write_text('raise RuntimeError("must not execute")\n')
    (tmp_path / "k.ptx").write_text(".visible .entry kernel() { ret; }")
    assert main(["inspect", str(tmp_path), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["kernel_candidates"][0]["path"] == "k.ptx"
    assert report["other_files"] == [
        {"path": "host.py", "reason": ".py without @kernel"}
    ]
    assert not (tmp_path / "output").exists()


def test_path_shorthand_and_default_directory_preserve_compile_flags(
    tmp_path, monkeypatch
):
    path = tmp_path / "k.ptx"
    path.write_text(".visible .entry kernel() { ret; }")
    calls = []

    def compile_source(src, out, options, name):
        calls.append((src, options.verify))
        return build.CompileResult(0)

    monkeypatch.setattr(build, "compile_paths", compile_source)
    assert main([str(path), "--verify"]) == 0
    monkeypatch.chdir(tmp_path)
    assert main(["compile"]) == 0
    assert [(p.resolve(), verify) for p, verify in calls] == [
        (path, True),
        (tmp_path, False),
    ]


def test_failed_compile_saves_reason_and_coverage_but_dry_run_writes_nothing(
    tmp_path, capsys
):
    (tmp_path / "host.py").write_text('raise RuntimeError("must not import")\n')
    assert main(["compile", str(tmp_path), "--dry-run"]) == 1
    assert not (tmp_path / "output").exists()
    assert main(["compile", str(tmp_path)]) == 1
    report = json.loads((tmp_path / "output/compile-error.json").read_text())
    assert report["status"] == "failed"
    assert "nothing to compile" in report["error"]
    assert not report["coverage"]["ready_for_capture_or_compilation"]
    assert "build diagnostic:" in capsys.readouterr().err


def test_internal_compiler_failure_produces_a_report_instead_of_losing_context(tmp_path, monkeypatch, capsys):
    (tmp_path / 'k.ptx').write_text('.visible .entry kernel() { ret; }')
    def fail(*args):
        raise RuntimeError('unexpected encoder failure')
    monkeypatch.setattr(build, 'compile_paths', fail)
    assert main(['compile', str(tmp_path)]) == 1
    report = json.loads((tmp_path / 'output/compile-error.json').read_text())
    assert 'RuntimeError: unexpected encoder failure' in report['error']
    assert 'internal compiler error' in capsys.readouterr().err
