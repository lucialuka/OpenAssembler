"""Native-first repository frontend policy."""
from __future__ import annotations

from pathlib import Path

import pytest

from openassembler import build
from openassembler.cli import main
from openassembler.repository import inspect, select_frontend
from openassembler.torchfront import frontend as torchfront


def configured_repo(tmp_path: Path, *, native: bool) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "openassembler.toml").write_text(
        '[application]\nfrontend="torch"\nentrypoint="train.py"\n'
    )
    (root / "train.py").write_text("print('host training')\n")
    if native:
        (root / "kernels.ptx").write_text(
            ".version 8.7\n.target sm_80\n.address_size 64\n"
            ".visible .entry k() { ret; }\n"
        )
    return root


def test_auto_is_native_first_when_application_and_native_sources_coexist(tmp_path):
    root = configured_repo(tmp_path, native=True)
    kind, app, plan = select_frontend(root, root / "output", "auto")
    assert kind == "native" and app is None
    assert plan is not None and plan.has_native and plan.has_application
    assert [c.rel for c in plan.native] == ["kernels.ptx"]


def test_auto_does_not_implicitly_capture_a_pytorch_only_repo(tmp_path, capsys):
    root = configured_repo(tmp_path, native=False)
    assert main(["compile", str(root), "--dry-run"]) == 1
    error = capsys.readouterr().err
    assert "--frontend torch" in error
    assert "not implicitly lowered" in error


@pytest.mark.parametrize("frontend_name", ["torch", "triton", "inductor"])
def test_application_frontend_requires_explicit_selection(tmp_path, frontend_name, monkeypatch, capsys):
    root = configured_repo(tmp_path, native=False)
    observed = []

    def dry_compile(root_arg, out_dir, **kwargs):
        observed.append((root_arg, out_dir, kwargs))
        return type("Result", (), {"rc": 0})()

    monkeypatch.setattr(torchfront, "compile_application", dry_compile)
    assert main(["compile", str(root), "--frontend", frontend_name, "--dry-run"]) == 0
    assert observed and observed[0][0] == root
    assert observed[0][2]["dry_run"] is True
    assert not capsys.readouterr().err


def test_native_source_file_inside_configured_repo_never_selects_application(tmp_path, monkeypatch):
    root = configured_repo(tmp_path, native=False)
    source = root / "kernel.ptx"
    source.write_text(".version 8.7\n.target sm_80\n.address_size 64\n.visible .entry k() { ret; }\n")
    observed = []

    def compile_file(src, out, options, name):
        observed.append((src, out))
        return build.CompileResult(0)

    monkeypatch.setattr(build, "compile_paths", compile_file)
    assert main(["compile", str(source)]) == 0
    assert observed and observed[0][0] == source


def test_repository_inspection_reports_build_system_and_static_kernel_context(tmp_path):
    root = configured_repo(tmp_path, native=True)
    (root / "CMakeLists.txt").write_text("project(example)\n")
    (root / "include.h").write_text("#define VALUE 1\n")
    (root / "kernels.cu").write_text('#include "include.h"\n__global__ void x() {}\n')
    plan = inspect(root, root / "output")
    assert plan.build_systems == ("cmake",)
    assert plan.application_frontend == "torch"
    assert plan.used_git is False
    assert plan.dependencies["kernels.cu"] == ("include.h",)
