"""The 0.2 CLI contract: source names are separate from output paths, and each
compile produces a manifest under an output root with mirrored cubin/SASS trees.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openassembler import __version__, oracle
from openassembler.cli import main
from openassembler.project import ProjectError, output_names, single_output_name, walk

ROOT = Path(__file__).resolve().parent.parent
needs_toolkit = pytest.mark.skipif(not all(oracle.available(t) for t in ("ptxas", "nvdisasm")),
                                   reason=oracle.NEEDS_TOOLKIT)


def test_version_and_startup_commands(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert __version__ in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        main(["compile", "--help"])
    assert e.value.code == 0
    help_text = capsys.readouterr().out
    assert "whole source tree" in help_text
    assert "output/cubin" in help_text
    assert "-o NAME" in help_text


def test_output_name_is_a_name_not_a_path():
    assert single_output_name(Path("k.ptx"), None) == ("k", None)
    assert single_output_name(Path("k.ptx"), "answer") == ("answer", None)
    name, note = single_output_name(Path("k.ptx"), "answer.cubin")
    assert name == "answer" and "without extension" in note
    with pytest.raises(ProjectError, match="takes a name, not a path"):
        single_output_name(Path("k.ptx"), "nested/answer")
    with pytest.raises(ProjectError, match="usable output name"):
        single_output_name(Path("k.ptx"), "..")


def test_repo_names_preserve_relative_tree_and_same_stem_extensions():
    assert output_names(["kernels/gemm.cu", "kernels/other.py"]) == {
        "kernels/gemm.cu": "kernels/gemm",
        "kernels/other.py": "kernels/other",
    }
    assert output_names(["a.cu", "a.py"]) == {"a.cu": "a.cu", "a.py": "a.py"}
    assert output_names(["x/a.cu", "y/a.py"]) == {"x/a.cu": "x/a", "y/a.py": "y/a"}


def test_repo_walk_ignores_hidden_dirs_and_output(tmp_path):
    src = tmp_path / "repo"
    (src / "kernels").mkdir(parents=True)
    (src / ".git").mkdir()
    (src / "node_modules").mkdir()
    out = tmp_path / "build"
    out.mkdir()
    (src / "kernels" / "a.ptx").write_text(".visible .entry a() { ret; }")
    (src / ".git" / "hidden.ptx").write_text(".visible .entry hidden() { ret; }")
    (src / "node_modules" / "vendor.ptx").write_text(".visible .entry vendor() { ret; }")
    (out / "old.ptx").write_text(".visible .entry old() { ret; }")
    files, used_git = walk(src, out)
    assert not used_git
    assert files == [src / "kernels" / "a.ptx"]


def test_cli_rejects_missing_source_and_non_source_extensions(tmp_path, capsys):
    assert main(["compile", str(tmp_path / "missing.ptx")]) == 1
    assert "does not exist" in capsys.readouterr().err
    script = tmp_path / "train.sh"
    script.write_text("#!/bin/sh\necho hello\n")
    assert main(["compile", str(script)]) == 1
    assert "supported" in capsys.readouterr().err


def test_compile_unknown_arch_is_a_clean_error(tmp_path, capsys):
    assert main(["compile", str(ROOT / "examples/vector_add.py"), "--arch", "sm_90",
                 "--out-dir", str(tmp_path)]) == 1
    assert "no SASS encoder data for 'sm_90'" in capsys.readouterr().err


@needs_toolkit
def test_single_compile_writes_cubin_sass_and_manifest(tmp_path, capsys):
    assert main(["compile", str(ROOT / "examples/div_by_const.ptx"), "-o", "quotient",
                 "--out-dir", str(tmp_path), "--verify"]) == 0
    cubin = tmp_path / "cubin" / "quotient.cubin"
    sass = tmp_path / "sass" / "quotient.sass"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert cubin.is_file() and sass.is_file()
    assert ".text.div_by_const" in sass.read_text() and "IMAD.HI" in sass.read_text()
    assert manifest["openassembler"] == __version__
    assert manifest["entries"][0]["cubin"] == "cubin/quotient.cubin"
    assert manifest["entries"][0]["sass"] == "sass/quotient.sass"
    assert manifest["entries"][0]["kernels"][0]["verify"] == "ok"
    assert "OK:" in capsys.readouterr().out


@needs_toolkit
def test_python_and_cuda_c_compile_to_output_tree(tmp_path):
    py_out = tmp_path / "python"
    assert main(["compile", str(ROOT / "examples/vector_add.py"), "--out-dir", str(py_out)]) == 0
    assert (py_out / "cubin/vector_add.cubin").is_file()
    assert (py_out / "sass/vector_add.sass").is_file()

    src = tmp_path / "external.cu"
    src.write_text((ROOT / "examples/vector_add.cu").read_text())
    cu_out = tmp_path / "cuda"
    assert main(["compile", str(src), "--out-dir", str(cu_out), "-o", "custom"]) == 0
    assert (cu_out / "cubin/custom.cubin").is_file()
    assert (cu_out / "sass/custom.sass").is_file()
    assert not src.with_suffix(".ptx").exists()


@needs_toolkit
def test_no_sass_sets_manifest_null_and_writes_cubin(tmp_path):
    assert main(["compile", str(ROOT / "examples/div_by_const.ptx"), "--out-dir", str(tmp_path),
                 "--no-sass"]) == 0
    assert (tmp_path / "cubin/div_by_const.cubin").is_file()
    assert not (tmp_path / "sass/div_by_const.sass").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["entries"][0]["sass"] is None


@needs_toolkit
def test_repo_compile_mirrors_paths_and_skips_non_kernel_files(tmp_path, capsys):
    src = tmp_path / "repo"
    (src / "kernels").mkdir(parents=True)
    (src / "kernels" / "sum.ptx").write_text((ROOT / "examples/div_by_const.ptx").read_text())
    (src / "README.md").write_text("repository metadata")
    (src / "script.sh").write_text("echo not a kernel")
    out = tmp_path / "artifacts"
    assert main(["compile", str(src), "--out-dir", str(out)]) == 0
    assert (out / "cubin/kernels/sum.cubin").is_file()
    assert (out / "sass/kernels/sum.sass").is_file()
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["entries"]) == 1
    summary = capsys.readouterr().out
    assert "files skipped" in summary


@needs_toolkit
def test_recompile_removes_only_manifest_owned_stale_outputs(tmp_path):
    src = tmp_path / "repo"
    src.mkdir()
    kernel = src / "k.ptx"
    kernel.write_text((ROOT / "examples/div_by_const.ptx").read_text())
    out = tmp_path / "output"
    assert main(["compile", str(src), "--out-dir", str(out)]) == 0
    stale = out / "cubin/unrelated.cubin"
    stale.parent.mkdir(exist_ok=True)
    stale.write_bytes(b"user file")
    kernel.unlink()
    # Empty source trees are an error and do not authorize deleting prior/user files.
    assert main(["compile", str(src), "--out-dir", str(out)]) == 1
    assert stale.read_bytes() == b"user file"
    assert (out / "cubin/k.cubin").is_file()


@needs_toolkit
def test_repo_compile_excludes_output_inside_source_tree(tmp_path, capsys):
    src = tmp_path / "repo"
    src.mkdir()
    (src / "k.ptx").write_text(".version 8.7\n.target sm_80\n.address_size 64\n.visible .entry k() { ret; }\n")
    out = src / "output"
    assert main(["compile", str(src), "--out-dir", str(out)]) == 0
    assert (out / "cubin/k.cubin").is_file()
    assert not (out / "output/cubin/k.cubin").exists()


@needs_toolkit
def test_failed_single_rebuild_marks_prior_artifact_unrunnable(tmp_path, capsys):
    from openassembler.runner import RunError, RunOptions, build_plan
    src = tmp_path / 'k.ptx'
    src.write_text((ROOT / 'examples/div_by_const.ptx').read_text())
    out = tmp_path / 'output'
    assert main(['compile', str(src), '--out-dir', str(out)]) == 0
    assert (out / 'cubin/k.cubin').is_file()
    src.write_text('.version 8.7\n.target sm_80\n.address_size 64\n.entry k( {')
    assert main(['compile', str(src), '--out-dir', str(out)]) == 1
    assert (out / 'cubin/k.cubin').is_file()
    with pytest.raises(RunError, match='latest build.*failed'):
        build_plan(RunOptions(out_dir=out, dry_run=True))
    capsys.readouterr()
