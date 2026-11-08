"""Repo-level PyTorch application frontend contracts that do not need a GPU."""

from __future__ import annotations

import io
from pathlib import Path
import shlex
import zipfile

import pytest

from openassembler.cli import main
from openassembler.torchfront.archive import unpack
from openassembler.project import ManifestError, load_manifest
from openassembler.torchfront.frontend import ApplicationError, load_config

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "testcases" / "test_whole_repo"


def test_transformer_repo_dry_run_selects_its_configured_workload(tmp_path, capsys):
    out = tmp_path / "output"
    assert main(["compile", str(EXAMPLE), "--frontend", "torch", "--out-dir", str(out), "--verify", "--dry-run"]) == 0
    report = capsys.readouterr().out
    assert "PyTorch application test_whole_repo: 5 files" in report
    config = load_config(EXAMPLE / "openassembler.toml")
    assert shlex.join([config["entrypoint"], *config["args"]]) in report
    assert "eager/compiled state verification on" in report
    assert not out.exists()


def test_application_source_bundle_ignores_generated_output_and_caches(tmp_path):
    root = tmp_path / "app"
    root.mkdir()
    (root / "train.py").write_text("print('ok')\n")
    (root / "openassembler.toml").write_text("[application]\nfrontend='torch'\nentrypoint='train.py'\n")
    output = root / "output"
    (output / ".build").mkdir(parents=True)
    (output / ".build" / "stale.ptx").write_text("stale")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "stale.pyc").write_bytes(b"stale")
    from openassembler.torchfront.frontend import project_files
    files = project_files(root, tmp_path / "other-output")
    assert sorted(files) == ["openassembler.toml", "train.py"]


def test_application_config_rejects_parent_entrypoint(tmp_path):
    config = tmp_path / "openassembler.toml"
    config.write_text('[application]\nfrontend="torch"\nentrypoint="../outside.py"\n')
    with pytest.raises(ApplicationError, match="relative file path"):
        load_config(config)


def test_application_config_rejects_parent_requirements(tmp_path):
    config = tmp_path / "openassembler.toml"
    config.write_text('[application]\nfrontend="torch"\nentrypoint="train.py"\nrequirements=["../outside.txt"]\n')
    with pytest.raises(ApplicationError, match="relative file paths"):
        load_config(config)


def test_source_digest_is_order_independent_and_content_sensitive():
    from openassembler.torchfront.frontend import source_digest

    first = {"train.py": b"print(1)", "requirements.txt": b"torch"}
    assert source_digest(first) == source_digest(dict(reversed(list(first.items()))))
    assert source_digest(first) != source_digest({**first, "train.py": b"print(2)"})


def test_archive_unpack_rejects_path_traversal(tmp_path):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("../outside.txt", "bad")
    with pytest.raises(ValueError, match="unsafe application archive member"):
        unpack(data.getvalue(), tmp_path / "source")
    assert not (tmp_path / "outside.txt").exists()


def test_archive_unpack_rejects_duplicate_paths(tmp_path):
    data = io.BytesIO()
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr("src/main.py", "first")
            archive.writestr("src/main.py", "second")
    with pytest.raises(ValueError, match="unsafe application archive member"):
        unpack(data.getvalue(), tmp_path / "source")


def test_archive_unpack_rejects_normalized_path_aliases(tmp_path):
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w') as archive:
        archive.writestr('src/main.py', 'first')
        archive.writestr('src//main.py', 'second')
    with pytest.raises(ValueError, match='unsafe application archive member'):
        unpack(data.getvalue(), tmp_path / 'source')


def test_manifest_rejects_application_paths_that_escape_output(tmp_path):
    import json

    manifest = {"schema": 1, "root": str(tmp_path), "entries": [],
                "applications": [{"name": "unsafe", "source_bundle": "../source.zip",
                                  "cache": ".build/cache.zip", "report": ".build/report.json",
                                  "graphs": ".build/graphs.json", "config": "app.json",
                                  "compiled_by": "torch-inductor-triton", "runtime": {},
                                  "source_bundle_sha256": "0" * 64, "cache_sha256": "0" * 64,
                                  "entrypoint": "train.py", "args": [], "kernels": []}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ManifestError, match="invalid output artifact path"):
        load_manifest(tmp_path)
