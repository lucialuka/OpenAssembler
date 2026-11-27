"""Homebrew source-tree installation keeps the CLI relocatable."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
FORMULA = ROOT / "Formula" / "openassembler.rb"


def test_formula_declares_runtime_and_complete_source_layout():
    formula = FORMULA.read_text()
    assert 'class Openassembler < Formula' in formula
    assert 'depends_on "python@3.12"' in formula
    assert 'depends_on "modal"' in formula
    for path in ("openassembler", "tools", "specs", "corpus", "examples", "docs", "env", "third_party"):
        assert f'"{path}"' in formula
    for command in ("openassembler", "openassembler-cuda", "openassembler-fetch-cuda"):
        assert f'bin / "{command}"' in formula
    assert 'refs/heads/main.tar.gz' in formula
    assert 'def post_install' not in formula and '"welcome"' not in formula
    assert subprocess.run(["ruby", "-c", str(FORMULA)], capture_output=True,
                          text=True, check=True).stdout.strip() == "Syntax OK"


def test_relocated_source_tree_runs_without_editable_install(tmp_path):
    # Mirror the formula's layout without copying generated corpus output.
    for directory in ("openassembler", "tools", "specs", "examples", "docs", "env", "third_party"):
        shutil.copytree(ROOT / directory, tmp_path / directory)
    (tmp_path / "corpus/sm_80").mkdir(parents=True)
    for name in ("pairs.jsonl", "machine_model.json"):
        shutil.copy2(ROOT / "corpus/sm_80" / name, tmp_path / "corpus/sm_80" / name)
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    version = subprocess.run([sys.executable, "-m", "openassembler", "--version"],
                             cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    assert version.stdout.strip() == "openassembler 0.2.0"
    help_text = subprocess.run([sys.executable, "-m", "openassembler", "--help"],
                               cwd=tmp_path, env=env, capture_output=True, text=True, check=True).stdout
    assert all(command in help_text for command in ("compile", "run", "cnr", "gemm", "rowwise"))
    status = subprocess.run([sys.executable, "-m", "openassembler", "status"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    assert "Repository regression" in status.stdout

    # Execute the formula's actual launcher from a different working directory,
    # including a conflicting package and inherited PYTHONPATH, as after brew install.
    import re
    launcher = re.search(r'\(bin / "openassembler"\).write <<~EOS\n(.*?)    EOS',
                         FORMULA.read_text(), re.S)[1]
    import textwrap
    launcher = textwrap.dedent(launcher).replace('#{libexec}', str(tmp_path))
    launcher = launcher.replace('#{formula_opt_bin("python@3.12")}/python3.12', sys.executable)
    script = tmp_path / 'openassembler-launcher'
    script.write_text(launcher)
    script.chmod(0o755)
    hostile = tmp_path / 'working-directory'
    hostile.mkdir()
    (hostile / 'openassembler.py').write_text('raise RuntimeError("wrong package")')
    env['PYTHONPATH'] = str(hostile)
    launched = subprocess.run([str(script)], input='help compiler\ninspect ../examples/native_model --json\nexit\n',
                              cwd=hostile, env=env, capture_output=True, text=True, check=True)
    assert '--backend' in launched.stdout
    assert '"frontend": "model"' in launched.stdout
    assert 'OpenAssembler v' not in launched.stdout
    assert not launched.stderr
