"""The logo belongs to each interactive session; command output stays clean."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from openassembler.cli import build_parser, main
from openassembler.ui import repl
from openassembler.ui.banner import _strip, render


def shell(monkeypatch, lines, *, interactive=True):
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: interactive)
    monkeypatch.setattr(sys.stdout, 'isatty', lambda: interactive)
    monkeypatch.setattr(repl, '_try_readline', lambda parser: lambda: None)
    inputs = iter(lines)

    def read(prompt):
        try:
            value = next(inputs)
        except StopIteration:
            raise EOFError from None
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr('builtins.input', read)
    return main([])


def test_each_interactive_launch_shows_logo_without_state(monkeypatch, tmp_path, capsys):
    # Even an existing stamp from an older version must not hide a new session.
    state = tmp_path / 'openassembler'
    state.mkdir()
    (state / 'welcome-shown').touch()
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path))
    monkeypatch.setenv('NO_COLOR', '1')
    for _ in range(2):
        assert shell(monkeypatch, ['exit']) == 0
        output = capsys.readouterr().out
        assert output.count('OpenAssembler v') == 1
        assert '▄' in output
        assert 'Session closed.' in output
        assert '\033' not in output


def test_terminal_commands_never_show_logo(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))
    monkeypatch.setattr(sys.stdout, 'isatty', lambda: True)
    monkeypatch.setattr(sys.stderr, 'isatty', lambda: True)
    assert main(['inspect', 'examples/native_model', '--json']) == 0
    assert json.loads(capsys.readouterr().out)['frontend'] == 'model'
    assert main(['inspect', 'examples/native_model']) == 0
    output = capsys.readouterr()
    assert '▄' not in output.out + output.err
    assert not (tmp_path / 'state').exists()


def test_help_and_errors_do_not_close_shell(monkeypatch, capsys):
    assert shell(monkeypatch, ['help compile', 'compiel', 'compile --bad-flag',
                               "compile 'unfinished", KeyboardInterrupt(), 'help gpu run', 'exit']) == 0
    output = capsys.readouterr()
    assert 'usage: compile [' in output.out
    assert 'usage: gpu run' in output.out
    assert 'Did you mean compile?' in output.err
    assert 'unrecognized arguments' in output.err
    assert 'No closing quotation' in output.err
    assert 'Input cancelled' in output.out
    assert 'Traceback' not in output.out + output.err


def test_shell_dispatches_alias_and_quoted_paths_and_recovers_from_interrupt(monkeypatch, capsys):
    seen = []

    def dispatch(args):
        seen.append(args)
        return 130 if len(seen) == 1 else 0

    monkeypatch.setattr(repl, 'dispatch', dispatch)
    assert shell(monkeypatch, ['compiler "my repo" --backend modal',
                               'cnr "my repo" \\', '--backend user@a100 --dry-run # plan only', 'exit']) == 0
    assert [(a.cmd, a.src, a.backend) for a in seen] == [
        ('compile', 'my repo', 'modal'), ('cnr', 'my repo', 'user@a100')]
    output = capsys.readouterr().out
    assert 'Interrupted' in output and 'Plan complete' in output
    assert output.count('OpenAssembler v') == 1


def test_shell_cd_and_source_shorthand(monkeypatch, tmp_path, capsys):
    repo = tmp_path / 'my repo'
    repo.mkdir()
    source = repo / 'kernel.ptx'
    source.write_text('.entry kernel() {}')
    monkeypatch.chdir(tmp_path)
    seen = []
    monkeypatch.setattr(repl, 'dispatch', lambda args: seen.append(args) or 0)
    assert shell(monkeypatch, ['cd "my repo"', 'pwd', 'kernel.ptx --backend local', 'exit']) == 0
    assert os.getcwd() == str(repo)
    assert seen[0].cmd == 'compile' and seen[0].src == 'kernel.ptx'
    assert str(repo) in capsys.readouterr().out


def test_readline_history_failure_does_not_prevent_exit(monkeypatch, tmp_path):
    import readline
    monkeypatch.setattr('pathlib.Path.home', lambda: tmp_path)
    monkeypatch.setattr(readline, 'read_history_file', lambda path: None)
    def fail(path):
        raise OSError('history is read-only')
    monkeypatch.setattr(readline, 'write_history_file', fail)
    repl._try_readline(build_parser())()


def test_completion_commands_options_backends_and_paths(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'my repo').mkdir()
    (tmp_path / 'my repo' / 'kernel.ptx').touch()
    parser = build_parser()
    for line, partial, expected in [
        ('comp', 'comp', 'compile '), ('ex', 'ex', 'exit '),
        ('compile --ba', '--ba', '--backend '),
        ('compile --backend mo', 'mo', 'modal '),
        ('gpu r', 'r', 'run '), ('help co', 'co', 'compile '),
        ('compile my', 'my', "'my repo/'"),
    ]:
        assert expected in repl.completion_candidates(line, partial, parser)


@pytest.mark.parametrize('width', [20, 40, 60, 80, 132, 140])
def test_banner_fits_terminal(width, monkeypatch):
    monkeypatch.setenv('NO_COLOR', '1')
    assert all(len(_strip(line)) <= width for line in render(width).splitlines())


def test_piped_session_preserves_json_and_last_exit_status(tmp_path):
    env = dict(os.environ, XDG_STATE_HOME=str(tmp_path / 'state'))
    good = subprocess.run([sys.executable, '-m', 'openassembler'],
                          input='inspect examples/native_model --json\nexit\n',
                          capture_output=True, text=True, env=env)
    assert good.returncode == 0
    assert json.loads(good.stdout)['frontend'] == 'model'
    assert not good.stderr
    bad = subprocess.run([sys.executable, '-m', 'openassembler'], input='compile --invalid\n',
                         capture_output=True, text=True, env=env)
    assert bad.returncode == 2
    assert not bad.stdout
    assert not (tmp_path / 'state').exists()
