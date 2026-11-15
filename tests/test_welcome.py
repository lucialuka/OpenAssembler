"""A welcome is shared across processes, terminals, installation and the REPL."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys

from openassembler.cli import main
from openassembler.ui import session


def test_atomic_welcome_has_one_winner_and_persists(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path))
    with ThreadPoolExecutor(max_workers=16) as pool:
        assert sum(pool.map(lambda _: session.consume_show_once(), range(32))) == 1
    assert not session.consume_show_once()
    child = subprocess.run([sys.executable, '-c', 'from openassembler.ui.session import consume_show_once; print(consume_show_once())'],
                           capture_output=True, text=True, check=True)
    assert child.stdout.strip() == 'False'


def test_unwritable_state_does_not_fail_or_repeat_logo(tmp_path, monkeypatch):
    blocked = tmp_path / 'file'
    blocked.write_text('not a directory')
    monkeypatch.setenv('XDG_STATE_HOME', str(blocked))
    assert not session.consume_show_once()
    assert not session.consume_show_once()


def test_installer_welcome_then_repl_does_not_repeat_logo(tmp_path, monkeypatch, capsys):
    from openassembler.ui import repl
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path))
    assert main(['welcome']) == 0
    first = capsys.readouterr().out
    assert '▄' in first and 'openassembler v' in first
    assert main(['welcome']) == 0
    assert not capsys.readouterr().out
    monkeypatch.setattr(repl.sys.stdout, 'isatty', lambda: True)
    monkeypatch.setattr(repl, '_try_readline', lambda: None)
    monkeypatch.setattr('builtins.input', lambda _: 'exit')
    assert repl.run() == 0
    assert '▄' not in capsys.readouterr().out


def test_json_help_and_version_do_not_consume_welcome(tmp_path):
    env = dict(os.environ, XDG_STATE_HOME=str(tmp_path / 'state'))
    for args in (['--version'], ['--help'], ['inspect', 'examples/native_model', '--json']):
        done = subprocess.run([sys.executable, '-m', 'openassembler', *args], env=env,
                              capture_output=True, text=True, check=True)
        assert '▄' not in done.stdout and '▄' not in done.stderr
    assert not (tmp_path / 'state').exists()


def test_terminal_json_stays_parseable_then_normal_command_shows_logo_once(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path))
    monkeypatch.setattr(sys.stdout, 'isatty', lambda: True)
    monkeypatch.setattr(sys.stderr, 'isatty', lambda: True)
    assert main(['inspect', 'examples/native_model', '--json']) == 0
    assert json.loads(capsys.readouterr().out)['frontend'] == 'model'
    assert not session._stamp_path().exists()
    assert main(['inspect', 'examples/native_model']) == 0
    assert '▄' in capsys.readouterr().err
    assert main(['inspect', 'examples/native_model']) == 0
    assert not capsys.readouterr().err
