"""Integration boundaries between scalar, tensor, application and CLI workflows."""
from __future__ import annotations

import base64
import hashlib
import json

import pytest

from openassembler import project
from openassembler.cli import main
from openassembler.torchfront import frontend as torchfront
from openassembler.torchfront.archive import pack


@pytest.fixture
def app_repo(tmp_path):
    root = tmp_path / 'training'
    root.mkdir()
    (root / 'train.py').write_text('print("training")\n')
    (root / 'openassembler.toml').write_text(
        '[application]\nfrontend="torch"\nentrypoint="train.py"\nbuild_backend="modal"\n')
    return root


def capture(request, config, root):
    binary = b'compiled CUDA kernel'
    artifacts = {'cubin/k.cubin': binary, 'graphs.json': b'[]',
                 'cache.zip': pack({'triton/k.cubin': binary})}
    if not request.get('no_sass'):
        artifacts['sass/k.sass'] = b'EXIT;'
    return {'report': {'status': 'ok', 'graphs': [{'nodes': 1}],
                      'runtime': {'torch': '2.7.1+cu128', 'cuda': '12.8', 'python': '3.11.13',
                                  'capability': [8, 0], 'device': 'A100'},
                      'verification': {'status': 'ok', 'tensors_checked': 2},
                      'kernels': [{'name': 'k', 'artifact': 'k', 'cache_path': 'triton/k.cubin',
                                   'sha256': hashlib.sha256(binary).hexdigest()}]},
            'artifacts': base64.b64encode(pack(artifacts)).decode()}


def test_repeated_app_publication_keeps_native_entries_and_removes_stale_sass(app_repo, tmp_path, monkeypatch):
    out = tmp_path / 'output'
    native = tmp_path / 'native.ptx'
    native.write_text('.visible .entry native() { ret; }')
    project.prepare_out_dir(out)
    project.save_manifest(out, tmp_path, 'sm_80', 'opt', [
        {'source': str(native), 'cubin': 'cubin/native.cubin', 'kernels': []}])
    monkeypatch.setattr(torchfront, '_worker_request', capture)
    first = torchfront.compile_application(app_repo, out)
    assert (out / first.entry['kernels'][0]['sass']).is_file()
    second = torchfront.compile_application(app_repo, out, no_sass=True)
    manifest = project.load_manifest(out)
    assert len(manifest['applications']) == len(manifest['entries']) == 1
    assert project.entry_source_abs(manifest, manifest['entries'][0]) == native
    assert not (out / first.entry['kernels'][0]['sass']).exists()
    assert second.entry['kernels'][0]['sass'] is None
    assert main(['run', str(out), '--application', 'training', '--dry-run']) == 0
    (out / second.entry['cache']).write_bytes(b'tampered')
    assert main(['run', str(out), '--application', 'training', '--dry-run']) == 1


def test_file_inside_application_repo_still_uses_file_compilation(app_repo, monkeypatch):
    from openassembler import build
    called = []
    def compile_file(src, out, options, name):
        called.append(src)
        return build.CompileResult(0)
    monkeypatch.setattr(build, 'compile_paths', compile_file)
    missing = app_repo / 'missing.ptx'
    assert main(['compile', str(missing)]) == 0
    assert called == [missing]


@pytest.mark.parametrize('args', [['--arch', 'sm_90'], ['--kernel', 'x'], ['--fallback', 'ptxas'],
                                  ['--dump-ptx'], ['--opt-level', 'safe'], ['--trials', '0']])
def test_application_rejects_ignored_compile_flags(app_repo, args, capsys):
    assert main(['compile', str(app_repo), '--frontend', 'torch', '--dry-run', *args]) == 1
    assert 'error:' in capsys.readouterr().err


@pytest.mark.parametrize('options', [['--budget', '0'], ['--budget', 'nan'],
                                    ['--job-timeout', '0'], ['--budget', '60', '--job-timeout', '90'],
                                    ['--reps', '1']])
def test_cnr_rejects_invalid_application_run_options_before_capture(app_repo, options, monkeypatch):
    monkeypatch.setattr(torchfront, '_worker_request', lambda *a: pytest.fail('invalid run options launched capture'))
    assert main(['cnr', str(app_repo), '--frontend', 'torch', '--dry-run', *options]) == 1


def test_standalone_compile_accepts_shared_transport_flags(tmp_path, capsys):
    src = tmp_path / 'k.ptx'
    src.write_text('.visible .entry k() { ret; }')
    assert main(['compile', str(src), '--build-backend', 'modal', '--dry-run']) == 0
    assert 'bundle on modal' in capsys.readouterr().out


@pytest.mark.parametrize('name', ['..', 'nested/out', 'x' * 81])
def test_application_dry_run_validates_output_names(app_repo, name, tmp_path, capsys):
    out = tmp_path / 'output'
    assert main(['compile', str(app_repo), '--frontend', 'torch', '--out-dir', str(out), '-o', name, '--dry-run']) == 1
    assert 'error:' in capsys.readouterr().err
    assert not out.exists()


def test_application_auto_backend_uses_modal_on_macos(app_repo, tmp_path, monkeypatch, capsys):
    import importlib.util
    import platform
    monkeypatch.setattr(platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(importlib.util, 'find_spec', lambda _: object())
    assert main(['compile', str(app_repo), '--frontend', 'torch', '--out-dir', str(tmp_path / 'output'),
                 '--build-backend', 'auto', '--dry-run']) == 0
    assert 'build: modal' in capsys.readouterr().out


def test_explicit_verification_cannot_be_silently_disabled(app_repo, tmp_path, capsys):
    config = app_repo / 'openassembler.toml'
    config.write_text(config.read_text() + 'verification="none"\n')
    assert main(['compile', str(app_repo), '--frontend', 'torch', '--verify', '--dry-run',
                 '--out-dir', str(tmp_path / 'output')]) == 1
    assert 'verification' in capsys.readouterr().err


def test_application_replay_saves_report_and_forwards_watchdog(app_repo, tmp_path, monkeypatch):
    out = tmp_path / 'output'
    monkeypatch.setattr(torchfront, '_worker_request', capture)
    torchfront.compile_application(app_repo, out)
    observed = []
    def replay(request, transport, root):
        observed.append(request)
        return {'report': {'status': 'ok', 'new_gpu_codegen': 0, 'optimizer_steps': 2,
                           'runtime': {'device': 'A100'},
                           'verification': {'status': 'ok', 'tensors_checked': 2}}}
    monkeypatch.setattr(torchfront, '_worker_request', replay)
    assert torchfront.run_application(out, name='training', budget=120.5, job_timeout=90, verify=True) == 0
    assert observed[0]['config']['job_timeout'] == 90
    assert observed[0]['budget'] == 120.5
    assert json.loads((out / 'run.json').read_text())['application'] == 'training'
    assert len(list((out / 'runs').glob('*.json'))) == 1
    with pytest.raises(torchfront.ApplicationError, match='job-timeout'):
        torchfront.run_application(out, name='training', budget=60, job_timeout=100, dry_run=True)


def test_missing_requirements_and_output_overlap_fail_before_capture(app_repo, capsys):
    config = app_repo / 'openassembler.toml'
    config.write_text(config.read_text() + 'requirements=["missing.txt"]\n')
    assert main(['compile', str(app_repo), '--frontend', 'torch', '--dry-run']) == 1
    assert 'requirements file is missing' in capsys.readouterr().err
    assert main(['compile', str(app_repo), '--frontend', 'torch', '--out-dir', str(app_repo), '--dry-run']) == 1
    assert 'inside the output folder' in capsys.readouterr().err


def test_explicit_application_selection_never_falls_through_to_kernel_execution(tmp_path, capsys):
    project.save_manifest(tmp_path, tmp_path, 'sm_80', 'opt', [])
    assert main(['run', str(tmp_path), '--application', 'missing', '--dry-run']) == 1
    assert 'no application' in capsys.readouterr().err
    assert main(['run', str(tmp_path), '--application', 'app', '--spec', 'pipeline.json']) == 1
    assert 'cannot be combined' in capsys.readouterr().err


def test_output_writers_exclude_each_other_and_release_on_failure(tmp_path):
    first, second = tmp_path / 'one', tmp_path / 'two'
    with project.output_lock(first):
        with pytest.raises(project.ProjectError, match='another build'):
            with project.output_lock(first):
                pytest.fail('concurrent output writer acquired the lock')
        with project.output_lock(second):
            pass
    with pytest.raises(RuntimeError):
        with project.output_lock(first):
            raise RuntimeError('interrupted build')
    with project.output_lock(first):
        pass


def test_gemm_emit_obeys_the_output_writer_lock(tmp_path, capsys):
    args = ['gemm', '--m', '128', '--n', '128', '--k', '128', '--out-dir', str(tmp_path), '--emit-only']
    with project.output_lock(tmp_path):
        assert main(args) == 1
        assert 'another build' in capsys.readouterr().err
        assert not (tmp_path / '.build/generated/gemm.ptx').exists()
    assert main(args) == 0


def test_application_publication_does_not_clobber_an_active_writer(app_repo, tmp_path, monkeypatch):
    out = tmp_path / 'output'
    monkeypatch.setattr(torchfront, '_worker_request', capture)
    with project.output_lock(out):
        with pytest.raises(torchfront.ApplicationError, match='another build'):
            torchfront.compile_application(app_repo, out)
    assert not (out / 'manifest.json').exists()
    torchfront.compile_application(app_repo, out)


def test_app_publication_refuses_existing_native_output_ownership(app_repo, tmp_path, monkeypatch):
    out = tmp_path / 'output'
    out.mkdir()
    native = tmp_path / 'native.ptx'
    native.write_text('native')
    project.save_manifest(out, tmp_path, 'sm_80', 'opt', [{
        'source': str(native), 'cubin': 'cubin/applications/training/k.cubin', 'kernels': []}])
    before = (out / 'manifest.json').read_bytes()
    monkeypatch.setattr(torchfront, '_worker_request', capture)
    with pytest.raises(torchfront.ApplicationError, match='overlap standalone kernel'):
        torchfront.compile_application(app_repo, out)
    assert (out / 'manifest.json').read_bytes() == before


def test_manifest_rejects_incomplete_runtime_before_run(app_repo, tmp_path, monkeypatch):
    out = tmp_path / 'output'
    monkeypatch.setattr(torchfront, '_worker_request', capture)
    torchfront.compile_application(app_repo, out)
    manifest = json.loads((out / 'manifest.json').read_text())
    manifest['applications'][0]['runtime'].pop('torch')
    (out / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(project.ManifestError, match='runtime metadata'):
        project.load_manifest(out)


def test_application_names_cannot_collide_on_case_insensitive_filesystems(app_repo, tmp_path, monkeypatch):
    out = tmp_path / 'output'
    monkeypatch.setattr(torchfront, '_worker_request', capture)
    torchfront.compile_application(app_repo, out, output_name='training')
    with pytest.raises(torchfront.ApplicationError, match='letter case'):
        torchfront.compile_application(app_repo, out, output_name='Training')


def test_compile_does_not_replace_corrupt_manifest(tmp_path, capsys):
    src = tmp_path / 'kernel.ptx'
    src.write_text('.version 8.7\n.target sm_80\n.address_size 64\n.visible .entry k() { ret; }')
    out = tmp_path / 'output'
    out.mkdir()
    original = b'{invalid manifest'
    (out / 'manifest.json').write_bytes(original)
    assert main(['compile', str(src), '--out-dir', str(out)]) == 1
    assert 'corrupt' in capsys.readouterr().err
    assert (out / 'manifest.json').read_bytes() == original


def test_repl_handles_bad_quoting_and_uses_the_shared_dispatcher(monkeypatch, capsys):
    from openassembler.ui import repl
    lines = iter(['compile "unclosed', 'status', 'quit'])
    monkeypatch.setattr('builtins.input', lambda _: next(lines))
    monkeypatch.setattr(repl, '_try_readline', lambda: None)
    calls = []
    monkeypatch.setattr(repl, 'dispatch', lambda args: calls.append(args.cmd) or 0)
    assert repl.run() == 0
    assert calls == ['status']
    assert 'No closing quotation' in capsys.readouterr().err
