"""Repository commands, destination parity and failure isolation without GPUs."""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess

import pytest
from tests.native_fixture import save_native_manifest

from openassembler import build, project, remote_build, runner, transport
from openassembler.cli import main
from openassembler.torchfront import frontend
from openassembler.torchfront.archive import pack
from openassembler.workflow import output_dir, resolve_run


def native_manifest(out, root, sources):
    project.prepare_out_dir(out)
    entries = []
    for i, source in enumerate(sources):
        entry = {'source': str(source), 'cubin': f'cubin/k{i}.cubin', 'sass': None,
                 'reference': f'.build/reference/k{i}.cubin', 'source_sha256': project.sha256_file(source),
                 'kernels': [{'name': 'k', 'params': [], 'compiled_by': 'openassembler'}]}
        for key in ('cubin', 'reference'):
            project.write_atomic(out / entry[key], b'compiled')
        entries.append(entry)
    save_native_manifest(out, root, 'sm_80', 'opt', entries)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / 'user repo'
    root.mkdir()
    (root / 'k.ptx').write_text('.visible .entry k() { ret; }')
    return root


def test_default_output_is_repo_relative_but_file_output_is_cwd_relative(repo, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert output_dir(repo) == repo / 'output'
    assert output_dir(repo / 'k.ptx') == tmp_path / 'output'
    observed = []
    def compile_paths(src, out, options, name):
        observed.append(out)
        return build.CompileResult(0)
    monkeypatch.setattr(build, 'compile_paths', compile_paths)
    assert main(['compile', str(repo)]) == 0
    assert observed == [repo / 'output']


def test_run_repo_from_another_working_directory_and_symlink(repo, tmp_path, monkeypatch):
    native_manifest(repo / 'output', repo, [repo / 'k.ptx'])
    alias = tmp_path / 'alias'
    alias.symlink_to(repo, target_is_directory=True)
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    calls = []
    monkeypatch.setattr(runner, 'run_command', lambda options: calls.append(options) or 0)
    monkeypatch.setattr(build, 'compile_paths', lambda *a: pytest.fail('run compiled sources'))
    assert main(['run', str(alias), '--backend', 'user@a100']) == 0
    assert calls[0].out_dir == repo / 'output'
    assert calls[0].only == {(str(repo / 'k.ptx'), 'k')}
    assert calls[0].backend == 'user@a100'


def test_source_selection_never_runs_a_different_repository(repo, tmp_path, monkeypatch):
    other = tmp_path / 'other'
    other.mkdir()
    source = other / 'k.ptx'
    source.write_text('.visible .entry k() { ret; }')
    out = tmp_path / 'shared'
    native_manifest(out, tmp_path, [repo / 'k.ptx', source])
    selected = resolve_run(repo, out)
    assert selected.only == {(str(repo / 'k.ptx'), 'k')}
    selected = resolve_run(source, out)
    assert selected.only == {(str(source), 'k')}
    (tmp_path / 'unrelated').mkdir()
    with pytest.raises(project.ProjectError, match='no compiled build'):
        resolve_run(tmp_path / 'unrelated', out)


def test_legacy_output_and_explicit_output_are_compatible(repo, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / 'output'
    native_manifest(out, repo, [repo / 'k.ptx'])
    assert resolve_run(repo).out_dir == out
    assert resolve_run(out).only is None
    assert resolve_run(None, out).out_dir == out
    native_manifest(repo / 'output', repo, [repo / 'k.ptx'])
    assert resolve_run(repo).out_dir == repo / 'output'
    with pytest.raises(project.ProjectError, match='multiple builds'):
        resolve_run(repo / 'k.ptx')
    assert resolve_run(repo, repo / 'output').out_dir == repo / 'output'


def test_missing_build_is_clear_and_does_not_create_output(repo, capsys):
    assert main(['run', str(repo), '--dry-run']) == 1
    assert '`run` never compiles' in capsys.readouterr().err
    assert not (repo / 'output').exists()


@pytest.mark.parametrize('value,host,expected', [
    ('local', None, ('local', None)), ('modal', None, ('modal', None)),
    ('auto', None, ('auto', None)), ('user@a100', None, ('ssh', 'user@a100')),
    ('ssh', 'gpu-alias', ('ssh', 'gpu-alias')), ('auto', 'user@a100', ('ssh', 'user@a100')),
    ('user@[2001:db8::1]', None, ('ssh', 'user@[2001:db8::1]')),
])
def test_backend_normalization(value, host, expected):
    assert transport.backend(value, host) == expected


@pytest.mark.parametrize('value,host', [
    ('ssh', None), ('-Fconfig', None), ('user@host;id', None), ('x@y@z', None),
    ('user@host:/path', None), ('user@host\n', None), ('ssh://host', None),
    ('local', 'user@host'), ('user@a100', 'user@different'), ('', '-host'),
    ('', None), (True, None), (0, None)])
def test_backend_rejects_ambiguous_or_unsafe_targets(value, host):
    with pytest.raises(transport.TransportError):
        transport.backend(value, host)


@pytest.mark.parametrize('command', ['compile', 'cnr'])
@pytest.mark.parametrize('destination', ['local', 'modal', 'user@a100'])
def test_dry_run_never_starts_a_build_or_transport(repo, monkeypatch, command, destination):
    monkeypatch.setattr(build, 'compile_paths', lambda *a: pytest.fail('dry-run built sources'))
    monkeypatch.setattr(remote_build, 'ssh_request', lambda *a: pytest.fail('dry-run uploaded sources'))
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: pytest.fail('dry-run started subprocess'))
    # project.walk uses git to discover sources, so avoid that optional subprocess.
    monkeypatch.setattr(project, '_git_files', lambda root: None)
    assert main([command, str(repo), '--backend', destination, '--dry-run']) == 0
    assert not (repo / 'output').exists()


@pytest.mark.parametrize('flags', [['--budget', 'nan'], ['--reps', '-1'], ['-n', '0'],
                                   ['--job-timeout', '0'], ['--spec', 'missing.json']])
def test_cnr_validates_before_compiling(repo, monkeypatch, flags):
    monkeypatch.setattr(build, 'compile_paths', lambda *a: pytest.fail('invalid invocation compiled sources'))
    assert main(['cnr', str(repo), *flags]) == 1


@pytest.mark.parametrize('command,flags', [('compile', ['--budget', '60']),
                                         ('compile', ['--build-budget', '60']),
                                         ('cnr', ['--build-budget', '60'])])
def test_local_native_build_never_silently_ignores_a_build_watchdog(repo, command, flags, capsys):
    assert main([command, str(repo), '--backend', 'local', '--dry-run', *flags]) == 1
    assert 'local native compilation has no build watchdog' in capsys.readouterr().err


@pytest.fixture
def app(repo, monkeypatch):
    (repo / 'model.py').write_text('def create_model(): pass')
    (repo / 'openassembler.toml').write_text(
        '[model]\nentrypoint="model.py"\ninput_shapes=[[17,8]]\nbuild_backend="modal"\n')
    calls = []
    from openassembler.modelfront import frontend as native
    from openassembler.modelfront import runtime
    # This fixture isolates transport routing; ownership validation has its own
    # real ELF and mutation tests and the A100 integration campaign.
    monkeypatch.setattr(runtime, 'validate_bundle', lambda *a: None)
    monkeypatch.setattr(native, 'deployment_files', lambda *a, **k: {})
    def request(request, config):
        calls.append((request, config))
        if request['action'] == 'run':
            return {'report': {'passed': True, 'backend': 'openassembler', 'gpu_codegen_during_replay': 0}}
        return {'report': {'passed': True}, 'artifacts': base64.b64encode(pack({
            'plan.json': b'{}', 'native-ledger.json': b'{}', 'model.json': json.dumps({
                'schema':1, 'backend':'openassembler', 'config':config, 'source_root':str(repo),
                'source_digest': request['source_digest']}).encode()
        })).decode()}
    monkeypatch.setattr(native, '_request', request)
    # These transport-only tests deliberately stub worker artifact semantics;
    # real ownership rejection/publication gates have separate regression tests.
    monkeypatch.setattr('openassembler.modelfront.runtime.validate_bundle',lambda *a,**k:({}, {}, b''))
    return repo, calls


@pytest.mark.parametrize('destination', ['local', 'modal', 'user@a100'])
def test_application_cnr_uses_one_destination_for_both_stages(app, destination):
    root, calls = app
    assert main(['cnr', str(root), '--frontend', 'model', '--backend', destination, '--verify',
                 '--budget', '120']) == 0
    assert [c[0]['action'] for c in calls] == ['compile', 'run']
    for request, config in calls:
        assert config['build_backend'] == ('ssh' if '@' in destination else destination)
        assert config.get('build_host') == (destination if '@' in destination else None)
        assert request['verify'] is True
    assert (root / 'output/model.json').is_file()


def test_application_run_infers_bundle_and_allows_new_ssh_destination(app):
    root, calls = app
    assert main(['compile', str(root), '--frontend', 'model', '--backend', 'modal']) == 0
    assert main(['run', str(root), '--backend', 'ssh', '--host', 'user@a100',
                 '--verify-application']) == 0
    assert calls[-1][1]['build_host'] == 'user@a100'
    assert calls[-1][0]['action'] == 'run'
    assert main(['run', str(root / 'output'), '--backend', 'gpu-alias', '--verify-application']) == 0
    assert calls[-1][1]['build_host'] == 'gpu-alias'


def test_application_cnr_allows_separate_build_transport(app):
    root, calls = app
    assert main(['cnr', str(root), '--frontend', 'model', '--build-backend', 'modal', '--backend', 'user@a100']) == 0
    assert calls[0][1]['build_backend'] == 'modal'
    assert calls[1][1]['build_backend'] == 'ssh'


def test_ssh_cleanup_failure_does_not_hide_success(tmp_path, monkeypatch):
    request, result = tmp_path / 'request.json', tmp_path / 'result.json'
    request.write_text('{}')
    calls = []
    def run(command, **kw):
        calls.append(command)
        if command[-1].startswith('rm -rf'):
            raise subprocess.TimeoutExpired(command, 30)
        if command[0] == 'scp' and str(result) == command[-1]:
            result.write_text('{"status":"ok"}')
        return subprocess.CompletedProcess(command, 0, '/tmp/openassembler_request_abc123\n'
                                           if command[-1].startswith('mktemp') else '')
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(transport.shutil, 'which', lambda _: '/bin/tool')
    transport.ssh_request('user@a100', request, result, pack({}), 'openassembler.remote_build', 60)
    assert json.loads(result.read_text())['status'] == 'ok'
    assert all('BatchMode=yes' in call for call in calls)


def test_remote_publication_rebases_sources_and_preserves_other_artifacts(repo, tmp_path, monkeypatch):
    out = repo / 'output'
    other = tmp_path / 'other.ptx'
    other.write_text('.visible .entry old() { ret; }')
    native_manifest(out, tmp_path, [other])
    user_file = out / 'user.txt'
    user_file.write_text('keep')
    def request(host, req, res, runtime, module, budget):
        body = json.loads(req.read_text())
        assert body['options']['opt_level'] == 'opt'
        manifest = {'schema': 1, 'root': '/tmp/remote/source', 'entries': [
            {'source': 'k.ptx', 'cubin': 'cubin/new.cubin', 'sass': 'sass/new.sass',
             'reference': '.build/reference/new.cubin', 'kernels': [{'name': 'k', 'params': []}]}]}
        artifacts = {'manifest.json': json.dumps(manifest).encode(), 'cubin/new.cubin': b'GPU words',
                     'sass/new.sass': b'listing', '.build/reference/new.cubin': b'reference'}
        res.write_text(json.dumps({'rc': 0, 'artifacts': base64.b64encode(pack(artifacts)).decode()}))
    monkeypatch.setattr(remote_build, 'ssh_request', request)
    rc, only = remote_build.compile_native(repo, out, build.Options(), backend='ssh', host='user@a100')
    assert rc == 0
    assert only == {(str(repo / 'k.ptx'), 'k')}
    manifest = project.load_manifest(out)
    assert {project.entry_source_abs(manifest, e) for e in manifest['entries']} == {other, repo / 'k.ptx'}
    assert user_file.read_text() == 'keep'
    assert (out / 'cubin/new.cubin').read_bytes() == b'GPU words'


def test_native_runtime_contains_compiler_data_and_no_research_results():
    files = transport.runtime_files(native=True)
    assert 'env/cuda.sh' in files and 'corpus/sm_80/pairs.jsonl' in files
    assert 'openassembler/backend/tensor.py' in files
    assert not any(p.startswith(('exp/', 'output/', '.git/')) or '__pycache__' in p for p in files)


@pytest.mark.parametrize('failure', [True, False])
def test_remote_failed_rebuild_blocks_old_binary_but_kernel_filter_preserves_others(repo, monkeypatch, failure):
    out = repo / 'output'
    native_manifest(out, repo, [repo / 'k.ptx'])
    original_binary = (out / 'cubin/k0.cubin').read_bytes()
    def request(host, req, res, runtime, module, budget):
        manifest = {'schema': 1, 'root': '/tmp/source', 'entries': []}
        res.write_text(json.dumps({'rc': 1 if failure else 0,
            'failed_sources': ['k.ptx'] if failure else [],
            'artifacts': base64.b64encode(pack({'manifest.json': json.dumps(manifest).encode()})).decode()}))
    monkeypatch.setattr(remote_build, 'ssh_request', request)
    remote_build.compile_native(repo, out, build.Options(kernel='selected'), backend='ssh', host='user@a100')
    entry = project.load_manifest(out)['entries'][0]
    assert bool(entry.get('build_failed')) is failure
    assert (out / 'cubin/k0.cubin').read_bytes() == original_binary


def test_remote_build_refuses_busy_output_before_upload(repo, monkeypatch):
    out = repo / 'output'
    monkeypatch.setattr(remote_build, 'ssh_request', lambda *a: pytest.fail('busy build uploaded sources'))
    with project.output_lock(out):
        with pytest.raises(project.ProjectError, match='another build'):
            remote_build.compile_native(repo, out, build.Options(), backend='ssh', host='user@a100')


def test_sources_and_external_include_directories_are_staged(repo, tmp_path):
    include = tmp_path / 'include'
    include.mkdir()
    (include / 'tile.h').write_text('#define TILE 32')
    (repo / 'node_modules').mkdir()
    (repo / 'node_modules/ignore.cu').write_text('__global__ void ignored() {}')
    root, files, includes = remote_build._sources(repo, repo / 'output', build.Options(include_dirs=[include]))
    assert root == repo and includes == ['include/0']
    assert files['include/0/tile.h'] == b'#define TILE 32'
    assert 'source/k.ptx' in files and not any('node_modules' in p for p in files)
