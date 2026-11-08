"""Repo-level PyTorch application bundling; configuration is explicit and validated."""
from __future__ import annotations

import hashlib
import json
import base64
import os
from pathlib import Path, PurePosixPath
import shlex
import tomllib
import zipfile
import io
import subprocess
import sys
import tempfile
import time

from .archive import MAX_BYTES, MAX_FILES, pack, unpack


class ApplicationError(Exception):
    """Actionable error in application configuration/capture/launch."""


def config_path(root: Path) -> Path | None:
    p = Path(root) / 'openassembler.toml'
    return p if p.is_file() else None


def load_config(path: Path) -> dict:
    try:
        obj = tomllib.loads(Path(path).read_text())
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ApplicationError(f"{path}: invalid application configuration: {exc}") from None
    app = obj.get('application')
    if not isinstance(app, dict) or app.get('frontend') != 'torch':
        raise ApplicationError(f"{path}: [application] frontend must be 'torch'")
    entry = app.get('entrypoint')
    args = app.get('args', [])
    requirements = app.get('requirements', ['requirements.txt'] if (Path(path).parent / 'requirements.txt').is_file() else [])
    if not isinstance(entry, str) or not entry or Path(entry).is_absolute() or '..' in Path(entry).parts:
        raise ApplicationError(f"{path}: application.entrypoint must be a relative file path")
    if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
        raise ApplicationError(f"{path}: application.args must be an array of strings")
    if (not isinstance(requirements, list) or not all(isinstance(x, str) and x and not Path(x).is_absolute()
            and '..' not in Path(x).parts for x in requirements)):
        raise ApplicationError(f"{path}: application.requirements must contain relative file paths")
    if type(app.get('seed', 7)) is not int:
        raise ApplicationError(f"{path}: application.seed must be an integer")
    for key, default in (('rtol', 1e-4), ('atol', 1e-5),
                         ('state_rtol', 1e-3), ('state_atol', 1e-3)):
        value = app.get(key, default)
        if type(value) not in (float, int) or not 0 <= value < float('inf'):
            raise ApplicationError(f"{path}: application.{key} must be a finite nonnegative number")
    runtime = app.get('torch', '2.7.1')
    if runtime != '2.7.1':
        raise ApplicationError(f"{path}: currently supported application runtime is torch==2.7.1")
    verify = app.get('verification', 'state')
    if verify not in ('state', 'none'):
        raise ApplicationError(f"{path}: application.verification must be 'state' or 'none'")
    backend = app.get('build_backend', 'auto')
    from openassembler.transport import backend as resolve_backend, TransportError
    try:
        backend, host = resolve_backend(backend, app.get('host'))
    except TransportError as exc:
        raise ApplicationError(f'{path}: {exc}') from None
    gpu = app.get('gpu', 'A100-40GB')
    if not isinstance(gpu, str) or not gpu:
        raise ApplicationError(f"{path}: application.gpu must be a nonempty GPU name")
    return {'entrypoint': entry, 'args': args, 'requirements': requirements,
            'seed': app.get('seed', 7), 'torch': runtime,
            'rtol': float(app.get('rtol', 1e-4)), 'atol': float(app.get('atol', 1e-5)),
            'state_rtol': float(app.get('state_rtol', 1e-3)),
            'state_atol': float(app.get('state_atol', 1e-3)),
            'verification': verify, 'build_backend': backend, 'build_host': host, 'gpu': gpu,
            'budget': app.get('budget', 900), 'job_timeout': app.get('job_timeout', 600)}


def project_files(root: Path, output: Path) -> dict[str, bytes]:
    from openassembler import project
    root, output = Path(root).resolve(), Path(output).resolve()
    files, _ = project.walk(root, output)
    files = [p for p in files if not project._inside(p.resolve(), output)]
    entries = {}
    total = 0
    for path in files:
        rel = path.relative_to(root).as_posix()
        if path.stat().st_size + total > MAX_BYTES or len(entries) >= MAX_FILES:
            raise ApplicationError(f'application source exceeds the bundle limit ({MAX_BYTES // 1024 // 1024} MiB, {MAX_FILES} files)')
        data = path.read_bytes()
        total += len(data)
        if total > MAX_BYTES or len(entries) == MAX_FILES:
            raise ApplicationError(f"application source exceeds the bundle limit ({MAX_BYTES // 1024 // 1024} MiB, {MAX_FILES} files)")
        entries[rel] = data
    return entries


def source_blob(root: Path, output: Path, config: dict) -> tuple[bytes, str, list[str]]:
    files = project_files(root, output)
    entry = PurePosixPath(config['entrypoint'])
    if entry.as_posix() not in files:
        raise ApplicationError(f"configured training entrypoint {config['entrypoint']!r} is missing from {root}")
    return pack(files), source_digest(files), sorted(files)


def source_digest(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, data in sorted(files.items()):
        encoded = name.encode()
        digest.update(len(encoded).to_bytes(8, 'big'))
        digest.update(encoded)
        digest.update(len(data).to_bytes(8, 'big'))
        digest.update(data)
    return digest.hexdigest()


def safe_name(root: Path) -> str:
    import re
    name = re.sub(r'[^A-Za-z0-9_.-]+', '-', Path(root).resolve().name).strip('-.') or 'application'
    return name[:80]


def decode_result(text: str) -> dict:
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ApplicationError(f"application worker returned invalid JSON: {exc}") from None
    if not isinstance(result, dict):
        raise ApplicationError('application worker response must be a JSON object')
    if 'error' in result:
        raise ApplicationError(result['error'][-16000:])
    return result


def artifact_files(blob: bytes) -> dict[str, bytes]:
    tmp = io.BytesIO(blob)
    with zipfile.ZipFile(tmp) as z:
        # Reuse traversal, duplicate-name, symlink and expansion-limit validation.
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(prefix='oa_torch_artifacts_') as d:
            root = Path(d)
            unpack(blob, root)
            return {i.filename: (root / i.filename).read_bytes() for i in z.infolist() if not i.is_dir()}


def _worker_request(request: dict, config: dict, repo_root: Path) -> dict:
    request_file = result_file = None
    try:
        with tempfile.TemporaryDirectory(prefix='oa-torch-request-') as directory:
            request_file, result_file = Path(directory) / 'request.json', Path(directory) / 'result.json'
            request_file.write_text(json.dumps(request))
            package_root = str(Path(__file__).resolve().parents[2])
            working_dir = Path(repo_root) if Path(repo_root).is_dir() else Path(package_root)
            env = dict(os.environ)
            env['PYTHONPATH'] = package_root + os.pathsep + env.get('PYTHONPATH', '')
            if config['build_backend'] == 'local':
                command = [sys.executable, '-m', 'openassembler.torchfront.worker',
                           '--request', str(request_file), '--result', str(result_file)]
                try:
                    completed = subprocess.run(command, cwd=working_dir, text=True,
                                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                               timeout=config['budget'], env=env)
                except subprocess.TimeoutExpired:
                    raise ApplicationError(f"local PyTorch worker exceeded {config['budget']} seconds") from None
                print(completed.stdout, end='')
                if completed.returncode and not result_file.exists():
                    raise ApplicationError(f"local PyTorch worker failed (exit {completed.returncode})")
            elif config['build_backend'] == 'modal':
                from openassembler import runner
                if (why := runner.modal_ready()):
                    raise ApplicationError(why)
                app_file = Path(__file__).with_name('modal_app.py')
                command = ['modal', 'run', str(app_file), '--request', str(request_file),
                           '--result', str(result_file), '--gpu', config['gpu'], '--budget', str(config['budget'])]
                try:
                    completed = subprocess.run(command, cwd=working_dir, text=True,
                                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                               timeout=config['budget'] + 300)
                except subprocess.TimeoutExpired:
                    raise ApplicationError("Modal application capture timed out; inspect `modal app list`") from None
                print(completed.stdout, end='')
                if completed.returncode and not result_file.exists():
                    raise ApplicationError(f"Modal PyTorch worker failed (exit {completed.returncode})")
            elif config['build_backend'] == 'ssh':
                from openassembler.transport import runtime_files, ssh_request, TransportError
                try:
                    ssh_request(config.get('build_host'), request_file, result_file,
                                pack(runtime_files()), 'openassembler.torchfront.worker', config['budget'])
                except TransportError as exc:
                    raise ApplicationError(str(exc)) from None
            else:
                raise ApplicationError("PyTorch application backend must be local, modal, or ssh")
            if not result_file.is_file():
                raise ApplicationError("PyTorch worker returned no capture result")
            return decode_result(result_file.read_text())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        if isinstance(exc, ApplicationError):
            raise
        raise ApplicationError(f"PyTorch worker could not run: {exc}") from None


def compile_application(root: Path, out_dir: Path, *, output_name: str | None = None,
                        verify: bool = True, dry_run: bool = False,
                        build_backend: str | None = None, gpu: str | None = None,
                        budget: float | None = None, no_sass: bool = False,
                        build_host: str | None = None, job_timeout: float | None = None):
    """Capture the configured CUDA training run and publish cubins, SASS and runtime."""
    from openassembler import project
    root, out_dir = Path(root).resolve(), Path(out_dir).resolve()
    if root.is_relative_to(out_dir):
        raise ApplicationError('application source is inside the output folder; choose another --out-dir')
    try:
        name, note = project.single_output_name(Path('application.py'), output_name or safe_name(root))
    except project.ProjectError as exc:
        raise ApplicationError(str(exc)) from None
    if note:
        print(note)
    if (not name or name in ('.', '..') or len(name) > 80
            or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in name)):
        raise ApplicationError('application output name must contain 1..80 letters, numbers, dots, dashes or underscores')
    if out_dir.exists() and not out_dir.is_dir():
        raise ApplicationError(f'output folder {out_dir} exists and is a file')
    config_file = config_path(root)
    if config_file is None:
        raise ApplicationError(f"{root} has no openassembler.toml application configuration")
    config = load_config(config_file)
    if verify and config['verification'] == 'none':
        raise ApplicationError("--verify requires application.verification='state'; it is disabled in openassembler.toml")
    entrypoint = (root / config['entrypoint']).resolve()
    if not entrypoint.is_relative_to(root) or not entrypoint.is_file():
        raise ApplicationError(f"configured application entrypoint is missing or escapes the repository: {config['entrypoint']}")
    if config['verification'] not in ('state', 'none'):
        raise ApplicationError("application.verification must be 'state' or 'none'")
    for key in ('budget', 'job_timeout'):
        if type(config[key]) not in (int, float) or not 0 < config[key] < float('inf'):
            raise ApplicationError(f"application.{key} must be a finite positive number")
    try:
        project.load_manifest(out_dir)
    except project.ProjectError as exc:
        raise ApplicationError(str(exc)) from None
    from openassembler.transport import backend as resolve_backend, TransportError
    selected = build_backend or ('ssh' if build_host else config['build_backend'])
    host = build_host or (config.get('build_host') if selected == 'ssh' else None)
    try:
        config['build_backend'], config['build_host'] = resolve_backend(selected, host)
    except TransportError as exc:
        raise ApplicationError(str(exc)) from None
    if gpu is not None:
        config['gpu'] = gpu
    if not isinstance(config['gpu'], str) or not config['gpu'].strip():
        raise ApplicationError("application GPU must be a nonempty GPU type")
    if budget is not None:
        if type(budget) not in (int, float) or not 30 <= budget <= 1800:
            raise ApplicationError("--budget must be from 30 through 1800 seconds")
        config['budget'] = budget
        config['job_timeout'] = min(config['job_timeout'], budget)
    if job_timeout is not None:
        config['job_timeout'] = job_timeout
    if type(config['job_timeout']) not in (int, float) or not 0 < config['job_timeout'] <= config['budget']:
        raise ApplicationError('application --job-timeout must be finite, positive, and no greater than --budget')
    if not 30 <= config['budget'] <= 1800:
        raise ApplicationError('application budget must be 30..1800 seconds')
    if config['build_backend'] == 'auto':
        import importlib.util
        import platform
        config['build_backend'] = 'local' if platform.system() == 'Linux' and importlib.util.find_spec('torch') else 'modal'
    source, digest, file_names = source_blob(root, out_dir, config)
    for rel in config['requirements']:
        if rel not in file_names:
            raise ApplicationError(f'configured requirements file is missing from the source bundle: {rel}')
    if dry_run:
        print(f"PyTorch application {name}: {len(file_names)} files, {len(source) / 1024:.1f} KiB source")
        print('entrypoint: ' + shlex.join([config['entrypoint'], *config['args']]))
        transport = f"SSH {config['build_host']}" if config['build_backend'] == 'ssh' else f"GPU {config['gpu']}"
        print(f"build: {config['build_backend']}, {transport}, torch=={config['torch']}, budget {config['budget']:g}s, "
              f"eager/compiled state verification {'on' if verify and config['verification'] == 'state' else 'off'}")
        print("capture executes the configured training workload to produce compiler specializations")
        return ApplicationBuildResult(0, out_dir, name, None, config)

    request = {'mode': 'compile', 'config': config, 'source': base64.b64encode(source).decode('ascii'),
               'digest': digest, 'budget': config['budget'], 'no_sass': no_sass,
               'verify': verify and config['verification'] == 'state'}
    started = time.monotonic()
    raw = _worker_request(request, config, root)
    report = raw.get('report')
    if not isinstance(report, dict) or report.get('status') != 'ok' or not report.get('kernels'):
        raise ApplicationError('PyTorch capture completed without a usable GPU-kernel report')
    if not report.get('graphs'):
        raise ApplicationError('PyTorch capture generated no compiled model graphs; execution may have skipped model training')
    if verify and config['verification'] == 'state' and report.get('verification', {}).get('status') != 'ok':
        raise ApplicationError('eager/compiled verification did not pass; generated artifacts were not published')
    raw_artifacts = raw.get('artifacts')
    if not isinstance(raw_artifacts, str):
        raise ApplicationError('PyTorch worker did not return its compiled artifact bundle')
    try:
        artifacts = artifact_files(base64.b64decode(raw_artifacts, validate=True))
    except (ValueError, OSError) as exc:
        raise ApplicationError(f"invalid compiled application artifacts: {exc}") from None
    if 'cache.zip' not in artifacts or 'graphs.json' not in artifacts:
        raise ApplicationError('PyTorch bundle is missing its runtime cache or graph record')
    cache_digest = hashlib.sha256(artifacts['cache.zip']).hexdigest()
    kernel_records = []
    app_rel = f'applications/{name}'
    published = {}
    for path, data in artifacts.items():
        if path == 'cache.zip':
            continue
        if path.startswith('cubin/'):
            destination = f'cubin/{app_rel}/{path.removeprefix("cubin/")}'
        elif path.startswith('sass/'):
            destination = f'sass/{app_rel}/{path.removeprefix("sass/")}'
        elif path.startswith('ptx/'):
            destination = f'.build/ptx/{app_rel}/{path.removeprefix("ptx/")}'
        else:
            destination = f'.build/{app_rel}/{path}'
        published[destination] = data
    for k in report['kernels']:
        rel = f"cubin/{k['artifact']}.cubin"
        if rel not in artifacts or hashlib.sha256(artifacts[rel]).hexdigest() != k['sha256']:
            raise ApplicationError(f"compiled-kernel checksum verification failed for {k.get('name', '?')}")
        kernel_records.append({**k, 'backend': 'torch-inductor-triton',
            'cubin': f'cubin/{app_rel}/{k["artifact"]}.cubin',
            'sass': f'sass/{app_rel}/{k["artifact"]}.sass' if f'sass/{k["artifact"]}.sass' in artifacts else None,
            'ptx': f'.build/ptx/{app_rel}/{k["artifact"]}.ptx' if f'ptx/{k["artifact"]}.ptx' in artifacts else None})
    published[f'.build/{app_rel}/source.zip'] = source
    published[f'.build/{app_rel}/cache.zip'] = artifacts['cache.zip']
    report.update(source_sha256=digest, source_files=file_names, runtime_seconds=time.monotonic() - started,
                  captured_by='torch-inductor-triton', verification=report.get('verification', {'status':'not-run'}))
    published[f'.build/{app_rel}/report.json'] = json.dumps(report, indent=1) + '\n'
    published[f'.build/{app_rel}/graphs.json'] = artifacts['graphs.json']
    entry = {'name': name, 'source': str(root), 'source_sha256': digest,
             'source_bundle': f'.build/{app_rel}/source.zip', 'cache': f'.build/{app_rel}/cache.zip',
             'source_bundle_sha256': hashlib.sha256(source).hexdigest(), 'cache_sha256': cache_digest,
             'report': f'.build/{app_rel}/report.json', 'graphs': f'.build/{app_rel}/graphs.json',
             'runtime': report['runtime'],
             'config': f'{app_rel}/config.json',
             'entrypoint': config['entrypoint'], 'args': config['args'], 'seed': config['seed'],
             'requirements': config['requirements'],
             'rtol': config['rtol'], 'atol': config['atol'], 'gpu': report['runtime']['device'],
             'state_rtol': config['state_rtol'], 'state_atol': config['state_atol'],
             'build_gpu': config['gpu'],
             'build_host': config.get('build_host'),
             'budget': config['budget'], 'job_timeout': config['job_timeout'],
             'build_backend': config['build_backend'], 'verification': report['verification'],
             'compiled_by': 'torch-inductor-triton', 'kernels': kernel_records}
    published[f'{app_rel}/config.json'] = json.dumps(entry, indent=1) + '\n'
    _publish_application(out_dir, root, name, published, entry)
    print(f"PyTorch application {name}: captured {len(report['graphs'])} graphs and {len(kernel_records)} generated CUDA kernels")
    print(f"compiled by torch-inductor-triton for {report['runtime']['device']}; runtime bundle {app_rel}")
    if report.get('verification', {}).get('status') == 'ok':
        print(f"eager/compiled verification: {report['verification']['tensors_checked']} tensors")
    return ApplicationBuildResult(0, out_dir, name, entry, config)


def _publish_application(out_dir, root, name, published, entry):
    from openassembler import project
    try:
        with project.output_lock(out_dir):
            old = project.load_manifest(out_dir)
            entries = ([{**entry, 'source': str(project.entry_source_abs(old, entry))}
                        for entry in old['entries']] if old else [])
            old_apps = old.get('applications', []) if old else []
            if any(a['name'].casefold() == name.casefold() and a['name'] != name for a in old_apps):
                raise ApplicationError('application output name differs only by letter case from an existing build')
            owned = {path.casefold() for e in entries for path in project.entry_outputs(e)}
            if any(path.casefold() in owned for path in published):
                raise ApplicationError('application outputs overlap standalone kernel files; choose another -o or --out-dir')
            stale = [a for a in old_apps if a.get('name') == name]
            apps = [a for a in old_apps if a.get('name') != name]
            stale_outputs = []
            for previous in stale:
                paths = [previous.get(key) for key in ('source_bundle', 'cache', 'report', 'graphs', 'config')]
                paths.extend(value for kernel in previous.get('kernels', [])
                             for value in (kernel.get('cubin'), kernel.get('sass'), kernel.get('ptx')))
                stale_outputs.extend(value for value in paths if value and value not in published)
            with project.defer_sigint():
                project.write_group(out_dir, published)
                manifest_root = Path(old['root']) if old and entries else root
                arch = old.get('arch', 'sm_80') if old else 'sm_80'
                opt_level = old.get('opt_level', 'application') if old and entries else 'application'
                project.save_manifest(out_dir, manifest_root, arch, opt_level, entries, applications=apps + [entry])
            project.remove_outputs(out_dir, stale_outputs)
    except project.ProjectError as exc:
        raise ApplicationError(str(exc)) from None


class ApplicationBuildResult:
    def __init__(self, rc, out_dir, name, entry, config=None):
        self.rc, self.out_dir, self.name, self.entry = rc, out_dir, name, entry
        self.config = config


def run_application(out_dir: Path, *, name: str | None = None, backend: str = 'auto', gpu=None,
                    budget=None, dry_run=False, extra_args=(), verify=False, job_timeout=None, host=None):
    """Replay a compiled application using its bundled Inductor/Triton cache."""
    from openassembler import project
    out_dir = Path(out_dir).resolve()
    try:
        manifest = project.load_manifest(out_dir)
    except project.ProjectError as exc:
        raise ApplicationError(str(exc)) from None
    apps = manifest.get('applications', []) if manifest else []
    if apps and manifest.get('entries') and name is None:
        return None
    if name:
        apps = [a for a in apps if a.get('name') == name]
    if not apps:
        if name is not None:
            raise ApplicationError(f"no application {name!r} in {out_dir}/manifest.json")
        return None
    if len(apps) != 1:
        raise ApplicationError('select one application with --application; found ' + ', '.join(a['name'] for a in apps))
    app = apps[0]
    if extra_args and list(extra_args) != app['args']:
        raise ApplicationError('application runtime args are fixed by the bundle; edit openassembler.toml and recompile')
    if backend == 'auto':
        backend = 'ssh' if host else app.get('build_backend', 'modal')
        if backend == 'auto':
            backend = 'modal'
    from openassembler.transport import backend as resolve_backend, TransportError
    try:
        backend, chosen_host = resolve_backend(backend, host or (app.get('build_host') if backend == 'ssh' else None))
    except TransportError as exc:
        raise ApplicationError(str(exc)) from None
    for key in ('source_bundle', 'cache', 'report', 'graphs', 'config'):
        if not project.artifact_path(out_dir, app[key]).is_file():
            raise ApplicationError(f"application runtime artifact is missing: {app[key]}; recompile")
    for kernel in app['kernels']:
        artifact = project.artifact_path(out_dir, kernel['cubin'])
        if not artifact.is_file() or project.sha256_file(artifact) != kernel['sha256']:
            raise ApplicationError(f"kernel {kernel['name']} cubin changed or is missing; recompile")
    source_blob = project.artifact_path(out_dir, app['source_bundle']).read_bytes()
    cache_blob = project.artifact_path(out_dir, app['cache']).read_bytes()
    if hashlib.sha256(source_blob).hexdigest() != app.get('source_bundle_sha256'):
        raise ApplicationError('application source bundle changed; recompile before running')
    if hashlib.sha256(cache_blob).hexdigest() != app.get('cache_sha256'):
        raise ApplicationError('application runtime cache changed; recompile before running')
    chosen_gpu = gpu or app.get('build_gpu', 'A100-40GB')
    duration = budget if budget is not None else app.get('budget', 900)
    if type(duration) not in (int, float) or not 30 <= duration <= 1800:
        raise ApplicationError('application --budget must be between 30 and 1800 seconds')
    watchdog = job_timeout if job_timeout is not None else min(app.get('job_timeout', duration), duration)
    if type(watchdog) not in (int, float) or not 0 < watchdog <= duration:
        raise ApplicationError('application --job-timeout must be finite, positive, and no greater than --budget')
    if dry_run:
        print(f"application {app['name']}: {len(app['kernels'])} kernels, runtime torch=={app['runtime']['torch']}, "
              f"{('SSH ' + chosen_host) if backend == 'ssh' else backend + ' ' + chosen_gpu}, budget {duration:g}s")
        print(f"would replay {app['entrypoint']} with {len(app['args'])} configured arguments; Triton code generation is blocked")
        return 0
    cache_files = artifact_files(cache_blob)
    for kernel in app['kernels']:
        cache_files[kernel['cache_path']] = project.artifact_path(out_dir, kernel['cubin']).read_bytes()
    request = {'mode': 'run', 'config': {'entrypoint': app['entrypoint'], 'args': app['args'],
                'seed': app['seed'], 'torch': app['runtime']['torch'], 'rtol': app['rtol'], 'atol': app['atol'],
                'state_rtol': app.get('state_rtol', 1e-3), 'state_atol': app.get('state_atol', 1e-3),
                'job_timeout': watchdog},
                'requirements': app.get('requirements', []),
                'source': base64.b64encode(source_blob).decode(),
                'cache': base64.b64encode(pack(cache_files)).decode(), 'digest': app['source_sha256'],
                'runtime': app['runtime'], 'budget': duration, 'verify': verify}
    # Worker config also controls transport; adding it here keeps compile/run config explicit.
    transport = {'build_backend': backend, 'gpu': chosen_gpu, 'budget': duration, 'job_timeout': duration}
    transport['build_host'] = chosen_host
    result = _worker_request(request, transport, Path(app['source']).resolve().parent)
    report = result.get('report', {})
    if report.get('status') != 'ok' or report.get('new_gpu_codegen') != 0:
        raise ApplicationError('application runtime did not confirm an exact cache-only replay')
    from datetime import datetime, timezone
    from openassembler import __version__
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    saved = {'openassembler': __version__, 'timestamp': stamp, 'application': app['name'],
             'backend': chosen_host if backend == 'ssh' else backend, 'source_sha256': app['source_sha256'], **report}
    project.write_atomic(out_dir / 'runs' / f'{stamp}.json', json.dumps(saved, indent=1) + '\n')
    project.write_atomic(out_dir / 'run.json', json.dumps(saved, indent=1) + '\n')
    print(f"application {app['name']} replayed on {report['runtime']['device']}; {report['optimizer_steps']} optimizer steps")
    print(f"new GPU code generation: {report['new_gpu_codegen']}")
    if report.get('verification', {}).get('status') == 'ok':
        print(f"eager/compiled verification: {report['verification']['tensors_checked']} tensors")
    return 0

