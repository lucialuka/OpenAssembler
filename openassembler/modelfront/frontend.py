"""Repository entrypoints and transport for the OA native model compiler.

Capture and reference evaluation run on the CPU. GPU computation is performed
only by the cubins admitted by modelfront.runtime, on every transport.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib

from openassembler import project
from openassembler.transport import backend as resolve_backend, runtime_files, ssh_request
from openassembler.torchfront.archive import pack, unpack


class ModelError(ValueError):
    pass


def load_config(root: Path) -> dict:
    path = Path(root) / 'openassembler.toml'
    try:
        config = tomllib.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ModelError(f'{path}: {exc}') from None
    model = config.get('model')
    if not isinstance(model, dict):
        raise ModelError('native model compilation requires [model] with entrypoint, factory and '
                         'input_shapes in openassembler.toml. The legacy Inductor application '
                         'frontend is disabled; no vendor GPU fallback is permitted.')
    entry = model.get('entrypoint')
    if (not isinstance(entry, str) or not entry.endswith('.py') or Path(entry).is_absolute()
            or '..' in Path(entry).parts or not (Path(root) / entry).is_file()
            or not (Path(root) / entry).resolve().is_relative_to(Path(root).resolve())):
        raise ModelError('model.entrypoint must name a Python file inside the repository')
    factory = model.get('factory', 'create_model')
    if not isinstance(factory, str) or not factory.isidentifier():
        raise ModelError('model.factory must be a Python function name')
    shapes = model.get('input_shapes')
    if (not isinstance(shapes, list) or not shapes or any(not isinstance(shape, list) or not shape
            or any(type(d) is not int or d <= 0 for d in shape) for shape in shapes)):
        raise ModelError('model.input_shapes must be a nonempty array of positive integer shapes')
    if any(math.prod(shape) > 2**28 for shape in shapes):
        raise ModelError('model input exceeds the bounded native model element limit')
    if model.get('input_dtypes', ['f32'] * len(shapes)) != ['f32'] * len(shapes):
        raise ModelError('native model inputs currently require f32; implicit precision changes are rejected')
    math_mode = model.get('math', 'fp32')
    if math_mode not in ('fp32', 'f16_rne_f32_accum'):
        raise ModelError('model.math must be fp32 or explicit f16_rne_f32_accum')
    mode = model.get('mode', 'inference')
    if mode not in ('inference', 'training'):
        raise ModelError('model.mode must be inference or training')
    if model.get('loss', 'mse') != 'mse' or model.get('optimizer', 'sgd') != 'sgd':
        raise ModelError('native model training currently supports MSE and plain SGD')
    kind, host = resolve_backend(model.get('build_backend', 'auto'), model.get('host'))
    values = dict(entrypoint=entry, factory=factory, input_shapes=shapes, input_dtypes=['f32'] * len(shapes),
                  math=math_mode, mode=mode, seed=model.get('seed', 7), steps=model.get('steps', 3),
                  learning_rate=model.get('learning_rate', 0.01), rtol=model.get('rtol', 1e-4),
                  atol=model.get('atol', 1e-5), build_backend=kind, build_host=host,
                  gpu=model.get('gpu', 'A100-40GB'), budget=model.get('budget', 900))
    if type(model.get('storage_views', True)) is not bool:
        raise ModelError('model.storage_views must be true or false')
    values['storage_views'] = model.get('storage_views', True)
    if type(model.get('reuse_storage', False)) is not bool:
        raise ModelError('model.reuse_storage must be true or false')
    values['reuse_storage'] = model.get('reuse_storage', False)
    execution = model.get('execution', 'direct')
    if execution not in ('direct', 'cuda_graph') or (execution == 'cuda_graph' and mode != 'training'):
        raise ModelError('model.execution must be direct or cuda_graph; cuda_graph requires training mode')
    values['execution'] = execution
    if type(model.get('sass_listings', True)) is not bool:
        raise ModelError('model.sass_listings must be true or false')
    values['sass_listings'] = model.get('sass_listings', True)
    if type(values['seed']) is not int or type(values['steps']) is not int or not 1 <= values['steps'] <= 1000:
        raise ModelError('model.seed must be an integer and steps must be in [1, 1000]')
    for key in ('learning_rate', 'rtol', 'atol'):
        v = values[key]
        if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
            raise ModelError(f'model.{key} must be finite and nonnegative')
    if not isinstance(values['gpu'], str) or not values['gpu']:
        raise ModelError('model.gpu must be a nonempty string')
    if type(values['budget']) not in (int, float) or not 30 <= values['budget'] <= 1800:
        raise ModelError('model.budget must be in [30, 1800] seconds')
    return values


def _request(request: dict, config: dict) -> dict:
    kind, host = resolve_backend(config['build_backend'], config.get('build_host'))
    if kind == 'auto':
        import platform
        kind = 'local' if platform.system() == 'Linux' else 'modal'
    with tempfile.TemporaryDirectory(prefix='oa-model-request-') as d:
        req, res = Path(d) / 'request.json', Path(d) / 'result.json'
        req.write_text(json.dumps(request, allow_nan=False))
        budget = config['budget']
        if kind == 'ssh':
            # Replay re-derives frozen tensor recipe/source contracts. Ship the
            # same checked compiler data on both actions, without running codegen.
            ssh_request(host, req, res, pack(runtime_files(native=True)),
                        'openassembler.modelfront.worker', budget)
        else:
            if kind == 'modal':
                from openassembler.runner import modal_ready
                if why := modal_ready():
                    raise ModelError(why)
                command = ['modal', 'run', str(Path(__file__).with_name('modal_app.py')),
                           '--request', str(req), '--result', str(res), '--budget', str(budget),
                           '--gpu', config['gpu']]
            elif kind == 'local':
                command = [sys.executable, '-m', 'openassembler.modelfront.worker',
                           '--request', str(req), '--result', str(res)]
            else:
                raise ModelError(f'unsupported model transport {kind}')
            env = dict(os.environ)
            env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get('PYTHONPATH', '')
            try:
                done = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, timeout=budget + 300)
            except subprocess.TimeoutExpired:
                raise ModelError(f'native model {request["action"]} exceeded its {budget:g}s budget') from None
            print(done.stdout, end='')
            if done.returncode and not res.is_file():
                raise ModelError(f'native model worker failed (exit {done.returncode})')
        if not res.is_file():
            raise ModelError('native model worker returned no result')
        result = json.loads(res.read_text())
        if 'error' in result:
            raise ModelError(result['error'])
        return result


def compile_model_repository(root: Path, out_dir: Path, *, backend: str | None = None,
                             host: str | None = None, gpu: str | None = None,
                             budget: float | None = None, verify: bool = False,
                             dry_run: bool = False, **unused) -> int:
    root, out_dir = Path(root).resolve(), Path(out_dir).resolve()
    config = load_config(root)
    if root.is_relative_to(out_dir):
        raise ModelError('model source must be outside the output directory')
    if backend and backend != 'auto':
        config['build_backend'], config['build_host'] = resolve_backend(backend, host)
    elif host:
        config['build_backend'], config['build_host'] = resolve_backend('ssh', host)
    if gpu is not None:
        config['gpu'] = gpu
    if budget is not None:
        if type(budget) not in (int, float) or not math.isfinite(budget) or not 30 <= budget <= 1800:
            raise ModelError('native model budget must be in [30,1800] seconds')
        config['budget'] = budget
    print(f'native model: {root / config["entrypoint"]}:{config["factory"]}, '
          f'{config["mode"]}, {config["math"]}, transport={config["build_backend"]}; OA GPU code only')
    if dry_run:
        print(f'dry run: would capture on CPU, emit OA cubins and publish {out_dir}; nothing was run')
        return 0
    from openassembler.torchfront.frontend import project_files, source_digest
    files = project_files(root, out_dir)
    result = _request({'action': 'compile', 'config': config, 'verify': verify,
                       'source_root': str(root),
                       'source_digest': source_digest(files),
                       'source': base64.b64encode(pack(files)).decode()}, config)
    if result.get('report', {}).get('passed') is False:
        raise ModelError('native model verification failed; failed artifacts were not published')
    artifacts = base64.b64decode(result['artifacts'], validate=True)
    # Validate in isolation before publishing. The model artifact tree owns its
    # output directory; unrelated native builds and user files are never retired.
    with tempfile.TemporaryDirectory(prefix='oa-model-publish-') as d:
        staging = Path(d)
        unpack(artifacts, staging)
        if not (staging / 'plan.json').is_file() or not (staging / 'native-ledger.json').is_file():
            raise ModelError('worker did not produce a complete native model bundle')
        from .runtime import validate_bundle
        validate_bundle(staging)
        metadata_path = staging / 'model.json'
        if not metadata_path.is_file():
            raise ModelError('worker model metadata is missing')
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get('schema') != 1 or metadata.get('backend') != 'openassembler'
                or metadata.get('source_root') != str(root) or metadata.get('config') != config
                or metadata.get('source_digest') != source_digest(files)):
            raise ModelError('worker model metadata differs from its source/configuration request')
        # Bind storage bytes and configuration in addition to executable text.
        manifest = {p.relative_to(staging).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in staging.rglob('*') if p.is_file() and not p.is_symlink()}
        (staging / 'model-artifacts.json').write_text(json.dumps({'schema': 1, 'files': manifest}, indent=2) + '\n')
        with project.output_lock(out_dir):
            for path in staging.rglob('*'):
                if path.is_file():
                    project.write_atomic(project.artifact_path(out_dir, path.relative_to(staging).as_posix()),
                                         path.read_bytes())
    print(json.dumps(result.get('report', {}), indent=2))
    return 0 if result.get('report', {}).get('passed', True) else 1


def resolve_model_output(source: Path, explicit: str | Path | None = None) -> Path | None:
    source = Path(source).resolve()
    candidate = Path(explicit).resolve() if explicit else source
    if (candidate / 'model.json').is_file():
        metadata = json.loads((candidate / 'model.json').read_text())
        if explicit and source != candidate and metadata.get('source_root') not in (None, str(source)):
            raise ModelError('native model output belongs to another source repository')
        return candidate
    candidate = Path(explicit).resolve() if explicit else source / 'output'
    if (candidate / 'model.json').is_file():
        metadata = json.loads((candidate / 'model.json').read_text())
        if source.is_dir() and metadata.get('source_root') not in (None, str(source)):
            raise ModelError('native model output belongs to another source repository')
        return candidate
    return None


def deployment_files(out_dir: Path, *, verification: bool = False) -> dict[str, bytes]:
    """Select only the saved plan, weights and owned native executable modules.

    Compiler diagnostics, reference cubins, caches and unrelated files never
    cross the execution transport. CPU source is included only for verification.
    """
    from .runtime import validate_bundle
    out_dir = Path(out_dir).resolve()
    plan, ledger, _ = validate_bundle(out_dir)
    names = {'plan.json', 'native-ledger.json', 'weights.json', ledger['cubin']}
    if (out_dir / 'model.json').is_file():
        names.add('model.json')
    names.update(record['module'] for record in ledger.get('tensor_kernels', {}).values())
    weights = json.loads((out_dir / 'weights.json').read_text())
    if set(weights) != {p['name'] for p in plan['parameters']}:
        raise ModelError('model weight inventory differs from the saved plan')
    names.update(item['path'] for item in weights.values())
    if verification:
        names.add('model-source.zip')
    result = {name: project.artifact_path(out_dir, name).read_bytes() for name in sorted(names)}
    result['model-artifacts.json'] = json.dumps({'schema': 1, 'files': {
        name: hashlib.sha256(data).hexdigest() for name, data in result.items()
    }}, indent=2).encode()
    return result


def export_model(out_dir: Path, destination: Path, *, verification: bool = False) -> Path:
    """Export a portable OA-only deployment ZIP without reference/toolkit data."""
    destination = Path(destination)
    project.write_atomic(destination, pack(deployment_files(out_dir, verification=verification)))
    return destination


def run_model_repository(out_dir: Path, *, backend: str = 'auto', host: str | None = None,
                         gpu: str | None = None, budget: float | None = None,
                         verify: bool = False, dry_run: bool = False) -> int:
    out_dir = Path(out_dir).resolve()
    metadata = json.loads((out_dir / 'model.json').read_text())
    if metadata.get('backend') != 'openassembler' or metadata.get('schema') != 1:
        raise ModelError('model bundle is not an OA native model')
    index_path = out_dir / 'model-artifacts.json'
    if not index_path.is_file():
        raise ModelError('native model artifact inventory is missing; recompile this model')
    index = json.loads(index_path.read_text())
    if index.get('schema') != 1 or not isinstance(index.get('files'), dict):
        raise ModelError('invalid native model artifact inventory')
    for relative, digest in index['files'].items():
        path = project.artifact_path(out_dir, relative)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ModelError(f'native model artifact changed or is missing: {relative}')
    from .runtime import validate_bundle
    validate_bundle(out_dir)
    config = dict(metadata['config'])
    if backend != 'auto' or host:
        config['build_backend'], config['build_host'] = resolve_backend(backend, host)
    if gpu is not None:
        config['gpu'] = gpu
    if budget is not None:
        if type(budget) not in (int, float) or not math.isfinite(budget) or not 30 <= budget <= 1800:
            raise ModelError('native model budget must be in [30,1800] seconds')
        config['budget'] = budget
    print(f'native model replay: {out_dir}; transport={config["build_backend"]}; OA GPU code only')
    if dry_run:
        print('dry run: would load saved OA cubins and launch the native plan; no compilation or GPU work')
        return 0
    files = deployment_files(out_dir, verification=verify)
    result = _request({'action': 'run', 'config': config, 'verify': verify,
                       'artifacts': base64.b64encode(pack(files)).decode()}, config)
    report = result['report']
    project.write_atomic(out_dir / 'model-run.json', json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report.get('passed') else 1
