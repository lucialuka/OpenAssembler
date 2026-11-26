"""Native compiler transport. The worker calls the unchanged build pipeline.

Only manifest-owned artifacts are published locally, under the existing writer
lock. Source paths are rebased to the user's local tree before publication.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import subprocess
import tempfile
import os

from openassembler import project
from openassembler.build import Options
from openassembler.torchfront.archive import MAX_BYTES, MAX_FILES, pack, unpack
from openassembler.transport import TransportError, runtime_files, ssh_request


def _sources(source: Path, output: Path, options: Options) -> tuple[Path, dict[str, bytes], list[str]]:
    root = source if source.is_dir() else source.parent
    files, _ = project.walk(root, output)
    # A git-ignored file explicitly named on the command line still participates.
    if source.is_file() and source not in files:
        files.append(source)
    payload = {}
    total = 0

    def add(name, path):
        nonlocal total
        if name in payload:
            if payload[name] != path.read_bytes():
                raise project.ProjectError(f'source dependency changed while bundling: {path}')
            return
        if len(payload) >= MAX_FILES or total + path.stat().st_size > MAX_BYTES:
            raise project.ProjectError('remote source bundle exceeds 256 MiB / 20000 files; '
                                       'exclude generated data or build locally')
        data = path.read_bytes()
        total += len(data)
        if total > MAX_BYTES:
            raise project.ProjectError('remote source bundle exceeds 256 MiB')
        payload[name] = data

    library_paths={Path(p).resolve()for p in options.device_libraries}
    for path in files:
        if path.resolve()not in library_paths:add('source/' + path.relative_to(root).as_posix(), path)
    from .compilation_database import discover as discover_commands
    database, commands = discover_commands(source, options.compile_commands)
    commands={path:command for path,command in commands.items()if path not in library_paths}
    if commands:
        def map_path(path):
            path = Path(path).resolve()
            if path.is_relative_to(root):
                return Path('source') / path.relative_to(root)
            return Path('build-dependencies') / Path(*path.parts[1:])
        def include_tree(directory):
            if not directory.is_dir():
                raise project.ProjectError(f'compilation database include directory missing: {directory}')
            destination = map_path(directory)
            for path in sorted(directory.rglob('*')):
                relative = path.relative_to(directory)
                if any(part in project.SKIP_DIRS-{'build'} or part == '.git' for part in relative.parts) or path.resolve().is_relative_to(output.resolve()):
                    continue
                if path.is_symlink():
                    raise project.ProjectError(f'compilation database dependency symlink must be materialized: {path}')
                if path.is_file():
                    add((destination/relative).as_posix(),path)
        quoted_seen=set()
        def include_quoted(path):
            import re
            if path in quoted_seen:return
            quoted_seen.add(path)
            if len(quoted_seen)>20000:raise project.ProjectError('quoted-header dependency closure exceeds 20000 files')
            for name in re.findall(r'^\s*#\s*include\s*"([^"\n]+)"',path.read_text(errors='replace'),flags=re.M):
                dependency=(path.parent/name).resolve()
                if dependency.is_file():
                    add(map_path(dependency).as_posix(),dependency)
                    include_quoted(dependency)
        records, origins = [], {}
        for index, command in enumerate(commands.values()):
            spelled=Path(os.path.abspath(command.directory/command.source_argument))
            if spelled!=command.source:
                raise project.ProjectError('remote compilation database source aliases must be materialized before relocation')
            source_relative = map_path(command.source)
            add(source_relative.as_posix(),command.source)
            # NVCC searches a quoted include beside its translation unit even
            # without -I. This also preserves ignored generated sibling inputs.
            include_quoted(command.source)
            for response in command.response_files:
                add(map_path(response).as_posix(),Path(response))
            working = map_path(command.directory)
            payload[(working/'.oa-directory').as_posix()] = b''
            flags=[];cursor=0
            while cursor<len(command.flags):
                token=command.flags[cursor]
                if token in ('-I','-isystem','-include','-iquote'):
                    path=command.resolve_path(command.flags[cursor+1]);target=map_path(path)
                    if token=='-include':
                        add(target.as_posix(),path)
                        include_quoted(path)
                    else:include_tree(path)
                    flags.extend((token,os.path.relpath(target,working)));cursor+=2
                elif token in ('-ccbin','--compiler-bindir') or token.startswith(('-ccbin=','--compiler-bindir=')):
                    raise project.ProjectError('remote compilation database cannot relocate a host compiler override; use the pinned worker toolchain')
                else:flags.append(token);cursor+=1
            records.append({'directory':os.path.relpath(working,Path('source')),
                            'file':os.path.relpath(source_relative,working),
                            'arguments':['nvcc',*flags,'-c',os.path.relpath(source_relative,working)]})
            origins[command.source.relative_to(root).as_posix()]=command.record()
        payload['source/compile_commands.json']=json.dumps(records).encode()
        payload['source/.oa-build-origins.json']=json.dumps(origins).encode()
    includes = []
    for index, directory in enumerate(options.include_dirs):
        directory = directory.resolve()
        if not directory.is_dir():
            raise project.ProjectError(f'include directory does not exist: {directory}')
        if directory.is_relative_to(root):
            includes.append('source/' + directory.relative_to(root).as_posix())
        else:
            prefix = f'include/{index}'
            includes.append(prefix)
            paths, _ = project.walk(directory, output)
            for path in paths:
                add(f'{prefix}/{path.relative_to(directory).as_posix()}', path)
    if options.device_libraries:
        from dataclasses import replace
        from .compilation_database import load
        records=[]
        for index, library in enumerate(options.device_libraries):
            library=Path(library).resolve()
            if not library.is_file():raise project.ProjectError('device library missing: '+str(library))
            selected=load(options.compile_commands).get(library)if options.compile_commands else None
            database=options.compile_commands if selected else None
            library_options=replace(options,device_libraries=[],device_library_options=[],compile_commands=database)
            _library_root,library_files,library_includes=_sources(library,output,library_options)
            _,library_commands=discover_commands(library,database)
            if not library_commands:
                library_files.pop('source/compile_commands.json',None)
            prefix=f'device-libraries/{index}/'
            for relative,data in library_files.items():
                if len(payload)>=MAX_FILES or total+len(data)>MAX_BYTES:
                    raise project.ProjectError('device library source bundle exceeds 256 MiB / 20000 files')
                payload[prefix+relative]=data;total+=len(data)
            records.append({'path':prefix+'source/'+library.name,
                'source_origin':str(library),'source_sha256':project.sha256_file(library),
                'include_dirs':[prefix+name for name in library_includes],
                'compile_commands':prefix+'source/compile_commands.json'if library_commands else None})
        payload['.oa-device-libraries.json']=json.dumps(records).encode()
    return root, payload, includes


def compile_native(source: Path, output: Path, options: Options, name: str | None = None,
                   *, backend='local', host=None, budget=900, dry_run=False):
    """Return (exit status, set of successfully built local source/kernel pairs)."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.exists():
        raise project.ProjectError(f'{source} does not exist')
    if source.is_relative_to(output):
        raise project.ProjectError('source is inside the output folder; choose another --out-dir')
    if options.arch != 'sm_80':
        raise project.ProjectError(f'no SASS encoder data for {options.arch!r}')
    if type(options.trials) is not int or options.trials < 1:
        raise project.ProjectError('--trials must be a positive integer')
    if type(budget) not in (int, float) or not 30 <= budget <= 1800:
        raise project.ProjectError('compile --budget must be finite and between 30 and 1800 seconds')
    if source.is_dir() and name is not None:
        raise project.ProjectError('-o names a single file; use --out-dir for a repository')
    if source.is_file():
        if project.language_of(source) is None:
            raise project.ProjectError(f'unsupported source file: {source}')
        project.single_output_name(source, name)
    project.load_manifest(output)
    if dry_run and backend == 'local':
        from .compilation_database import discover as discover_commands
        _, commands=discover_commands(source,options.compile_commands)
        files = project.walk(source, output)[0] if source.is_dir() else [source]
        library_paths={Path(p).resolve()for p in options.device_libraries}
        count = len(({path for path in files if project.classify(path,path.name).language} | set(commands))-library_paths)
        if not count:
            raise project.ProjectError('nothing to compile: add supported GPU kernels or configure a PyTorch application')
        print(f'native build: {count} source(s) on local; would write {output}/cubin/ and {output}/sass/')
        print('nothing was compiled or uploaded')
        return 0, set()
    root, files, includes = _sources(source, output, options)
    from .compilation_database import discover as discover_commands
    _, commands = discover_commands(source,options.compile_commands)
    candidates = [rel.removeprefix('source/') for rel in files if rel.startswith('source/')
                  and (source.is_file() and root / rel.removeprefix('source/') == source
                       or source.is_dir() and (root / rel.removeprefix('source/') in commands
                           or project.classify(root / rel.removeprefix('source/'), rel.removeprefix('source/')).language))]
    library_paths={Path(p).resolve()for p in options.device_libraries}
    candidates=[rel for rel in candidates if (root/rel).resolve()not in library_paths]
    if not candidates:
        raise project.ProjectError('nothing to compile: add supported GPU kernels, or configure '
                                   '[application] frontend="torch" in openassembler.toml for a PyTorch repository')
    payload = pack(files)
    destination = host if backend == 'ssh' else backend
    print(f'native build: {len(candidates)} source(s), {len(payload) / 1024:.1f} KiB bundle on {destination}', flush=True)
    if dry_run:
        print(f'would write cubin/ and sass/ under {output}; nothing was compiled or uploaded')
        return 0, set()
    with project.output_lock(output):
        return _compile_locked(source, output, options, name, backend, host, budget,
                    root, payload, includes, candidates,
                    json.loads(files.get('.oa-device-libraries.json',b'[]')))


def _compile_locked(source, output, options, name, backend, host, budget,
                    root, payload, includes, candidates, libraries=None):
    from dataclasses import asdict
    libraries=libraries or []
    request = {'source': base64.b64encode(payload).decode(),
               'path': 'source' if source.is_dir() else 'source/' + source.name,
               'options': {**asdict(options), 'include_dirs': includes,
                           'compile_commands':'source/compile_commands.json' if options.compile_commands else None,
                           'compile_command':None,
                           'device_libraries':[row['path']for row in libraries],
                           'device_library_options':[{k:v for k,v in row.items()if k!='path'}for row in libraries]}, 'name': name}
    with tempfile.TemporaryDirectory(prefix='oa-native-request-') as d:
        request_file, result_file = Path(d) / 'request.json', Path(d) / 'result.json'
        request_file.write_text(json.dumps(request))
        if backend == 'ssh':
            ssh_request(host, request_file, result_file, pack(runtime_files(native=True)),
                        'openassembler.remote_build', budget)
        elif backend == 'modal':
            from openassembler.runner import modal_ready
            if why := modal_ready():
                raise TransportError(why)
            command = ['modal', 'run', str(Path(__file__).with_name('build_modal.py')),
                       '--request', str(request_file), '--result', str(result_file), '--budget', str(budget)]
            try:
                subprocess.run(command, check=True, timeout=budget + 300)
            except subprocess.TimeoutExpired:
                raise TransportError('Modal native build timed out; inspect `modal app list`') from None
            except subprocess.CalledProcessError as exc:
                raise TransportError(f'Modal native build failed (exit {exc.returncode})') from None
        else:
            raise TransportError(f'unsupported remote build backend: {backend}')
        if not result_file.is_file():
            raise TransportError('native compiler worker returned no result')
        result = json.loads(result_file.read_text())
    if not isinstance(result, dict):
        raise TransportError('native compiler worker returned an invalid result object')
    if result.get('error'):
        raise TransportError(result['error'])
    if type(result.get('rc')) is not int or result['rc'] not in (0, 1, 2):
        raise TransportError('native compiler worker returned an invalid status')
    errors = result.get('errors', [])
    if not isinstance(errors, list) or any(not isinstance(e, dict) for e in errors):
        raise TransportError('native compiler worker returned invalid error records')
    failed_sources = result.get('failed_sources', [])
    if not isinstance(failed_sources, list) or not all(rel in candidates for rel in failed_sources):
        raise TransportError('native compiler worker returned invalid failed-source records')
    if not isinstance(result.get('artifacts'), str):
        raise TransportError('native compiler worker returned no artifact bundle')
    blob = base64.b64decode(result['artifacts'], validate=True)
    with tempfile.TemporaryDirectory(prefix='oa-native-artifacts-') as d:
        staged = Path(d)
        unpack(blob, staged)
        manifest = project.load_manifest(staged)
        if manifest is None:
            if result['rc'] != 1:
                raise TransportError('native compiler returned no manifest for a successful build')
            manifest = {'entries': []}
            failed_sources = candidates
        entries = []
        published = {}
        for entry in manifest['entries']:
            relative = Path(entry['source'])
            if relative.is_absolute() or '..' in relative.parts:
                raise TransportError('native worker returned an invalid source path')
            if relative.as_posix() not in candidates:
                raise TransportError('native worker returned a source outside the submitted compilation')
            entry = {**entry, 'source': str(root / relative)}
            entries.append(entry)
            for rel in project.entry_outputs(entry):
                path = project.artifact_path(staged, rel)
                if not path.is_file():
                    raise TransportError(f'native worker artifact missing: {rel}')
                published[rel] = path.read_bytes()
        only = {(entry['source'], kernel['name']) for entry in entries for kernel in entry['kernels']}
        old = project.load_manifest(output)
        apps = old.get('applications', []) if old else []
        owned = {rel.casefold() for app in apps for rel in project.application_outputs(app)}
        if owned.intersection(rel.casefold() for rel in published):
            raise project.ProjectError('native outputs overlap an application; choose another -o or --out-dir')
        merged = {e['cubin']: {**e, 'source': str(project.entry_source_abs(old, e).resolve())}
                  for e in old['entries']} if old else {}
        stale = set()
        built = {e['source']: e['cubin'] for e in entries}
        attempted = {str(root / rel) for rel in candidates}
        failed = {str(root / rel) for rel in failed_sources}
        for cubin, entry in list(merged.items()):
            path = Path(entry['source'])
            if (source.is_dir() and not options.kernel and path.is_relative_to(root)
                    and entry['source'] not in attempted):
                stale.update(project.entry_outputs(entry))
                del merged[cubin]
            elif entry['source'] in built and built[entry['source']] != cubin:
                stale.update(project.entry_outputs(entry))
                del merged[cubin]
            elif entry['source'] in failed:
                merged[cubin] = {**entry, 'build_failed': True}
        for entry in entries:
            if previous := merged.get(entry['cubin']):
                stale.update(project.entry_outputs(previous))
            merged[entry['cubin']] = entry
        with project.defer_sigint():
            project.write_group(output, published)
            project.save_manifest(output, root, options.arch, options.opt_level, list(merged.values()), apps)
            project.remove_outputs(output, sorted(stale - published.keys()))
    project.write_atomic(output / 'build-errors.json', json.dumps(errors, indent=2) + '\n')
    print(f'build artifacts: {output}/cubin/ and {output}/sass/ (manifest.json)')
    return result['rc'], only


def process(request: dict) -> dict:
    """Isolated Linux worker; source files never execute on the client."""
    from openassembler.build import compile_paths
    try:
        with tempfile.TemporaryDirectory(prefix='oa-native-build-') as d:
            staging, output = Path(d), Path(d) / 'output'
            unpack(base64.b64decode(request['source'], validate=True), staging)
            source = project.artifact_path(staging, request['path'])
            options = dict(request['options'])
            options['include_dirs'] = [project.artifact_path(staging, rel) for rel in options['include_dirs']]
            if options.get('compile_commands'):
                options['compile_commands']=project.artifact_path(staging,options['compile_commands'])
            options['device_libraries']=[project.artifact_path(staging,rel)for rel in options.get('device_libraries',[])]
            for library in options.get('device_library_options',[]):
                library['include_dirs']=[project.artifact_path(staging,rel)for rel in library['include_dirs']]
                if library.get('compile_commands'):
                    library['compile_commands']=project.artifact_path(staging,library['compile_commands'])
            result = compile_paths(source, output, Options(**options), request.get('name'))
            # The archive stores relative source records; local publication rebases them.
            files = {p.relative_to(output).as_posix(): p.read_bytes() for p in output.rglob('*')
                     if p.is_file() and p.name != 'writer.lock'}
            source_root = source if source.is_dir() else source.parent
            return {'rc': result.rc, 'artifacts': base64.b64encode(pack(files)).decode(),
                    'errors': result.errors,
                    'failed_sources': [p.relative_to(source_root).as_posix() for p in result.failed_sources]}
    except Exception as exc:
        return {'error': f'native build failed: {exc}'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    # zipfile extraction over SSH does not preserve the executable bit.
    wrapper = Path(__file__).resolve().parent.parent / 'env/cuda.sh'
    if wrapper.is_file():
        wrapper.chmod(wrapper.stat().st_mode | 0o100)
    args.result.write_text(json.dumps(process(json.loads(args.request.read_text()))))


if __name__ == '__main__':
    main()
