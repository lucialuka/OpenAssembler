"""Source/output resolution shared by the three repository commands.

No implicit compilation in ``run`` and no global registry: the conventional
build root for a directory is REPO/output. Explicit artifact directories and
the old working-directory output layout remain usable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openassembler import project


def output_dir(source: Path, explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).resolve()
    source = Path(source).resolve()
    return source / 'output' if source.is_dir() else Path('output').resolve()


@dataclass
class RunTarget:
    out_dir: Path
    application: str | None = None
    only: set[tuple[str, str]] | None = None


def _selection(source: Path, out: Path, manifest: dict, name: str | None) -> RunTarget | None:
    apps = [app for app in manifest.get('applications', [])
            if Path(app['source']).resolve() == source and (name is None or app['name'] == name)]
    entries = [e for e in manifest['entries']
               if ((path := project.entry_source_abs(manifest, e).resolve()) == source
                   or (source.is_dir() and path.is_relative_to(source)))]
    if apps:
        if len(apps) > 1:
            raise project.ProjectError('select one application with --application; found '
                                       + ', '.join(app['name'] for app in apps))
        return RunTarget(out, apps[0]['name'])
    if entries and name is None:
        return RunTarget(out, only={(str(project.entry_source_abs(manifest, e).resolve()), k['name'])
                                    for e in entries for k in e['kernels']})
    return None


def resolve_run(path: str | Path | None, explicit: str | Path | None = None,
                application: str | None = None) -> RunTarget:
    """Resolve a source repository/file or an already-built output directory.

    Source paths restrict execution to their own manifest records. A custom
    output must be supplied explicitly, rather than searching arbitrary trees.
    """
    source = Path(path).resolve() if path is not None else Path.cwd().resolve()
    if path is None and explicit is not None:
        source, explicit = Path(explicit).resolve(), None
    if explicit is None and (source / project.MANIFEST).is_file():
        manifest = project.load_manifest(source)
        apps = manifest.get('applications', [])
        if application is None and not manifest['entries'] and len(apps) == 1:
            application = apps[0]['name']
        return RunTarget(source, application)
    candidates = [Path(explicit).resolve()] if explicit is not None else [output_dir(source)]
    if explicit is None:
        # Compatibility with builds created before repo/output became the default.
        candidates += [Path('output').resolve()]
        if source.is_file():
            candidates += [source.parent / 'output']
    selections = []
    for out in dict.fromkeys(candidates):
        if (manifest := project.load_manifest(out)) is not None:
            if selected := _selection(source, out, manifest, application):
                if source.is_dir() and out == output_dir(source) and explicit is None:
                    return selected  # the conventional repository-local build takes priority
                selections.append(selected)
    if len(selections) > 1:
        raise project.ProjectError('multiple builds match this source; select --out-dir: '
                                   + ', '.join(str(s.out_dir) for s in selections))
    if selections:
        return selections[0]
    if path is None and explicit is None:
        # The historical bare `run` also accepts a copied output without its sources.
        return RunTarget(Path('output').resolve(), application)
    expected = candidates[0]
    raise project.ProjectError(f'no compiled build for {source} in {expected}; '
                               f'run `compile {source}` in OpenAssembler first, or select '
                               '--out-dir DIR from its previous build. `run` never compiles sources.')
