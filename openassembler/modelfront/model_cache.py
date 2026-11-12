"""Content-addressed cache for validated OpenAssembler model bundles.

Weights stay out of the cache. Only OA-owned executable modules, their plan and
ledger, and OA PTX diagnostics are retained; reference cubins never enter a hit.
"""
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .kernels import canonical


def default_cache_dir():
    override=os.environ.get('OPENASSEMBLER_MODEL_CACHE')
    return Path(override).expanduser() if override else Path.home()/'.cache'/'openassembler'/'native-models'


@lru_cache(maxsize=1)
def toolchain_identity():
    from openassembler import oracle
    return {name:oracle.run_tool(name,'--version').stdout for name in ('ptxas','nvdisasm')}


def key_for(plan,*,opt_level='opt',sass_listings=True):
    from .kernels import prepare,tensor_recipes
    from openassembler.gemm_dispatch import compiler_identity
    package=Path(__file__).resolve().parents[1]
    sources={path.relative_to(package).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
             for path in package.rglob('*') if path.is_file() and path.suffix in ('.py','.json')
             and not path.is_symlink() and '__pycache__' not in path.parts}
    specs,steps,kernels=prepare(plan)
    generated=[kernels[name].ptx for name in sorted(kernels)]
    recipes=tensor_recipes(plan,specs)
    generated.extend(stage.ptx for recipe in recipes.values() for stage in recipe.stages)
    generated_source_sha256=hashlib.sha256(('\n'.join(generated)).encode()).hexdigest()
    identity={'schema':1,'compiler_identity':compiler_identity(),'sources':sources,
              'generated_source_sha256':generated_source_sha256,
              'toolchain':toolchain_identity(),'plan':plan,'arch':'sm_80',
              'opt_level':opt_level,'sass_listings':sass_listings}
    return hashlib.sha256(canonical(identity)).hexdigest(),identity


@contextmanager
def _key_lock(cache,key):
    cache.mkdir(parents=True,exist_ok=True)
    handle=(cache/(key+'.lock')).open('a+b')
    try:
        try:
            import fcntl
        except ImportError:
            yield
        else:
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX)
            try:yield
            finally:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
    finally:handle.close()


def _cacheable(relative):
    path=Path(relative)
    if path.name=='reference.cubin' or '.build' in path.parts:return False
    if relative in ('plan.json','native-ledger.json','sources/native.ptx'):return True
    if len(path.parts)==3 and path.parts[0]=='native' and path.parts[1] in ('cubin','sass'):
        return path.suffix in ('.cubin','.sass')
    if len(path.parts)==3 and path.parts[0]=='native_tensor':
        return path.name in ('native.cubin','native.sass','kernel.ptx')
    return False


def _remove_donor_artifacts(output,ledger):
    owned={ledger['cubin'],*(row['module'] for row in ledger.get('tensor_kernels',{}).values())}
    for path in output.rglob('*.cubin'):
        relative=path.relative_to(output).as_posix()
        if path.is_file() and not path.is_symlink() and relative not in owned:path.unlink()
    for path in sorted(output.rglob('.build'),key=lambda p:len(p.parts),reverse=True):
        if path.is_dir():shutil.rmtree(path)
    # The compiler listing contains ptxas-reference filenames even though the
    # OA cubin is the only executable artifact retained for deployment.
    (output/'native'/'manifest.json').unlink(missing_ok=True)


def _file_hashes(root):
    result={}
    for path in root.rglob('*'):
        if not path.is_file() or path.is_symlink():continue
        relative=path.relative_to(root).as_posix()
        if _cacheable(relative):result[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _checked_entry(root,plan,key,inputs):
    from .runtime import validate_bundle
    root=Path(root)
    if root.is_symlink() or not root.is_dir():raise ValueError('cache entry is not a regular directory')
    record=json.loads((root/'cache-record.json').read_text())
    files=record.get('files')
    if record.get('schema')!=1 or record.get('key')!=key or record.get('inputs')!=inputs:
        raise ValueError('model cache identity mismatch')
    if not isinstance(files,dict) or not files or any(not _cacheable(name) for name in files):
        raise ValueError('model cache inventory is invalid')
    for path in root.rglob('*'):
        if path.is_symlink():raise ValueError('model cache contains a symbolic link')
        if path.is_file():
            name=path.relative_to(root).as_posix()
            if name!='cache-record.json' and name not in files:
                raise ValueError('model cache contains an unaccounted artifact')
    for name,want in files.items():
        candidate=root/name
        if candidate.is_symlink() or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('model cache path is unsafe')
        path=candidate.resolve(strict=True)
        if not path.is_relative_to(root.resolve()) or not path.is_file():raise ValueError('model cache file is missing')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=want:raise ValueError('model cache integrity mismatch')
    validate_bundle(root,plan)
    return files


def cached_build_bundle(plan,weights,output_dir,*,cache_dir=None,opt_level='opt',sass_listings=True,build=None):
    """Build or restore an OA bundle and always serialize the caller's weights."""
    from .kernels import _build_bundle_uncached
    from .runtime import save_weights,validate_bundle
    build=build or _build_bundle_uncached
    output=Path(output_dir).resolve();cache=Path(cache_dir).expanduser().resolve() if cache_dir else default_cache_dir().resolve()
    key,inputs=key_for(plan,opt_level=opt_level,sass_listings=sass_listings);entry=cache/key
    rebuild_reason='cache miss'
    with _key_lock(cache,key):
        try:
            files=_checked_entry(entry,plan,key,inputs)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            files=None
            rebuild_reason=str(exc)
        if files is not None:
            output.mkdir(parents=True,exist_ok=True)
            for name in files:
                destination=output/name;destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes((entry/name).read_bytes())
            if weights is not None:save_weights(weights,plan,output)
            else:
                (output/'weights.json').unlink(missing_ok=True)
                shutil.rmtree(output/'weights',ignore_errors=True)
            ledger=json.loads((output/'native-ledger.json').read_text())
            _remove_donor_artifacts(output,ledger)
            validate_bundle(output,plan)
            return {'hit':True,'key':key,'cached_files':len(files),
                    'native_ledger':json.loads((output/'native-ledger.json').read_text())}
        ledger=build(plan,weights,output,opt_level=opt_level,sass_listings=sass_listings)
        files=_file_hashes(output)
        if not files:raise ValueError('native model build produced no cacheable OA artifacts')
        with tempfile.TemporaryDirectory(prefix='.oa-model-cache-',dir=cache) as temporary:
            staging=Path(temporary)/'entry';staging.mkdir()
            for name in files:
                source=output/name;destination=staging/name;destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes(source.read_bytes())
            record={'schema':1,'key':key,'inputs':inputs,'files':files}
            (staging/'cache-record.json').write_bytes(canonical(record))
            _checked_entry(staging,plan,key,inputs)
            if entry.is_symlink():entry.unlink()
            elif entry.exists():shutil.rmtree(entry)
            staging.rename(entry)
        # Drop ptxas donor cubins from the private build tree after OA output is
        # fully built and validated. They are not runtime artifacts or cache inputs.
        _remove_donor_artifacts(output,ledger)
        validate_bundle(output,plan)
        return {'hit':False,'key':key,'cached_files':len(files),'rebuild_reason':rebuild_reason,
                'native_ledger':ledger}
