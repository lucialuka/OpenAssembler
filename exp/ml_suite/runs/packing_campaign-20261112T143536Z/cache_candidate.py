"""Content-addressed native bundles; weights excluded and validated on every hit."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile
from functools import lru_cache

from openassembler.modelfront.kernels import canonical


@lru_cache(maxsize=1)
def toolchain():
    from openassembler import oracle
    return {tool:oracle.run_tool(tool,'--version').stdout for tool in ['ptxas','nvdisasm']}


def key_for(plan,*,opt_level='opt',sass_listings=True):
    from openassembler.gemm_dispatch import compiler_identity
    import openassembler.modelfront.kernels as kernels
    root=Path(kernels.__file__).parent
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')}
    inputs={'schema':1,'compiler':compiler_identity(),'model_sources':sources,'toolchain':toolchain(),
            'plan':plan,'opt_level':opt_level,'sass_listings':sass_listings,'arch':'sm_80'}
    return hashlib.sha256(canonical(inputs)).hexdigest(),inputs


def checked_files(root,plan,key):
    from openassembler.modelfront.runtime import validate_bundle
    record=json.loads((root/'cache-record.json').read_text())
    if record.get('schema')!=1 or record.get('key')!=key:raise ValueError('native cache identity mismatch')
    files=record.get('files')
    if not isinstance(files,dict) or not files:raise ValueError('native cache inventory missing')
    for name,want in files.items():
        path=(root/name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(root.resolve()) or not path.is_file() or path.is_symlink():
            raise ValueError('native cache file escapes or is missing')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=want:raise ValueError('native cache integrity mismatch')
    validate_bundle(root,plan)
    return files


def cached_build(plan,weights,output,cache,*,build=None,opt_level='opt',sass_listings=False):
    from openassembler.modelfront.kernels import build_bundle
    from openassembler.modelfront.runtime import save_weights
    build=build or build_bundle
    output,cache=Path(output),Path(cache)
    key,inputs=key_for(plan,opt_level=opt_level,sass_listings=sass_listings)
    entry=cache/key
    miss_reason=None
    if entry.is_dir():
        try:
            files=checked_files(entry,plan,key)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            miss_reason=str(exc)
        else:
            for name in files:
                path=output/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes((entry/name).read_bytes())
            if weights is not None:save_weights(weights,plan,output)
            return {'hit':True,'key':key,'native_ledger':json.loads((output/'native-ledger.json').read_text())}
    ledger=build(plan,weights,output,opt_level=opt_level,sass_listings=sass_listings)
    cache.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.oa-cache-',dir=cache) as directory:
        staging=Path(directory)/'entry';staging.mkdir()
        files={}
        for path in output.rglob('*'):
            if not path.is_file() or path.is_symlink():continue
            name=path.relative_to(output).as_posix()
            if name=='weights.json' or name.startswith('weights/') or name.endswith('writer.lock'):continue
            if not (name in ('plan.json','native-ledger.json') or name.startswith(('native/','native_tensor/','sources/'))):continue
            destination=staging/name;destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(path.read_bytes());files[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        (staging/'cache-record.json').write_bytes(canonical({'schema':1,'key':key,'inputs':inputs,'files':files}))
        checked_files(staging,plan,key)
        if entry.exists():
            # Invalid cache entries never become trusted simply because a fresh
            # build succeeded; replace their directory only after validation.
            retired=cache/(key+'.invalid.'+str(os.getpid()))
            if retired.exists():shutil.rmtree(retired)
            entry.rename(retired)
            staging.rename(entry)
            shutil.rmtree(retired)
        else:
            staging.rename(entry)
    return {'hit':False,'key':key,'native_ledger':ledger,'rebuild_reason':miss_reason}
