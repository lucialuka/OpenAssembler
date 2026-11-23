"""Static NVCC compilation database contracts; no original command is executed.

A record selects preprocessor and device compilation flags, not arbitrary host
build orchestration. CUDA translation units retain exact per-source settings;
unsupported commands fail instead of silently reverting to a different build.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shlex

from .project import ProjectError

_PATH_FLAGS = {'-I':'-I','--include-path':'-I','-isystem':'-isystem','--system-include':'-isystem',
               '-include':'-include','--pre-include':'-include','-iquote':'-iquote'}
_VALUES = {'-D','--define-macro','-U','--undefine-macro','-std','--std','-Xcompiler','--compiler-options',
           '-Xcudafe','--cudafe-options','-Xptxas','--ptxas-options','-Xnvlink','--nvlink-options',
           '--diag-suppress','--diag-warn','--diag-error','-Werror','--Werror','-ccbin','--compiler-bindir',
           '-maxrregcount','--maxrregcount','--threads','-t','--default-stream','-rdc','--relocatable-device-code'}
_OUTPUTS = {'-o','--output-file','-MF','--dependency-output','-MT','--dependency-target-name',
            '-MQ','-odir','--output-directory'}
_STAGES = {'-c','--compile','-ptx','--ptx','-M','-MM','-MD','-MMD','-MP','--generate-dependencies',
           '--generate-nonsystem-dependencies','--generate-dependencies-with-compile',
           '--generate-nonsystem-dependencies-with-compile','--generate-dependency-targets'}
_ARCH = {'-arch','--gpu-architecture','-code','--gpu-code','-gencode','--generate-code'}
_UNSUPPORTED = {'-dlink','--device-link','-lib','--lib','-link','--link','-run','--run','-E','--preprocess',
                '-fatbin','--fatbin','-cubin','--cubin','-l','--library','-L','--library-path','--keep','-keep',
                '--keep-dir','-keep-dir','--cudart','--cudadevrt','--optix-ir','--ltoir'}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path(value, directory):
    path=Path(value)
    return (directory/path).resolve() if not path.is_absolute() else path.resolve()


def _value(args, index, flag):
    if index+1 >= len(args):raise ProjectError(f'compilation database: {flag} needs a value')
    return args[index+1]


def _expand(args, directory, files, active=()):
    expanded=[]
    index=0
    while index<len(args):
        arg=args[index];index+=1
        names = None
        if arg.startswith('@'):names=[arg[1:]]
        elif arg in ('--options-file','-optf'):
            names=_value(args,index-1,arg).split(',');index+=1
        elif arg.startswith(('--options-file=','-optf=')):names=arg.split('=',1)[1].split(',')
        if names is not None:
            for name in names:
                path=_path(name,directory)
                if path in active or len(active)>=8:raise ProjectError('compilation database: cyclic/deep response files')
                if not path.is_file() or path.stat().st_size>4*1024*1024:
                    raise ProjectError(f'compilation database: response file missing or too large: {path}')
                files[str(path)]=_sha(path)
                try:tokens=shlex.split(path.read_text(),posix=True)
                except (OSError,ValueError) as exc:raise ProjectError(f'compilation database response file: {exc}') from None
                expanded.extend(_expand(tokens,directory,files,(*active,path)))
        else:expanded.append(arg)
    return expanded


@dataclass(frozen=True)
class CompileCommand:
    source: Path
    source_argument: str
    directory: Path
    flags: tuple[str,...]
    original_arguments: tuple[str,...]
    database: Path
    database_sha256: str
    response_files: dict[str,str]
    submitted_origin: dict | None = None

    def record(self):
        record = {'schema':1,'kind':'nvcc-compilation-database','database':str(self.database),
                'database_sha256':self.database_sha256,'directory':str(self.directory),
                'source':str(self.source),'original_arguments':list(self.original_arguments),
                'source_argument':str(self.source_argument),
                'device_flags':list(self.flags),'response_files':dict(self.response_files),
                'scope':'recorded translation unit settings; host outputs/link stages are not executed'}
        if self.submitted_origin is not None:record['submitted_origin']=self.submitted_origin
        return record

    def resolve_path(self,value):
        return _path(value,self.directory)


def _command(row, database, digest):
    if not isinstance(row,dict) or not isinstance(row.get('file'),str) or not isinstance(row.get('directory'),str):
        raise ProjectError('compilation database records require string file and directory')
    directory=_path(row['directory'],database.parent)
    source=_path(row['file'],directory)
    source_argument = row['file']
    if not directory.is_dir():raise ProjectError(f'compilation database working directory missing: {directory}')
    arguments=row.get('arguments')
    if arguments is None:
        command=row.get('command')
        if not isinstance(command,str) or re.search(r'[;&|<>`$]',command):
            raise ProjectError('compilation database command must be a plain invocation; use arguments for literal tokens')
        try:arguments=shlex.split(command,posix=True)
        except ValueError as exc:raise ProjectError(f'compilation database command: {exc}') from None
    if not isinstance(arguments,list) or not arguments or any(not isinstance(x,str) or '\0' in x for x in arguments):
        raise ProjectError('compilation database arguments must be a nonempty list of strings')
    original=tuple(arguments)
    if Path(arguments[0]).name in ('ccache','sccache'):
        arguments=arguments[1:]
    if not arguments or Path(arguments[0]).name!='nvcc':
        if source.suffix.lower() in ('.cu','.cuh') or any(x in ('cuda','cu') for x in arguments):
            raise ProjectError(f'CUDA compilation command requires supported nvcc settings: {source}')
        return None
    if not source.is_file():raise ProjectError(f'compilation database source missing (run its source generator first): {source}')
    responses={};args=_expand(arguments[1:],directory,responses);flags=[];seen_source=False;i=0
    while i<len(args):
        token=args[i];key,sep,inline=token.partition('=')
        if token in ('&&','||',';','|','>','<'):raise ProjectError('compilation database does not execute shell commands')
        if key in _UNSUPPORTED or token in _UNSUPPORTED:
            raise ProjectError(f'compilation database unsupported build stage/option: {token}')
        if key in _OUTPUTS:
            i+=1 if sep else 2
            if not sep:_value(args,i-2,key)
            continue
        if key in _STAGES:i+=1;continue
        if key in _ARCH:
            value=inline if sep else _value(args,i,key);i+=1 if sep else 2
            targets=re.findall(r'(?:sm|compute)_\d+[a-z]?',value)
            if not targets or any(target not in ('sm_80','compute_80') for target in targets):
                raise ProjectError(f'compilation database target must be SM80: {value}')
            # The ordinary backend sets the one admitted architecture. Preserve
            # the original target in original_arguments for provenance.
            continue
        if key in ('-x','--x'):
            value=inline if sep else _value(args,i,key);i+=1 if sep else 2
            if value!='cu':raise ProjectError('compilation database NVCC language must be cu')
            continue
        if key in _PATH_FLAGS:
            value=inline if sep else _value(args,i,key);i+=1 if sep else 2
            for part in value.split(','):flags.extend((_PATH_FLAGS[key],part))
            continue
        if token.startswith('-I') and len(token)>2:
            for part in token[2:].split(','):flags.extend(('-I',part))
            i+=1;continue
        if key in _VALUES:
            if sep:flags.append(token);i+=1
            else:flags.extend((token,_value(args,i,key)));i+=2
            continue
        if token.startswith('-'):
            flags.append(token);i+=1;continue
        if _path(token,directory)==source:
            if seen_source:raise ProjectError('compilation database repeats its source operand')
            source_argument=token
            seen_source=True;i+=1;continue
        raise ProjectError(f'compilation database unrecognized positional argument: {token}')
    if not seen_source:raise ProjectError(f'compilation database arguments omit their file: {source}')
    origin=None
    origins=database.parent/'.oa-build-origins.json'
    if origins.is_file() and source.is_relative_to(database.parent):
        origin=json.loads(origins.read_text()).get(source.relative_to(database.parent).as_posix())
    if origin is not None:
        prior=origin.get('device_flags')
        if not isinstance(prior,list) or len(prior)!=len(flags):
            raise ProjectError('relocated compilation settings disagree with submitted origin')
        mappings={}
        for index,token in enumerate(flags[:-1]):
            if token in ('-I','-isystem','-include','-iquote'):
                before,after=flags[index+1],prior[index+1]
                if before in mappings and mappings[before]!=after:
                    raise ProjectError('relocated include aliases have conflicting original file spellings')
                mappings[before]=after
            elif index==0 or flags[index-1] not in ('-I','-isystem','-include','-iquote'):
                if token!=prior[index]:raise ProjectError('relocated non-path setting differs from submitted origin')
        if flags and len(flags)==len(prior) and flags[-1]!=prior[-1] and (len(flags)<2 or flags[-2] not in ('-I','-isystem','-include','-iquote')):
            raise ProjectError('relocated non-path setting differs from submitted origin')
        old_argument=origin.get('source_argument')
        if not isinstance(old_argument,str):raise ProjectError('relocated source argument has no submitted origin')
        mappings[str(source_argument)]=old_argument
        needs_mapping=any(before!=after for before,after in mappings.items())
        if needs_mapping and any('prefix-map=' in flag for flag in prior):
            raise ProjectError('relocation of an existing compiler prefix map requires an explicitly composed configuration')
        # GCC applies the last matching prefix. A source-specific identity map
        # protects an unchanged relative filename from a broader include map.
        for before,after in (sorted(mappings.items(),key=lambda item:len(item[0])) if needs_mapping else ()):
            if '=' in before or '=' in after or ',' in before or ',' in after:
                raise ProjectError('NVCC cannot represent this relocated file-prefix mapping')
            flags.append(f'-Xcompiler=-ffile-prefix-map={before}={after}')
    return CompileCommand(source,source_argument,directory,tuple(flags),original,database,digest,responses,origin)


def load(path):
    path=Path(path).resolve()
    try:
        if path.stat().st_size>16*1024*1024:raise ProjectError('compilation database exceeds 16 MiB')
        rows=json.loads(path.read_text())
    except (OSError,ValueError) as exc:raise ProjectError(f'compilation database {path}: {exc}') from None
    if not isinstance(rows,list):raise ProjectError('compilation database must contain an array of command objects')
    result={};digest=_sha(path)
    for row in rows:
        command=_command(row,path,digest)
        if command is None:continue
        old=result.get(command.source)
        if old is not None and (old.directory!=command.directory or old.flags!=command.flags or old.source_argument!=command.source_argument):
            raise ProjectError(f'ambiguous compilation database variants for {command.source}; provide one selected configuration')
        result[command.source]=command
    return result


def discover(source, explicit=None):
    """Return the selected database path and resolved per-unit contracts."""
    source=Path(source).resolve();root=source if source.is_dir() else source.parent
    path=Path(explicit).resolve() if explicit is not None else root/'compile_commands.json'
    if explicit is None and not path.is_file():return None,{}
    commands=load(path)
    selected={s:c for s,c in commands.items() if s==source or source.is_dir() and s.is_relative_to(root)}
    if explicit is not None and not selected:raise ProjectError('compilation database contains no NVCC units for the selected source')
    return path,selected


def dependencies(command, *, include_dirs=(), extra_flags=()):
    """Resolve the preprocessed dependency graph with the actual NVCC flags."""
    from . import oracle
    from .cppfront import CppFrontendError
    for path in (command.source,command.directory):oracle.tool_path(path)
    flags=[value for directory in include_dirs for value in ('-I',str(Path(directory).resolve()))]
    result=oracle.run_tool('nvcc','-M','-MT','oa_dependency','-x','cu','-arch=sm_80',
                          *flags,*command.flags,*extra_flags,str(command.source_argument),
                          cwd=command.directory,check=False)
    if result.returncode:
        raise CppFrontendError(f'NVCC dependency resolution failed for {command.source}:\n{result.stderr or result.stdout}')
    try:
        body=result.stdout.replace('\\\n',' ').split(':',1)[1]
        paths=shlex.split(body,posix=True)
    except (IndexError,ValueError) as exc:
        raise CppFrontendError(f'could not parse NVCC dependency graph: {exc}') from None
    paths=[str(_path(name.replace('$$','$'),command.directory)) for name in paths]
    resolved={}
    # On macOS the toolkit runs in Colima: /usr/include is the VM's tree,
    # so host-side Path.read_bytes() would hash a different file or no file.
    for start in range(0,len(paths),128):
        batch=paths[start:start+128]
        result=oracle.run_tool('sha256sum','--',*batch,cwd=command.directory,check=False)
        lines=result.stdout.splitlines()
        if result.returncode or len(lines)!=len(batch):
            raise CppFrontendError(f'could not hash NVCC dependency graph in tool environment: {result.stderr}')
        for name,line in zip(batch,lines):
            digest=line.lstrip('\\').split(maxsplit=1)[0]
            if not re.fullmatch('[0-9a-f]{64}',digest):raise CppFrontendError('invalid dependency digest from tool environment')
            resolved[name]=digest
    if str(command.source) not in resolved:raise CppFrontendError('NVCC dependency graph omitted its translation unit')
    return resolved
