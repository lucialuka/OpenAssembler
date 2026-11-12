"""Validated backend selection and bounded SSH request transport.

The transport only stages code and data; it never changes kernel selection,
scheduling, binaries or launch contracts.
"""
from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path


class TransportError(ValueError):
    pass


def ssh_host(host: str) -> str:
    # SCP treats ':' as a path separator. Bracketed IPv6 and SSH config aliases
    # are allowed; flags, URIs, shell syntax and paths are not destinations.
    if not isinstance(host, str) or not re.fullmatch(
            r'(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?(?:[A-Za-z0-9][A-Za-z0-9_.-]*|\[[0-9A-Fa-f:.]+\])', host):
        raise TransportError('SSH target must be user@host or an SSH config alias; '
                             'configure ports, keys and ProxyJump in ~/.ssh/config')
    return host


def backend(value: str | None, host: str | None = None) -> tuple[str, str | None]:
    value = 'auto' if value is None else value
    if not isinstance(value, str) or not value:
        raise TransportError('--backend must be auto, local, modal, or an SSH target')
    if value == 'ssh':
        if not host:
            raise TransportError('--backend ssh requires --host user@host (or a saved application host)')
        return 'ssh', ssh_host(host)
    if value in ('auto', 'local', 'modal'):
        if host is not None:
            if value == 'auto':
                return 'ssh', ssh_host(host)
            raise TransportError('--host requires --backend ssh, auto, or the same SSH target')
        return value, None
    target = ssh_host(value)
    if host is not None and target != ssh_host(host):
        raise TransportError('conflicting SSH targets in --backend and --host')
    return 'ssh', target


def ssh_destination(host: str) -> str:
    """SSH uses bare IPv6, whereas SCP needs the bracketed destination."""
    return ssh_host(host).replace('[', '').replace(']', '')


def runtime_files(*, native: bool = False) -> dict[str, bytes]:
    root = Path(__file__).resolve().parent.parent
    paths = list((root / 'openassembler').rglob('*.py'))
    if native:
        paths += list((root / 'openassembler/kernels').glob('*.json'))
        paths += list((root / 'openassembler/backend').glob('*.json'))
        paths += list((root / 'specs/sm_80').rglob('*.json'))
        paths += list((root / 'tools').glob('*.py'))
        paths += [root / 'corpus/sm_80/pairs.jsonl', root / 'corpus/sm_80/machine_model.json',
                  root / 'env/cuda.sh']
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in paths
            if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts}


def ssh_request(host: str, request: Path, result: Path, runtime: bytes,
                module: str, budget: float) -> None:
    """Run one JSON worker on an existing Linux host, returning its result file.

    Remote directories are unique per invocation. Failed cleanup must not hide
    the original result/error. BatchMode prevents unattended password prompts.
    """
    host = ssh_host(host)
    destination = ssh_destination(host)
    if not all(shutil.which(t) for t in ('ssh', 'scp')):
        raise TransportError('SSH transport needs ssh and scp on this machine')
    opts = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20']

    def call(cmd, *, timeout=120):
        try:
            return subprocess.run(cmd, check=True, timeout=timeout, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except subprocess.TimeoutExpired:
            raise TransportError(f'SSH transport to {host} timed out after {timeout:g}s') from None
        except subprocess.CalledProcessError as exc:
            raise TransportError(f'SSH transport to {host} failed (exit {exc.returncode}): '
                                 f'{(exc.stdout or "")[-4000:]}') from None

    rdir = call(['ssh', *opts, destination, 'mktemp -d /tmp/openassembler_request_XXXXXXXXXX']).stdout.strip()
    if not re.fullmatch(r'/tmp/openassembler_request_[A-Za-z0-9]+', rdir):
        raise TransportError(f'SSH host returned an invalid temporary directory: {rdir!r}')
    bundle = request.parent / 'runtime.zip'
    bundle.write_bytes(runtime)
    q = shlex.quote
    try:
        call(['scp', '-q', *opts, str(bundle), str(request), f'{host}:{rdir}/'])
        # zipfile is stdlib, and the archive is generated from this installation.
        setup = "import zipfile; zipfile.ZipFile('runtime.zip').extractall('.')"
        command = (f'cd {q(rdir)} && python3 -c {q(setup)} && '
                   f'timeout {budget + 60:g}s python3 -m {q(module)} '
                   f'--request {q(request.name)} --result {q(result.name)}')
        completed = call(['ssh', *opts, destination, command], timeout=budget + 120)
        print(completed.stdout, end='')
        call(['scp', '-q', *opts, f'{host}:{rdir}/{result.name}', str(result)])
    finally:
        try:
            subprocess.run(['ssh', *opts, destination, f'rm -rf -- {q(rdir)}'], timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            pass
