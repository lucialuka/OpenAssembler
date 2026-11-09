"""CUDA C/C++ -> PTX, via nvcc's own frontend and NVVM optimizer (--ptx).

Unlike the SASS side of this project, we don't reverse-engineer C++ parsing
or NVVM's optimizer -- nvcc is already vendored as a black-box *oracle*
(env/cuda.sh, used everywhere else in this repo to check our own work), and
turning C++ into PTX is squarely what it's for. What we replace is
everything downstream of PTX: our own backend (openassembler.backend.ra) takes
nvcc's PTX and generates the SASS itself, same as for hand-written PTX or
openassembler.pyfront's output. See README.md "How it works" -- this is the same
"reuse the container, replace the code generator" split, just one level
earlier in the pipeline.

Kernels should be `extern "C" __global__ void name(...)` so nvcc doesn't
C++-name-mangle them -- otherwise --entry has to match the mangled symbol,
which this module does not attempt to demangle for you.
"""

from __future__ import annotations

import re
from pathlib import Path

from openassembler import oracle

ROOT = oracle.ROOT
_ENTRY = re.compile(r"(?:\.(?:visible|weak)\s+)?\.entry\s+([\w$]+)\s*\(")


class CppFrontendError(Exception):
    pass


def _kernel_names(ptx: str) -> list[str]:
    return _ENTRY.findall(ptx)


def compile_to_ptx_module(path: Path, *, arch: str = "sm_80", include_dirs: list[Path] | None = None,
                          extra_flags: list[str] | None = None) -> tuple[str, list[str]]:
    """Run `nvcc --ptx` on a .cu/.cuh/.cpp/.cc/.cxx file; return (ptx_text, every
    __global__ entry in it). The file is compiled as CUDA (`-x cu`) whatever its
    extension: nvcc treats .cpp as host-only C++ (where __global__ is undefined)
    and refuses .cuh outright, but a repo's kernels live in all of these.
    include_dirs become -I flags (the file's own directory is always first, so
    `#include "local.h"` works even when the source is staged, see below);
    extra_flags are passed to nvcc verbatim (-D..., -std=c++17, ...).
    On macOS nvcc runs in the colima VM, which only mounts $HOME; a source outside
    it is compiled from a scratch copy, and its #includes must then be reachable
    from under $HOME too. Raises CppFrontendError with nvcc's own message on a
    compile error, or if nvcc produced no kernels. The intermediate .ptx goes to
    a scratch file and is removed -- nothing is written next to your source."""
    import os
    import uuid

    scratch = oracle.SCRATCH
    scratch.mkdir(parents=True, exist_ok=True)
    tag = f"cppfront_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    out_path = scratch / f"{tag}.ptx"
    staged = None
    try:
        src_arg = oracle.tool_path(path)
    except oracle.OracleError:
        staged = scratch / f"{tag}{path.suffix}"
        staged.write_bytes(path.read_bytes())
        src_arg = oracle.tool_path(staged)
    incs = []
    for d in [path.resolve().parent, *(include_dirs or [])]:
        try:
            flag = f"-I{oracle.tool_path(d)}"
        except oracle.OracleError:
            continue          # macOS, outside $HOME: the VM can't see it anyway
        if flag not in incs:
            incs.append(flag)
    try:
        r = oracle.run_tool("nvcc", "--ptx", "-x", "cu", f"-arch={arch}", *incs, *(extra_flags or []),
                            "-o", oracle.tool_path(out_path), src_arg, check=False)
        if r.returncode != 0:
            msg = "\n".join(l for l in (r.stderr or r.stdout).splitlines()
                            if not l.startswith("time=")).strip()      # drop colima's own exit-status noise
            hint = ""
            if staged is not None and "No such file" in msg:
                hint = ("\n  (this source is outside $HOME, so nvcc -- which runs in the colima VM on "
                        "macOS -- compiled a copy and can't see its #includes; move it under $HOME)")
            raise CppFrontendError(f"nvcc failed on {path}:\n{msg}{hint}")
        ptx = out_path.read_text()
    finally:
        out_path.unlink(missing_ok=True)
        if staged is not None:
            staged.unlink(missing_ok=True)
    names = _kernel_names(ptx)
    if not names:
        raise CppFrontendError(f"nvcc produced no __global__ kernels from {path} -- "
                                "is the kernel only a template that's never instantiated?")
    return ptx, names


def compile_to_ptx(path: Path, *, entry: str | None, arch: str = "sm_80") -> tuple[str, str]:
    """Single-kernel form of compile_to_ptx_module: (ptx_text, kernel_name), where
    `entry` picks the kernel when the file has several."""
    ptx, names = compile_to_ptx_module(path, arch=arch)
    if entry:
        if entry not in names:
            raise CppFrontendError(f"no kernel named {entry!r} in {path} (nvcc may have "
                                    f"C++-mangled it; use extern \"C\"); have {names}")
        return ptx, entry
    if len(names) != 1:
        raise CppFrontendError(f"{path} has {len(names)} kernels, pass --kernel NAME; have {names}")
    return ptx, names[0]
