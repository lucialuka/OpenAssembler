"""Turns any supported source file into PTX for the backend, so
openassembler.backend.ra always starts from PTX text regardless of whether the
source was already PTX, a Python kernel (openassembler.pyfront), or CUDA C/C++
(openassembler.cppfront, which shells out to nvcc once -- see its docstring for
why that's not "cheating" the same way ptxas's code generation would be).

to_ptx_module() is what `openassembler compile` uses: one source file -> one PTX
module holding ALL of its kernels (they end up in one cubin). to_ptx() is the
older single-kernel form, kept for callers that want exactly one kernel.
"""

from __future__ import annotations

from pathlib import Path

from openassembler.project import CPP_EXTS, CU_EXTS, PTX_EXTS, PY_EXTS


class FrontendError(Exception):
    pass


def _python_module(path: Path, arch: str) -> tuple[str, list[str], list[str]]:
    """Statically compile all supported @kernel definitions; never import user code."""
    from openassembler.pyfront.compile import compile_file
    ptx, names = compile_file(path, arch)
    return ptx, list(names), []


def to_ptx_module(path: Path, *, arch: str, include_dirs: list[Path] | None = None,
                  nvcc_flags: list[str] | None = None) -> tuple[str, list[str], list[str]]:
    """(ptx_text, kernel names in source order, notes). Raises FrontendError for an
    unsupported extension; the frontends' own errors (pyfront.NotSupported,
    cppfront.CppFrontendError) propagate unchanged for the caller to report."""
    suf = path.suffix.lower()
    if suf in PTX_EXTS:
        from openassembler.cppfront import _kernel_names

        text = path.read_text()
        return text, _kernel_names(text), []
    if suf in PY_EXTS:
        return _python_module(path, arch)
    if suf in CU_EXTS or suf in CPP_EXTS:
        from openassembler.cppfront import compile_to_ptx_module

        ptx, names = compile_to_ptx_module(path, arch=arch, include_dirs=include_dirs, extra_flags=nvcc_flags)
        return ptx, names, []
    raise FrontendError(f"don't know how to compile {path.suffix or 'extension-less'!r} files "
                        f"(supported: {' '.join(PTX_EXTS + PY_EXTS + CU_EXTS + CPP_EXTS)})")


def to_ptx(path: Path, *, entry: str | None, arch: str) -> tuple[str, str]:
    """Returns (ptx_text, kernel_name). `entry` selects which kernel/function
    to compile when the source has more than one; for .ptx with exactly one
    kernel it's optional, same as before this function existed."""
    suf = path.suffix.lower()
    if suf in PTX_EXTS:
        text = path.read_text()
        if entry:
            return text, entry
        # Caller resolves the single-kernel case itself (needs parse_module,
        # which lives in openassembler.ptx -- avoid importing it here just for that).
        return text, None  # type: ignore[return-value]

    if suf in PY_EXTS:
        from openassembler.pyfront.compile import find_kernel

        fn = find_kernel(path, entry)
        from openassembler.pyfront.compile import compile_kernel as py_compile

        return py_compile(fn, arch=arch), fn.__name__

    if suf in CU_EXTS or suf in CPP_EXTS:
        from openassembler.cppfront import compile_to_ptx

        return compile_to_ptx(path, entry=entry, arch=arch)

    raise FrontendError(f"don't know how to compile {path.suffix!r} files "
                         f"(supported: {PTX_EXTS + PY_EXTS + CU_EXTS + CPP_EXTS})")
