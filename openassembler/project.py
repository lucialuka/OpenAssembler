"""Project layout for `openassembler compile` / `run` / `cnr`: which files in a
source tree are kernels, where their outputs go, and the manifest that ties the
two together.

Output folder (default ./output, `--out-dir DIR`):

    output/cubin/<name>.cubin            our SASS, in a ptxas-built ELF container (loadable by the driver)
    output/sass/<name>.sass              nvdisasm's listing of exactly that cubin
    output/manifest.json                 what was compiled, from what, by whom (schema below)
    output/.build/ptx/<name>.ptx         the PTX that was compiled (frontend output for .py/.cu)
    output/.build/reference/<name>.cubin ptxas -O3 on the same PTX: `run`'s correctness/speed reference
    output/run.json                      the last `run`/`cnr` report (written by openassembler.runner)

<name> is the source's stem for a single file (or `-o NAME`); for a directory it
is the source's path relative to the root with the extension replaced
(kernels/gemm.cu -> cubin/kernels/gemm.cubin), keeping the full filename when two
sources in one directory share a stem (a.cu + a.py -> a.cu.cubin, a.py.cubin).

manifest.json schema (version 1; `openassembler run` reads nothing else):

    {"openassembler": "0.2.0",              version that wrote it (run warns on a mismatch)
     "arch": "sm_80", "opt_level": "opt",
     "root": "/abs/path",                   SRC itself (a directory) or a single SRC's directory
     "entries": [{
        "source": "kernels/gemm.cu",        relative to root when under it, else absolute
        "source_sha256": "...",             run warns "out of date" when the file changed
        "language": "ptx" | "python" | "cuda",
        "cubin": "cubin/kernels/gemm.cubin",            all four paths relative to the output folder
        "sass": "sass/kernels/gemm.sass" | null,        (null with --no-sass)
        "ptx": ".build/ptx/kernels/gemm.ptx",
        "reference": ".build/reference/kernels/gemm.cubin",
        "kernels": [{"name": "gemm", "params": [["u64", "a"], ["u32", "n"], ...],
                     "compiled_by": "openassembler" | "ptxas-fallback",
                     "backend": "openassembler" | "ptxas-fallback",
                     "instructions": 42, "registers": 24,
                     "verify": "ok" | "failed" | "not-run" | "not-verified",
                     "note": "...",                       optional: why it fell back / wasn't verified
                     "source_params": ["a", "b", "n"]}],  optional: CUDA C names of params (nvcc's
                                                          PTX calls them <kernel>_param_<i>)
        "failed": [{"name": "k2", "error": "..."}]}]}   kernels left out of this cubin (optional)

Rules the compile command keeps (see openassembler.build): an entry's outputs are
exactly the files listed in it; a single-file compile replaces only the entry with
the same cubin path; outputs of entries whose source no longer exists are deleted,
and nothing outside the output folder or not listed in the previous manifest is
ever deleted. Every file is written to a temp name and renamed into place, so an
interrupted compile (Ctrl-C) never leaves a half-written output.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import os
import re
import signal
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from openassembler import __version__

PTX_EXTS = (".ptx",)
PY_EXTS = (".py",)
CU_EXTS = (".cu", ".cuh")
CPP_EXTS = (".cpp", ".cc", ".cxx", ".c")
SOURCE_EXTS = PTX_EXTS + PY_EXTS + CU_EXTS + CPP_EXTS
# never walked into: VCS/editor dirs (all hidden), virtualenvs, caches, build output
# Generated application/kernel artifacts are not source inputs.  In particular,
# walking a previously captured PyTorch repository must not package its output
# cubins, PTX cache or reports into the next source bundle.
SKIP_DIRS = {".venv", "venv", "node_modules", "__pycache__", "build", "dist",
             "output", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox"}
MANIFEST = "manifest.json"
MANIFEST_SCHEMA = 1


class ProjectError(Exception):
    """A user-facing problem with the source tree or output folder (one clear line)."""


def language_of(path: Path) -> str | None:
    suf = path.suffix.lower()
    if suf in PTX_EXTS:
        return "ptx"
    if suf in PY_EXTS:
        return "python"
    if suf in CU_EXTS + CPP_EXTS:
        return "cuda"
    return None


# ---- what is a kernel source (static checks only: nothing is imported or compiled) ----

_C_COMMENT = re.compile(r"//[^\\n]*|/\\*.*?\\*/", re.S)
_PTX_ENTRY = re.compile(r"\.entry\s+[\w$]+")


def has_cuda_kernel(text: str) -> bool:
    """Detect a __global__ function outside comments and string/char literals."""
    text = re.sub(r"//[^\n]*|/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', " ", text)
    return bool(re.search(r"\b__global__\b", text))


def has_ptx_entry(text: str) -> bool:
    return bool(_PTX_ENTRY.search(re.sub(r"//[^\n]*|/\*.*?\*/", " ", text, flags=re.S)))


def python_kernel_names(text: str) -> list[str] | None:
    """Find top-level functions decorated by this package without importing them."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    kernel_aliases, module_aliases = {"kernel"}, set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("openassembler.pyfront"):
            for item in node.names:
                if item.name in ("kernel", "*"):
                    kernel_aliases.add(item.asname or item.name)
        elif isinstance(node, ast.Import):
            for item in node.names:
                if item.name == "openassembler.pyfront":
                    module_aliases.add(item.asname or item.name)
                elif item.name == "openassembler":
                    module_aliases.add((item.asname or item.name) + ".pyfront")

    def dotted(expr: ast.expr) -> str:
        if isinstance(expr, ast.Name):
            return expr.id
        if isinstance(expr, ast.Attribute):
            return dotted(expr.value) + "." + expr.attr
        return ""

    out = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            name = dotted(target)
            if name in kernel_aliases or (name.endswith(".kernel") and name[:-7] in module_aliases):
                out.append(node.name)
                break
    return out


@dataclass
class Classified:
    path: Path                 # absolute
    rel: str                   # posix path relative to the root (or the file name)
    language: str | None       # ptx/python/cuda, or None if skipped
    skip_reason: str = ""      # why it isn't compiled (summary key)
    py_kernels: list[str] = field(default_factory=list)


def classify(path: Path, rel: str) -> Classified:
    lang = language_of(path)
    if lang is None:
        return Classified(path, rel, None, f"{path.suffix.lower() or '(no extension)'} files")
    try:
        text = path.read_text(errors="replace")
    except OSError as e:
        return Classified(path, rel, None, f"unreadable ({e.strerror})")
    if lang == "ptx":
        return Classified(path, rel, lang) if has_ptx_entry(text) else \
            Classified(path, rel, None, ".ptx without .entry")
    if lang == "cuda":
        return Classified(path, rel, lang) if has_cuda_kernel(text) else \
            Classified(path, rel, None, f"{path.suffix.lower()} host code (no __global__)")
    ks = python_kernel_names(text)
    if ks is None:
        return Classified(path, rel, None, ".py not parseable as Python 3")
    if not ks:
        return Classified(path, rel, None, ".py without @kernel")
    return Classified(path, rel, lang, py_kernels=ks)


# ---- walking a tree ---------------------------------------------------------------

def _inside(p: Path, d: Path) -> bool:
    try:
        p.relative_to(d)
        return True
    except ValueError:
        return False


def _git_files(root: Path) -> list[Path] | None:
    """Tracked + untracked-but-not-ignored files under `root` (git's own .gitignore
    handling), or None when root isn't in a git work tree, git is missing, or root
    itself is ignored by an enclosing repo (then git would list nothing at all)."""
    try:
        inside = subprocess.run(["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
                                capture_output=True, text=True, timeout=30)
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            return None
        if subprocess.run(["git", "-C", str(root), "check-ignore", "-q", "."],
                          capture_output=True, timeout=30).returncode == 0:
            return None
        r = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached", "--others",
                            "--exclude-standard"], capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    # paths are relative to root (git -C); deleted-but-still-indexed files are dropped
    return [root / p for p in r.stdout.decode(errors="surrogateescape").split("\0") if p]


def walk(root: Path, out_dir: Path) -> tuple[list[Path], bool]:
    """All candidate files under root (absolute, sorted), and whether git chose them.
    The output folder is never part of the walk, wherever it is."""
    root, out_dir = root.resolve(), out_dir.resolve()
    files = _git_files(root)
    used_git = files is not None
    if files is None:
        files = []
        for d, dirs, names in os.walk(root):
            dp = Path(d)
            dirs[:] = sorted(x for x in dirs if not x.startswith(".") and x not in SKIP_DIRS
                             and not _inside((dp / x).resolve(), out_dir))
            files += [dp / n for n in names if not n.startswith(".")]
    keep = []
    for f in files:
        rel = f.relative_to(root)
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1]):
            continue
        if rel.name.startswith("."):          # .gitignore, .clang-format, ...: config, never a kernel
            continue
        if f.is_symlink() or not _inside(f.resolve(), root):
            continue
        if not f.is_file() or _inside(f.resolve(), out_dir):
            continue
        keep.append(f)
    return sorted(keep), used_git


def output_names(rels: list[str]) -> dict[str, str]:
    """rel source path -> output base name (no extension), mirroring the tree:
    k/gemm.cu -> k/gemm. Two compiled sources in one directory with the same stem
    (case-insensitively: macOS file systems are) keep their full file name instead:
    a.cu -> a.cu, a.py -> a.py (so cubin/a.cu.cubin, cubin/a.py.cubin)."""
    groups: dict[tuple[str, str], list[str]] = {}
    for r in rels:
        p = Path(r)
        groups.setdefault((str(p.parent).lower(), p.stem.lower()), []).append(r)
    out = {}
    for members in groups.values():
        for r in members:
            p = Path(r)
            out[r] = (p.parent / p.stem).as_posix() if len(members) == 1 else p.as_posix()
    # a.cu (-> "a.cu") vs a file literally named a.cu.py (-> "a.cu"): fall back to full names
    seen: dict[str, list[str]] = {}
    for r, n in out.items():
        seen.setdefault(n.lower(), []).append(r)
    for n, members in seen.items():
        if len(members) > 1:
            for r in members:
                out[r] = Path(r).as_posix()
    collisions = {}
    for r, name in out.items():
        collisions.setdefault(name.casefold(), []).append(r)
    for members in collisions.values():
        if len(members) > 1:
            for r in members:
                out[r] += "-" + hashlib.sha256(r.encode()).hexdigest()[:10]
    if len({n.casefold() for n in out.values()}) != len(out):
        raise ProjectError("source names collide after output normalization; rename a source")
    return out


def single_output_name(src: Path, o: str | None) -> tuple[str, str | None]:
    """(name, note) for `compile FILE [-o NAME]`. -o is a NAME: outputs always go to
    <out-dir>/cubin/NAME.cubin and <out-dir>/sass/NAME.sass."""
    if o is None:
        return src.stem, None
    if "/" in o or "\\" in o or os.sep in o:
        raise ProjectError(f"-o takes a name, not a path ({o!r}); every output goes into the "
                           f"output folder -- use --out-dir {str(Path(o).parent)!r} -o {Path(o).name!r}")
    note = None
    for ext in (".cubin", ".sass"):
        if o.lower().endswith(ext) and len(o) > len(ext):
            o, note = o[: -len(ext)], (f"note: -o takes a name without extension; writing "
                                       f"cubin/{o[: -len(ext)]}.cubin and sass/{o[: -len(ext)]}.sass")
            break
    if o in ("", ".", "..") or o.startswith(".") or any(ord(c) < 32 for c in o):
        raise ProjectError(f"-o {o!r} is not a usable output name (pick something like -o mykernel)")
    return o, note


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- output folder ---------------------------------------------------------------

def prepare_out_dir(out_dir: Path) -> Path:
    """Create the output folder (if needed) and check it is writable, with one clear
    error for: an existing file in the way, no permission, read-only file system."""
    out_dir = Path(out_dir)
    if out_dir.exists() and not out_dir.is_dir():
        raise ProjectError(f"output folder {out_dir} exists and is a file; pass --out-dir to choose another")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        fd, probe = tempfile.mkstemp(dir=out_dir, prefix=".oa-write-test-")
        os.close(fd)
        os.unlink(probe)
    except OSError as e:
        raise ProjectError(f"can't write to output folder {out_dir}: {e.strerror or e}") from None
    return out_dir.resolve()


@contextlib.contextmanager
def output_lock(out_dir: Path):
    """Serialize writers to one output tree; independent trees remain independent.

    Keep the lock inode after release: unlinking it would let a new process
    acquire a different inode while an earlier writer still holds its lock.
    """
    import fcntl

    root = prepare_out_dir(out_dir)
    lock_path = root / '.build' / 'writer.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path(root, '.build/writer.lock')
    with lock_path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ProjectError(f'output folder {root} is in use by another build; choose another --out-dir or retry') from None
        try:
            yield root
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


# Temp files of writes in flight; removed on any exception (incl. KeyboardInterrupt).
_PENDING: set[Path] = set()


def write_atomic(path: Path, data: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_p = Path(tmp)
    _PENDING.add(tmp_p)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data.encode() if isinstance(data, str) else data)
        os.replace(tmp_p, path)
    finally:
        _PENDING.discard(tmp_p)
        tmp_p.unlink(missing_ok=True)


def cleanup_pending() -> None:
    for p in list(_PENDING):
        p.unlink(missing_ok=True)
        _PENDING.discard(p)


@contextlib.contextmanager
def defer_sigint():
    """Run a short critical section (renaming a set of finished files into place,
    rewriting the manifest) with Ctrl-C held until it ends, so an interrupt can't
    leave a cubin without its .sass or outputs the manifest doesn't know about."""
    try:
        old = signal.getsignal(signal.SIGINT)
        got: list[int] = []
        signal.signal(signal.SIGINT, lambda *_: got.append(1))
    except ValueError:            # not the main thread: nothing to defer
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old)
        if got:
            raise KeyboardInterrupt


# ---- manifest ------------------------------------------------------------------

class ManifestError(ProjectError):
    pass


def load_manifest(out_dir: Path) -> dict | None:
    """The manifest, None if there is none. Raises ManifestError if it is unreadable
    or not the shape described in this module's docstring."""
    p = Path(out_dir) / MANIFEST
    if not p.exists():
        return None
    try:
        m = json.loads(p.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ManifestError(f"{p} is corrupt ({e}); delete it, or recompile with "
                            "openassembler compile <src>") from None
    if not isinstance(m, dict) or not isinstance(m.get("entries"), list) or \
            not all(isinstance(e, dict) and "source" in e and "cubin" in e and isinstance(e.get("kernels"), list)
                    for e in m["entries"]):
        raise ManifestError(f"{p} is not an openassembler manifest (missing entries/source/cubin/kernels); "
                            "delete it, or recompile with openassembler compile <src>")
    if "applications" in m and (not isinstance(m["applications"], list) or
            not all(isinstance(a, dict) and isinstance(a.get("name"), str)
                    and isinstance(a.get("kernels"), list) for a in m["applications"])):
        raise ManifestError(f"{p}: invalid application records; recompile")
    if m.get("schema") != MANIFEST_SCHEMA or not isinstance(m.get("root"), str):
        raise ManifestError(f"{p}: unsupported manifest schema or missing source root; recompile")
    for e in m["entries"]:
        if not isinstance(e["source"], str) or not all(isinstance(k, dict) and isinstance(k.get("name"), str)
                and isinstance(k.get("params"), list) and all(isinstance(v, list) and len(v) == 2
                and all(isinstance(x, str) for x in v) for v in k["params"])
                and ("source_params" not in k or isinstance(k["source_params"], list)
                     and len(k["source_params"]) == len(k["params"])
                     and all(isinstance(v, str) for v in k["source_params"]))
                and ("backend" not in k or k["backend"] in ("openassembler", "ptxas-fallback"))
                for k in e["kernels"]):
            raise ManifestError(f"{p}: invalid kernel or source record; recompile")
        for rel in entry_outputs(e):
            artifact_path(out_dir, rel)
    for app in m.get("applications", []):
        required = ("source_bundle", "cache", "report", "graphs", "config", "entrypoint", "compiled_by")
        if (not all(isinstance(app.get(k), str) and app[k] for k in required)
                or not isinstance(app.get("args"), list) or not all(isinstance(x, str) for x in app["args"])
                or not isinstance(app.get("runtime"), dict)
                or not isinstance(app.get("source_bundle_sha256"), str)
                or not isinstance(app.get("cache_sha256"), str)
                or not all(isinstance(k, dict) and isinstance(k.get("name"), str)
                           and isinstance(k.get("cubin"), str) and isinstance(k.get("sha256"), str)
                           and ("backend" not in k or k["backend"] == "torch-inductor-triton")
                           and isinstance(k.get("cache_path"), str)
                           and not Path(k["cache_path"]).is_absolute() and ".." not in Path(k["cache_path"]).parts
                           for k in app["kernels"])):
            raise ManifestError(f"{p}: invalid application record; recompile")
        for key in ("source_bundle", "cache", "report", "graphs", "config"):
            if app.get(key) is not None:
                artifact_path(out_dir, app[key])
        for kernel in app["kernels"]:
            for key in ("cubin", "sass", "ptx"):
                if kernel.get(key) is not None:
                    artifact_path(out_dir, kernel[key])
        if (not isinstance(app.get('source'), str) or not app['source']
                or type(app.get('seed')) is not int
                or not isinstance(app.get('source_sha256'), str)
                or not all(isinstance(app['runtime'].get(key), str) and app['runtime'][key]
                           for key in ('torch', 'cuda', 'python', 'device'))
                or app['runtime'].get('capability') != [8, 0]
                or not all(type(app.get(key)) in (int, float) and 0 <= app[key] < float('inf')
                           for key in ('rtol', 'atol', 'state_rtol', 'state_atol'))
                or not isinstance(app.get('requirements', []), list)
                or not all(isinstance(x, str) for x in app.get('requirements', []))):
            raise ManifestError(f'{p}: invalid application runtime metadata; recompile')
    return m


def artifact_path(out_dir: Path, rel: str) -> Path:
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
        raise ManifestError(f"invalid output artifact path: {rel!r}")
    root, target = Path(out_dir).resolve(), (Path(out_dir) / rel).resolve()
    if target == root or not _inside(target, root):
        raise ManifestError(f"output artifact escapes {out_dir}: {rel!r}")
    return target


def entry_source_abs(m: dict, e: dict) -> Path:
    s = Path(e["source"])
    return s if s.is_absolute() else Path(m.get("root") or ".") / s


def entry_outputs(e: dict) -> list[str]:
    return [e[k] for k in ("cubin", "sass", "ptx", "reference") if e.get(k)]


def application_outputs(app: dict) -> list[str]:
    return [app[key] for key in ('source_bundle', 'cache', 'report', 'graphs', 'config') if app.get(key)] + [
        kernel[key] for kernel in app.get('kernels', []) for key in ('cubin', 'sass', 'ptx') if kernel.get(key)]


def rel_source(src: Path, root: Path) -> str:
    try:
        return src.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(src.resolve())


def remove_outputs(out_dir: Path, rels: list[str]) -> list[str]:
    """Delete output files listed in a manifest -- only ever inside the output folder."""
    gone = []
    out_dir = out_dir.resolve()
    for r in rels:
        p = (out_dir / r).resolve()
        if not _inside(p, out_dir) or p == out_dir:
            continue
        if p.is_file():
            p.unlink()
            gone.append(r)
            # drop now-empty mirror directories, never the top-level layout dirs' parent
            d = p.parent
            while d != out_dir and _inside(d, out_dir):
                try:
                    d.rmdir()
                except OSError:
                    break
                d = d.parent
    return gone


def save_manifest(out_dir: Path, root: Path, arch: str, opt_level: str, entries: list[dict],
                  applications: list[dict] | None = None) -> None:
    """entries carry absolute "source" paths in memory; stored relative to root."""
    stored = []
    for e in sorted(entries, key=lambda e: e["cubin"]):
        e = dict(e)
        e["source"] = rel_source(Path(e["source"]), root)
        stored.append(e)
    m = {"openassembler": __version__, "schema": MANIFEST_SCHEMA, "arch": arch, "opt_level": opt_level,
         "root": str(root.resolve()), "entries": stored}
    if applications is None:
        previous = load_manifest(Path(out_dir))
        applications = previous.get("applications", []) if previous else []
    if applications:
        m["applications"] = applications
    write_atomic(Path(out_dir) / MANIFEST, json.dumps(m, indent=1) + "\n")


def write_group(out_dir: Path, files: dict[str, bytes | str]) -> None:
    """Stage a group of outputs before replacing destinations.

    Callers defer SIGINT around publication and the manifest update. Each replacement
    is atomic; a filesystem error midway through the group may still leave a partial set.
    """
    staged = []
    try:
        for rel, data in files.items():
            path = artifact_path(out_dir, rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
            tp = Path(tmp)
            _PENDING.add(tp)
            staged.append((tp, path))
            with os.fdopen(fd, "wb") as f:
                f.write(data.encode() if isinstance(data, str) else data)
        for tp, path in staged:
            os.replace(tp, path)
    finally:
        for tp, _ in staged:
            _PENDING.discard(tp)
            tp.unlink(missing_ok=True)
