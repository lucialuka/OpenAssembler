"""Static repository inspection and frontend selection.

Repository inspection deliberately stays ahead of execution: it reads source
text and configuration, discovers native kernel candidates, records build-system
hints, and never imports or runs user code. An explicit [model] config selects native
model capture; other repositories use static kernel discovery.
Native model capture executes a bounded CPU workload and specializes exact shapes.
Triton and Inductor GPU compilation are disabled.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import ast
import tomllib

from openassembler import project


APPLICATION_FRONTENDS = {"model", "torch"}
BUILD_MARKERS = {
    "pyproject.toml": "python",
    "setup.py": "python",
    "setup.cfg": "python",
    "CMakeLists.txt": "cmake",
    "Makefile": "make",
    "meson.build": "meson",
    "Cargo.toml": "cargo",
}


@dataclass(frozen=True)
class RepositoryPlan:
    root: Path
    files: tuple[Path, ...]
    native: tuple[project.Classified, ...]
    build_systems: tuple[str, ...]
    application_frontend: str | None
    application_config: Path | None
    used_git: bool
    dependencies: dict[str, tuple[str, ...]]

    @property
    def has_native(self) -> bool:
        return bool(self.native)

    @property
    def has_application(self) -> bool:
        return self.application_config is not None

    def summary(self) -> str:
        native = ", ".join(f"{c.rel}:{c.language}" for c in self.native[:8])
        if len(self.native) > 8:
            native += f", +{len(self.native) - 8} more"
        systems = ", ".join(self.build_systems) or "none"
        dep_count = sum(len(v) for v in self.dependencies.values())
        return (f"native={len(self.native)} [{native or 'none'}]; "
                f"application={self.application_frontend or 'none'}; build_systems={systems}; "
                f"dependencies={dep_count}; discovery={'git' if self.used_git else 'filesystem'}")


def _application_config(root: Path) -> tuple[str | None, Path | None]:
    path = root / "openassembler.toml"
    if not path.is_file():
        return None, None
    try:
        obj = tomllib.loads(path.read_text())
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        # The Torch frontend reports the detailed configuration error when it is
        # explicitly selected. Native discovery should remain available even if
        # an unrelated application config is malformed.
        return None, path
    if isinstance(obj.get('model'), dict):
        return 'openassembler', path
    app = obj.get("application")
    frontend = app.get("frontend") if isinstance(app, dict) else None
    return frontend if isinstance(frontend, str) else None, path


def _dependencies(path: Path) -> tuple[str, ...]:
    """Extract import/include names without resolving or executing them."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ()
    if path.suffix.lower() == ".py":
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return ()
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
        return tuple(sorted(set(names)))
    names = []
    for match in re.finditer(r"#\s*include\s*[<\"]([^>\"]+)[>\"]|\.include\s+\"([^\"]+)\"", text):
        names.append(match.group(1) or match.group(2))
    return tuple(sorted(set(names)))


def inspect(root: Path, out_dir: Path | None = None) -> RepositoryPlan:
    root = Path(root).resolve()
    if not root.is_dir():
        raise project.ProjectError(f"repository path is not a directory: {root}")
    files, used_git = project.walk(root, (out_dir or root / "output").resolve())
    classes = tuple(project.classify(path, path.relative_to(root).as_posix()) for path in files)
    native = tuple(c for c in classes if c.language is not None)
    systems = tuple(sorted({kind for path in files if path.parent == root
                            for marker, kind in BUILD_MARKERS.items()
                            if path.name == marker}))
    frontend, config = _application_config(root)
    dependencies = {}
    for path in files:
        deps = _dependencies(path)
        if deps:
            dependencies[path.relative_to(root).as_posix()] = deps
    return RepositoryPlan(root, tuple(files), native, systems, frontend, config, used_git, dependencies)


def native_candidates(path: Path, out_dir: Path | None = None) -> tuple[project.Classified, ...]:
    path = Path(path).resolve()
    if path.is_file():
        classified = project.classify(path, path.name)
        return (classified,) if classified.language else ()
    return inspect(path, out_dir).native


def select_frontend(path: Path, out_dir: Path, requested: str = "auto") -> tuple[str, Path | None, RepositoryPlan | None]:
    """Return ``(native|model, model_root, plan)``.

    ``auto`` selects a configured [model] factory, otherwise static native
    kernel discovery. ``native`` explicitly selects standalone kernels even
    in model repositories. ``torch`` is a compatibility alias for ``model``;
    Triton and Inductor frontends reject.
    """
    requested = (requested or "auto").lower()
    if requested in {'triton', 'inductor'}:
        raise project.ProjectError('Triton/Inductor GPU compilation is disabled. Use --frontend model '
                                   'with an OA [model] entrypoint; unsupported GPU operations reject.')
    if requested not in {"auto", "native", *APPLICATION_FRONTENDS}:
        raise project.ProjectError(f"unknown frontend {requested!r}; use auto, native, model, or torch")
    path = Path(path).resolve()
    if not path.is_dir():
        if requested in APPLICATION_FRONTENDS:
            raise project.ProjectError("application frontend requires a repository directory with openassembler.toml")
        return "native", None, None
    plan = inspect(path, out_dir)
    if requested in APPLICATION_FRONTENDS:
        if plan.application_config is None:
            raise project.ProjectError(f"{path} has no openassembler.toml application configuration; "
                                       f"use --frontend native for discovered kernels")
        return "model", path, plan
    if requested == 'auto' and plan.application_frontend == 'openassembler':
        return 'model', path, plan
    # Native is the default even when an application config is present.
    return "native", None, plan


def native_error(plan: RepositoryPlan) -> project.ProjectError:
    if plan.has_application:
        return project.ProjectError(
            f"no native kernels discovered in {plan.root}: {plan.summary()}. "
            "Host programs, build systems and PyTorch graphs are not implicitly "
            "lowered by OpenAssembler. Use --frontend model with a [model] factory "
            "for native model compilation, or add supported .ptx/.cu/.cpp "
            "kernels and Python @kernel functions.")
    return project.ProjectError(
        f"nothing to compile in {plan.root}: {plan.summary()}. "
        "Supported native inputs are PTX .entry kernels, CUDA/C++ __global__ "
        "kernels, and Python functions decorated with @kernel.")
