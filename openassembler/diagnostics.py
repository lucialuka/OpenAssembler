"""Static source coverage and local build failure reports."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import platform

from . import __version__, project
from .repository import inspect


def inspect_source(source: Path, out_dir: Path | None = None) -> dict:
    source = Path(source).resolve()
    if source.is_dir():
        plan = inspect(source, out_dir)
        classes = [
            project.classify(p, p.relative_to(source).as_posix()) for p in plan.files
        ]
        configured = plan.application_frontend == "openassembler"
        systems = list(plan.build_systems)
    elif source.is_file():
        classes = [project.classify(source, source.name)]
        configured, systems = False, []
    else:
        raise project.ProjectError(f"source does not exist: {source}")
    candidates = [
        {"path": c.rel, "language": c.language, "python_kernels": c.py_kernels}
        for c in classes
        if c.language
    ]
    skipped = [
        {"path": c.rel, "reason": c.skip_reason} for c in classes if not c.language
    ]
    return {
        "schema": 1,
        "openassembler": __version__,
        "source": str(source),
        "frontend": "model" if configured else "native",
        "build_systems": systems,
        "kernel_candidates": candidates,
        "other_files": skipped,
        "counts": dict(Counter(c.language or "other" for c in classes)),
        "ready_for_capture_or_compilation": configured or bool(candidates),
        "contract": "Static discovery only; compilation and numerical verification determine support. "
        "Host programs and arbitrary build systems are not automatically lowered. "
        "GPU execution uses OA-generated cubins; vendor fallback is disabled.",
        "next_step": (
            "compile --verify"
            if configured or candidates
            else "Add an openassembler.toml [model] factory, Python @kernel, CUDA __global__, or PTX .entry."
        ),
    }


def save_failure(args, error: str, rc: int = 1) -> Path | None:
    """Collect coverage and errors without copying source files or environment values."""
    if args.cmd not in ("compile", "cnr") or getattr(args, "dry_run", False):
        return None
    if getattr(args, "frontend", None) in ("triton", "inductor"):
        return None  # Disabled execution routes reject before creating a build root.
    from .workflow import output_dir

    source = Path(args.src)
    out = output_dir(source, args.out_dir)
    errors = getattr(args, "_build_errors", [])
    try:
        coverage = inspect_source(source, out)
    except (OSError, ValueError, project.ProjectError) as exc:
        coverage = {"source": str(source), "inspection_error": str(exc)}
    report = {
        "schema": 1,
        "status": "failed",
        "returncode": rc,
        "command": args.cmd,
        "error": error[-16000:],
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "coverage": coverage,
        "failed_sources": getattr(args, "_failed_sources", []),
        "build_errors": errors,
        "next_step": "Review the error and coverage, then attach this report and a minimal reproducer "
        "to an OpenAssembler bug report. No issue has been submitted.",
    }
    path = out / "compile-error.json"
    project.write_atomic(path, json.dumps(report, indent=2) + "\n")
    return path
