"""Compile shape-specialized tensor GEMMs through the normal CLI build path.

The source embeds a checked launch contract. Recompiling the generated PTX with
``compile`` retains that contract only when its body exactly matches our emitter.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from openassembler import project

MARKER = "// openassembler-gemm-v1 "


def source_for(plan) -> str:
    from openassembler.kernels.gemm import generate
    return MARKER + json.dumps(asdict(plan), sort_keys=True, separators=(",", ":")) + "\n" + generate(plan)


def contract_from_source(source: str) -> dict | None:
    """Only assign inferred buffers/launches to an exact, validated generated body."""
    if not source.startswith(MARKER):
        return None
    from openassembler.kernels.gemm import GemmPlan, generate
    try:
        header, body = source.split("\n", 1)
        plan = GemmPlan(**json.loads(header[len(MARKER):]))
        if body != generate(plan):
            return None
    except (ValueError, TypeError, KeyError):
        return None
    return {"schema": 1, **asdict(plan), "launch": plan.launch,
            "input_layout": "A[M,K] and B[N,K], contiguous in K; C=A@B.T",
            "output_dtype": "f32", "accumulation": "f32"}


def compile_gemm(plan, out_dir: Path, name: str = "gemm", *, no_sass: bool = False,
                 emit_only: bool = False, dispatch: dict | None = None) -> int:
    from openassembler.build import Options, _compile_paths
    # Apply exactly the same filename rules as the normal compile command.
    name, _ = project.single_output_name(Path("gemm.ptx"), name)
    out_dir = Path(out_dir)
    with project.output_lock(out_dir):
        source = project.artifact_path(out_dir, f'.build/generated/{name}.ptx')
        project.write_atomic(source, source_for(plan))
        if dispatch is not None:
            decision_path = project.artifact_path(out_dir, "gemm-dispatch.json")
            project.write_atomic(decision_path,
                                 json.dumps({**dispatch, "plan": asdict(plan)}, indent=2) + "\n")
            policy = dispatch.get("policy", "explicit")
            print(f"dispatch: {dispatch.get('mode', 'manual')} / {policy} "
                  f"({dispatch.get('reason', '')})")
        if emit_only:
            print(source)
            return 0
        result = _compile_paths(source, out_dir, Options(arch=plan.arch, opt_level="opt", no_sass=no_sass), name)
    if result.rc == 0:
        print(f"launch: grid {plan.launch['grid']}, block {plan.launch['threads']}, "
              f"dynamic shared memory {plan.launch['smem']} bytes")
        print(f"run: openassembler run {out_dir}")
    return result.rc
