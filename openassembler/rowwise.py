"""Compile checked SM80 cooperative FP32 row reductions through the normal CLI.

Generated source carries its exact operation, shape and launch contract. Edited
or unannotated PTX does not receive inferred buffers or launches.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from openassembler import project

MARKER = "// openassembler-rowwise-v1 "


def source_for(plan) -> str:
    from openassembler.kernels.reductions import generate
    contract = asdict(plan)
    # Preserve the established default generated PTX byte-for-byte. Older
    # annotations omit this field and the dataclass constructor supplies W4.
    if plan.warps_per_cta == 4:
        contract.pop("warps_per_cta", None)
    return MARKER + json.dumps(contract, sort_keys=True, separators=(",", ":")) + "\n" + generate(plan)


def contract_from_source(source: str) -> dict | None:
    if not source.startswith(MARKER):
        return None
    from openassembler.kernels.reductions import ReductionPlan, generate
    try:
        header, body = source.split("\n", 1)
        plan = ReductionPlan(**json.loads(header[len(MARKER):]))
        if body != generate(plan):
            return None
    except (ValueError, TypeError, KeyError):
        return None
    return {"schema": 1, **asdict(plan), **plan.metadata, "launch": plan.launch}


def compile_rowwise(plan, out_dir: Path, name: str | None = None, *, no_sass: bool = False,
                    emit_only: bool = False, dispatch: dict | None = None) -> int:
    from openassembler.build import Options, _compile_paths
    name, _ = project.single_output_name(Path(f"{plan.op}.ptx"), name)
    out_dir = Path(out_dir)
    with project.output_lock(out_dir):
        source = project.artifact_path(out_dir, f".build/generated/{name}.ptx")
        project.write_atomic(source, source_for(plan))
        if dispatch is not None:
            project.write_atomic(project.artifact_path(out_dir, "rowwise-dispatch.json"),
                                 json.dumps({**dispatch, "plan": asdict(plan)}, indent=2) + "\n")
            print(f"dispatch: {dispatch.get('mode', 'manual')} / {dispatch.get('policy', 'w4')} "
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
