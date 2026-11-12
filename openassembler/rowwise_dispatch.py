"""Fail-closed exact-shape CTA geometry selection for rowwise FP32 kernels."""
from __future__ import annotations

from dataclasses import replace
import ast
import hashlib
import json
from pathlib import Path

from openassembler.kernels.reductions import ReductionPlan, generate

TABLE = Path(__file__).with_name("kernels") / "sm80_rowwise_dispatch.json"


def identity_inputs() -> dict[str, Path]:
    """Rowwise-reachable code/data files, scoped away from unrelated GEMM work."""
    root = Path(__file__).resolve().parent.parent
    files: dict[str, Path] = {}
    for folder in ("backend", "opt", "ptx", "isa", "container"):
        for path in (root / "openassembler" / folder).rglob("*.py"):
            files[path.relative_to(root).as_posix()] = path
    for relative in (
        "openassembler/kernels/reductions.py", "openassembler/rowwise.py",
        "openassembler/rowwise_dispatch.py", "openassembler/build.py",
        "openassembler/project.py", "openassembler/oracle.py", "openassembler/native_ownership.py",
        "openassembler/gpu/cuda_driver.py",
    ):
        path = root / relative
        files[relative] = path
    for path in (root / "specs" / "sm_80").glob("*.json"):
        files[path.relative_to(root).as_posix()] = path
    for relative in ("corpus/sm_80/pairs.jsonl", "corpus/sm_80/machine_model.json"):
        path = root / relative
        files[relative] = path
    return files


def _runtime_source_sections() -> dict[str, bytes]:
    """Hash only rowwise CLI/runner slices, not unrelated GEMM CLI changes."""
    root = Path(__file__).resolve().parent.parent
    cli_path = root / "openassembler/cli.py"
    cli = cli_path.read_text()
    begin = '    pw = sub.add_parser("rowwise"'
    end = "    pw.set_defaults(func=cmd_rowwise)"
    first, last = cli.index(begin), cli.index(end, cli.index(begin)) + len(end)
    cli_parser = cli[first:last]

    runner_path = root / "openassembler/runner.py"
    runner = runner_path.read_text()
    tree = ast.parse(runner)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "rowwise_case")
    runner_lines = runner.splitlines(keepends=True)
    runner_case = "".join(runner_lines[node.lineno - 1:node.end_lineno])

    cli_tree = ast.parse(cli)
    node = next(n for n in cli_tree.body if isinstance(n, ast.FunctionDef) and n.name == "cmd_rowwise")
    cli_lines = cli.splitlines(keepends=True)
    cli_command = "".join(cli_lines[node.lineno - 1:node.end_lineno])
    return {
        "openassembler/cli.py::<cmd_rowwise>": cli_command.encode(),
        "openassembler/cli.py::<rowwise_parser>" : cli_parser.encode(),
        "openassembler/runner.py::<rowwise_case>": runner_case.encode(),
    }


def compiler_identity() -> str:
    h = hashlib.sha256()
    for name, path in sorted(identity_inputs().items()):
        h.update(name.encode() + b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    for name, source in sorted(_runtime_source_sections().items()):
        h.update(name.encode() + b"\0")
        h.update(source)
        h.update(b"\0")
    return h.hexdigest()


def contract_key(plan: ReductionPlan) -> dict:
    return {"op": plan.op, "rows": plan.rows, "width": plan.width,
            "epsilon_f32": plan.epsilon, "input_dtype": "f32", "output_dtype": "f32",
            "accumulation_dtype": "f32", "arch": plan.arch,
            "layout": "contiguous_row_major", "input_alignment": 4,
            "output_alignment": 4, "tail_policy": "masked_columns_uniform_row_exit"}


def select_plan(plan: ReductionPlan, *, table: dict | None = None) -> tuple[ReductionPlan, dict]:
    baseline = replace(plan, warps_per_cta=4)
    decision = {"schema": 1, "mode": "measured", "policy": "w4",
                "reason": "no exact measured rowwise geometry", "contract": contract_key(baseline),
                "scope": "exact FP32 rowwise kernel contract; no other shape or whole-model inference"}
    if table is None:
        try:
            table = json.loads(TABLE.read_text())
        except (OSError, ValueError):
            return baseline, {**decision, "reason": "rowwise dispatch table unavailable"}
    if not isinstance(table, dict) or table.get("schema") != 1:
        return baseline, {**decision, "reason": "invalid rowwise dispatch table"}
    if table.get("compiler_identity") != compiler_identity():
        return baseline, {**decision, "reason": "rowwise table belongs to a different compiler"}
    entries = table.get("entries")
    if not isinstance(entries, list):
        return baseline, {**decision, "reason": "invalid rowwise dispatch entries"}
    # Validate the whole small table before using any row. A malformed or
    # duplicated entry must not be silently ignored while another row routes.
    seen_contracts = set()
    for row in entries:
        if not isinstance(row, dict) or not isinstance(row.get("contract"), dict):
            return baseline, {**decision, "reason": "invalid rowwise dispatch entry"}
        contract = row["contract"]
        try:
            candidate = ReductionPlan(rows=contract["rows"], width=contract["width"],
                                      op=contract["op"], epsilon=contract["epsilon_f32"],
                                      arch=contract["arch"])
        except (KeyError, TypeError, ValueError):
            return baseline, {**decision, "reason": "invalid rowwise dispatch entry"}
        if contract_key(candidate) != contract:
            return baseline, {**decision, "reason": "invalid rowwise dispatch entry"}
        frozen_contract = json.dumps(contract, sort_keys=True, separators=(",", ":"))
        if frozen_contract in seen_contracts:
            return baseline, {**decision, "reason": "duplicate exact rowwise entry"}
        seen_contracts.add(frozen_contract)
        candidate_warps = row.get("warps_per_cta")
        if (row.get("status") != "promoted" or row.get("gates_passed") is not True
                or type(candidate_warps) is not int or candidate_warps not in (1, 2, 4, 8)
                or hashlib.sha256(generate(replace(candidate, warps_per_cta=candidate_warps)).encode()).hexdigest()
                != row.get("ptx_body_sha256")):
            return baseline, {**decision, "reason": "invalid rowwise dispatch entry"}
    matched = [row for row in entries if isinstance(row, dict) and row.get("contract") == contract_key(baseline)]
    if len(matched) != 1:
        return baseline, {**decision, "reason": "no unique exact rowwise entry"}
    entry = matched[0]
    warps = entry.get("warps_per_cta")
    chosen = replace(baseline, warps_per_cta=warps)
    return chosen, {**decision, "policy": f"w{warps}", "reason": "exact held-out rowwise contract",
                    "evidence": entry.get("evidence", []), "device_scope": entry.get("measurement_cohorts", []),
                    "measured_speedup": entry.get("speedup_range")}
