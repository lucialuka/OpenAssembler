"""Run a reproducible CLI integration matrix and save timing/results as JSON.

The matrix distinguishes local compiler time from GPU execution time. GPU rows use
the same backend accepted by ``run``/``cnr``: ``modal``, ``local`` or ``user@host``.
PyTorch application rows use ``--build-backend`` for capture and the same transport
for replay. The command never invents a success for a missing CUDA toolkit, GPU,
PyTorch runtime or SSH host; each row records ``ok``, ``skipped`` or ``failed``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def run_row(name: str, command: list[str], *, execute: bool) -> dict:
    started = time.perf_counter()
    row = {"name": name, "command": command, "status": "skipped" if not execute else "failed"}
    if execute:
        try:
            proc = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, timeout=1800)
            row["status"] = "ok" if proc.returncode == 0 else "failed"
            row["returncode"] = proc.returncode
            row["output"] = proc.stdout[-12000:]
        except subprocess.TimeoutExpired as exc:
            row.update(status="failed", returncode=None, output=(exc.stdout or "")[-12000:], timeout=True)
    row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="modal", help="GPU backend: modal, local, or user@host")
    ap.add_argument("--build-backend", choices=("modal", "local", "ssh"), default="modal")
    ap.add_argument("--build-host", default=None, metavar="USER@HOST")
    ap.add_argument("--out-dir", type=Path, default=Path("output/cli-benchmark"))
    ap.add_argument("--budget", type=float, default=300)
    ap.add_argument("--execute", action="store_true", help="run commands; default records a dry-run matrix")
    ap.add_argument("--skip-optional", action="store_true", help="skip toolkit/corpus/NN optional rows")
    args = ap.parse_args()
    py = sys.executable
    out = args.out_dir
    rows = []

    def oa(*parts: str) -> list[str]:
        return [py, "-m", "openassembler", *parts]

    rows.append(run_row("version", oa("--version"), execute=args.execute))
    rows.append(run_row("status", oa("status"), execute=args.execute))
    rows.append(run_row("compile_ptx", oa("compile", "examples/div_by_const.ptx", "--verify",
                                           "--out-dir", str(out / "ptx")), execute=args.execute))
    rows.append(run_row("compile_python", oa("compile", "examples/vector_add.py", "--verify",
                                              "--out-dir", str(out / "python")), execute=args.execute))
    rows.append(run_row("compile_cuda", oa("compile", "examples/vector_add.cu", "--verify",
                                            "--out-dir", str(out / "cuda")), execute=args.execute))
    rows.append(run_row("compile_repo", oa("compile", "testcases/test_whole_repo", "--dry-run",
                                           "--out-dir", str(out / "repo")), execute=args.execute))
    rows.append(run_row("gemm_emit", oa("gemm", "--m", "1024", "--n", "1024", "--k", "128",
                                         "--dtype", "bf16", "--emit-only", "--out-dir", str(out / "gemm")),
                        execute=args.execute))

    gpu_args = ["--backend", args.backend, "--budget", str(args.budget), "--dry-run"]
    if args.execute:
        run_args = ["--backend", args.backend, "--budget", str(args.budget), "--spec", "examples/run_spec.json"]
        cnr_args = ["--backend", args.backend, "--budget", str(args.budget), "--spec", "examples/run_spec.json"]
    else:
        run_args = [*gpu_args, "--spec", "examples/run_spec.json"]
        cnr_args = [*gpu_args, "--spec", "examples/run_spec.json"]
    rows.append(run_row("run_vector_add", oa("run", str(out / "python"), *run_args), execute=args.execute))
    rows.append(run_row("cnr_vector_add", oa("cnr", "examples/vector_add.py", *cnr_args,
                                              "--out-dir", str(out / "cnr")), execute=args.execute))
    probe_args = ["--backend", args.backend, "--budget", str(args.budget), "--trials", "2"]
    if not args.execute:
        probe_args.append("--dry-run")
    rows.append(run_row("gpu_probe", oa("gpu", "run", *probe_args), execute=args.execute))

    app_compile = ["compile", "testcases/test_whole_repo", "--verify", "--out-dir", str(out / "application"),
                   "--build-backend", args.build_backend]
    if args.build_host:
        app_compile += ["--build-host", args.build_host]
    rows.append(run_row("transformer_compile", oa(*app_compile), execute=args.execute))
    transformer_run = ["run", str(out / "application"), "--application", "test_whole_repo",
                       "--backend", args.backend, "--budget", str(args.budget)]
    if not args.execute:
        transformer_run.append("--dry-run")
    rows.append(run_row("transformer_run", oa(*transformer_run), execute=args.execute))

    if not args.skip_optional:
        nn_args = ["--backend", args.backend, "--budget", str(max(args.budget, 180)),
                   "--rounds", "3", "--reps", "20"]
        if not args.execute:
            nn_args.append("--dry-run")
        rows.append(run_row("nn_kernel_study", [py, "tools/nn_study.py", *nn_args], execute=args.execute))
        rows.append(run_row("fastmm_dry", oa("fastmm", "run", "--dry-run"), execute=args.execute))

    passed = sum(row["status"] == "ok" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    report = {"schema": 1, "timestamp": stamp, "openassembler": "0.2.0",
              "backend": args.backend, "build_backend": args.build_backend,
              "executed": args.execute, "passed": passed, "failed": failed,
              "rows": rows}
    target = out / "runs" / f"cli-benchmark-{stamp}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"CLI benchmark: {passed} passed, {failed} failed, {len(rows) - passed - failed} skipped")
    print(f"evidence: {target}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
