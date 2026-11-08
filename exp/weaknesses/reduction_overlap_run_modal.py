"""Source-inclusive A100 validation for streamed reduction candidates.

This experiment builds both the ordinary rowwise PTX and the streamed
load/FMA candidate through the OpenAssembler encoder inside a Modal image that
contains the repository's specs/corpus. It then uses the existing validated
operator worker for full outputs, guards, unchanged inputs, graph replay and
paired OA/ptxas timing.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import modal


HERE = Path(__file__).resolve().parent
ROOT = Path.cwd()
IMAGE = (modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
         .add_local_dir(str(ROOT / "openassembler"), "/root/repo/openassembler")
         .add_local_dir(str(ROOT / "specs"), "/root/repo/specs")
         .add_local_dir(str(ROOT / "corpus"), "/root/repo/corpus")
         .add_local_dir(str(ROOT / "env"), "/root/repo/env")
         .add_local_dir(str(ROOT / "tools"), "/root/repo/tools",
                        ignore=["**/__pycache__", "**/*.pyc"])
         .add_local_dir(str(ROOT / "exp" / "nn_round2"), "/root/repo/exp/nn_round2",
                        ignore=["**/__pycache__", "**/*.pyc"])
         .add_local_dir(str(ROOT / "exp" / "weaknesses"), "/root/repo/exp/weaknesses",
                        ignore=["**/__pycache__", "**/*.pyc"]))
APP = modal.App("openassembler-reduction-overlap-a100-run")


def _cases() -> list[dict]:
    return [
        {"op": op, "width": width, "warps": warps, "rows": 4096}
        for op in ("rmsnorm_backward", "softmax_backward")
        for width in (512, 513, 1024)
        for warps in (4, 8)
    ]


@APP.function(image=IMAGE, gpu="A100-40GB", timeout=1800, max_containers=1, retries=0)
def execute(cases: list[dict], samples: int = 9, graph_repeats: int = 16) -> dict:
    import shutil
    import subprocess
    import sys
    import tempfile

    sys.path.insert(0, "/root/repo")
    sys.path.insert(0, "/root/repo/tools")
    os.environ.pop("OPENASSEMBLER_CUDA_HOME", None)
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")

    import openassembler.oracle as oracle
    oracle._available.clear()
    oracle.available = lambda tool="ptxas": shutil.which(tool) is not None
    oracle.run_tool = lambda *args, check=True: subprocess.run(
        list(args), cwd="/root/repo", capture_output=True, text=True, check=check
    )

    from openassembler.cli import main as cli_main
    from openassembler.container.cubin import regcount
    from exp.nn_round2.operator_suite import rowwise_source
    from exp.nn_round2 import operator_worker
    from exp.weaknesses.reduction_overlap_codegen import OverlapPlan, generate

    def compile_one(source: str, kernel: str, root: str) -> tuple[bytes, bytes, dict]:
        ptx = os.path.join(root, kernel + ".ptx")
        out = os.path.join(root, "out")
        os.makedirs(out, exist_ok=True)
        with open(ptx, "w") as f:
            f.write(source)
        rc = cli_main(["compile", ptx, "-o", kernel, "--out-dir", out])
        if rc:
            raise RuntimeError({"compile_rc": rc, "kernel": kernel})
        manifest = json.load(open(os.path.join(out, "manifest.json")))
        entry = manifest["entries"][0]
        return (open(os.path.join(out, entry["cubin"]), "rb").read(),
                open(os.path.join(out, entry["reference"]), "rb").read(), manifest)

    def static(cubin: bytes, kernel: str, root: str) -> dict:
        path = os.path.join(root, kernel + ".cubin")
        open(path, "wb").write(cubin)
        proc = subprocess.run(["nvdisasm", "-c", path], capture_output=True, text=True, check=True)
        lines = [line for line in proc.stdout.splitlines() if ";" in line and "/*" in line]
        return {"instructions": len(lines), "non_nop": sum(" NOP" not in line for line in lines),
                "registers": regcount(cubin, kernel),
                "sass_sha256": hashlib.sha256(proc.stdout.encode()).hexdigest(),
                "hmma": proc.stdout.count("HMMA"), "shfl": proc.stdout.count("SHFL"),
                "warpsync": proc.stdout.count("WARPSYNC")}

    results = []
    for spec in cases:
        op, width, warps, rows = spec["op"], spec["width"], spec["warps"], spec["rows"]
        baseline_kernel = f"op_{op}_w{width}_c{warps}_baseline"
        overlap_kernel = f"op_{op}_w{width}_c{warps}_overlap"
        original_name, original = rowwise_source(op, width, warps)
        baseline_body = original.replace(".entry " + original_name + "(", ".entry " + baseline_kernel + "(", 1)
        overlap_body, metadata = generate(OverlapPlan(op, width, warps))
        overlap_body = overlap_body.replace(".entry " + original_name + "_overlap(", ".entry " + overlap_kernel + "(", 1)
        header = ".version 8.7\n.target sm_80\n.address_size 64\n"
        with tempfile.TemporaryDirectory() as td:
            base_cubin, base_ptxas, _ = compile_one(header + baseline_body, baseline_kernel, os.path.join(td, "baseline"))
            over_cubin, over_ptxas, _ = compile_one(header + overlap_body, overlap_kernel, os.path.join(td, "overlap"))
            rows_out = []
            for label, kernel, oa, ptxas, body in (
                ("baseline", baseline_kernel, base_cubin, base_ptxas, baseline_body),
                ("overlap", overlap_kernel, over_cubin, over_ptxas, overlap_body),
            ):
                payload = {
                    "schema": 1,
                    "oa": base64.b64encode(oa).decode(),
                    "ptxas": base64.b64encode(ptxas).decode(),
                    "oa_sha256": hashlib.sha256(oa).hexdigest(),
                    "ptxas_sha256": hashlib.sha256(ptxas).hexdigest(),
                    "source_sha256": hashlib.sha256(body.encode()).hexdigest(),
                    "source_identities": {},
                    "kernels": [{"kernel": kernel, "op": op, "mode": f"{label}_warp{warps}",
                                 "candidate": label, "kind": "rowwise", "width": width,
                                 "warps": warps, "inputs": 2, "outputs": 1, "dtype": "float",
                                 "rtol": 2e-4, "atol": 5e-5}],
                }
                worker = operator_worker.run(payload, samples=samples, graph_repeats=graph_repeats,
                                             rows=rows, seed=int(f"{width}{warps}{rows}"))
                rows_out.append({"label": label, "static_oa": static(oa, kernel, td),
                                 "static_ptxas": static(ptxas, kernel, td),
                                 "worker": worker})
            results.append({"case": spec, "metadata": metadata, "variants": rows_out})
    return {"device": "A100-40GB", "cases": results,
            "methodology": {"full_output": True, "canaries": True,
                            "inputs_unchanged": True, "graph_poison": True,
                            "bitwise_oa_ptxas": True, "samples": samples,
                            "graph_repeats": graph_repeats}}


@APP.local_entrypoint()
def main(samples: int = 9, graph_repeats: int = 16, subset: str = ""):
    cases = _cases()
    if subset:
        cases = [case for case in cases if subset in f"{case['op']}_{case['width']}_{case['warps']}"]
    result = execute.remote(cases, samples, graph_repeats)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = HERE / f"reduction_overlap_run_{stamp}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(out)

