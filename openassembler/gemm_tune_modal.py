"""Explicit Modal transport for native GEMM tuning; never imported by default."""
from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("openassembler-native-gemm-tune")
image = modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")


@app.function(image=image, gpu="A100-40GB", timeout=1860, max_containers=1, retries=0)
def tune(request: dict, runtime: dict[str, bytes]) -> dict:
    import os
    import subprocess
    import sys
    import tempfile
    with tempfile.TemporaryDirectory(prefix="oa-tuning-runtime-") as directory:
        for name, data in runtime.items():
            path = Path(directory) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (Path(directory) / "env/cuda.sh").chmod(0o700)
        source, result = Path(directory) / "request.json", Path(directory) / "result.json"
        source.write_text(json.dumps(request))
        code = ("import json,sys; from dataclasses import asdict; "
                "from openassembler.gemm_tune import benchmark; "
                "from openassembler.kernels.gemm import GemmPlan; "
                "r=json.load(open(sys.argv[1])); "
                "v=benchmark(GemmPlan(**r['plan']),rounds=r['rounds'],samples=r['samples']); "
                "json.dump(v,open(sys.argv[2],'w'),allow_nan=False)")
        subprocess.run([sys.executable, "-c", code, str(source), str(result)], cwd=directory,
                       env=dict(os.environ, PYTHONPATH=directory), check=True, timeout=1800)
        return json.loads(result.read_text())


@app.local_entrypoint()
def main(request: str, result: str):
    from openassembler.transport import runtime_files
    response = tune.remote(json.loads(Path(request).read_text()), runtime_files(native=True))
    Path(result).write_text(json.dumps(response, allow_nan=False))
