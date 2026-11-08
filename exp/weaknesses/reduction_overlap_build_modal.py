"""Modal compile/disassembly gate for streamed reduction templates.

This is a build-only experiment. It does not execute a GPU kernel; the later
worker must still check outputs, canaries, timing and Nsight counters.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal


HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
IMAGE = (modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
         .add_local_python_source("openassembler", copy=False)
         .add_local_dir(str(EXP_ROOT), remote_path="/root/exp"))
APP = modal.App("openassembler-reduction-overlap-build")


@APP.function(image=IMAGE, timeout=600, max_containers=1, retries=0)
def build_remote():
    import hashlib
    import subprocess
    import tempfile

    sys.path.insert(0, "/root")
    from exp.weaknesses.reduction_overlap_codegen import OverlapPlan, generate

    rows = []
    for plan in (OverlapPlan("rmsnorm_backward", 513, 4),
                 OverlapPlan("softmax_backward", 513, 4)):
        source_body, metadata = generate(plan)
        source = ".version 8.7\n.target sm_80\n.address_size 64\n" + source_body
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ptx = root / (metadata["kernel"] + ".ptx")
            ptx.write_text(source)
            ptxas = root / "ptxas.cubin"
            proc = subprocess.run(["ptxas", "-arch=sm_80", "-O3", "-o", str(ptxas), str(ptx)],
                                  capture_output=True, text=True)
            if proc.returncode:
                raise RuntimeError({"ptxas": proc.stderr})
            dis = subprocess.run(["nvdisasm", "-c", str(ptxas)], capture_output=True, text=True)
            if dis.returncode:
                raise RuntimeError({"nvdisasm": dis.stderr})
            rows.append({**metadata,
                         "ptxas_sass_sha256": hashlib.sha256(dis.stdout.encode()).hexdigest(),
                         "ptxas_hmma": dis.stdout.count("HMMA"),
                         "ptxas_shfl": dis.stdout.count("SHFL")})
    return {"device_target": "sm_80", "cases": rows,
            "status": "ptxas_compiled_and_disassembled", "openassembler": "not_run_in_modal_build",
            "execution": "not_run"}


@APP.local_entrypoint()
def main():
    print(json.dumps(build_remote.remote(), indent=2))

