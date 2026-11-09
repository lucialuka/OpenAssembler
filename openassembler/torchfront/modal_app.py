"""Modal builder/runner for the pinned PyTorch CUDA application runtime."""
from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App('openassembler-pytorch-app')
image = modal.Image.from_registry('pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel')
image = image.add_local_dir(str(Path(__file__).resolve().parents[1]), '/root/openassembler',
                            copy=True, ignore=['**/__pycache__', '**/*.pyc'])
gpu_function = app.function(image=image, gpu='A100-40GB', timeout=960,
                            max_containers=1, retries=0)


@gpu_function
def process(request_text: str) -> str:
    import sys
    sys.path.insert(0, '/root')
    from openassembler.torchfront.worker import process as run_request
    response = run_request(json.loads(request_text))
    return json.dumps(response)


@app.local_entrypoint()
def main(request: str, result: str, gpu: str = 'A100-40GB', budget: float = 900) -> None:
    from pathlib import Path
    import math
    request_text = Path(request).read_text()
    fn = process.with_options(gpu=gpu, timeout=math.ceil(budget) + 60)
    try:
        response = fn.remote(request_text)
    except modal.exception.FunctionTimeoutError:
        raise SystemExit('PyTorch capture/run exceeded the Modal hard timeout; see budget in application config.')
    Path(result).write_text(response)
