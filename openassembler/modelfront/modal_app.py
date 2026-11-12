"""Identical native model worker for Modal and the local/SSH entrypoint."""
from __future__ import annotations

import json
import math
from pathlib import Path
import modal

app = modal.App('openassembler-native-model')
image = modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04', add_python='3.12').pip_install(
    'torch==2.7.1', index_url='https://download.pytorch.org/whl/cu128')


@app.function(image=image, gpu='A100-40GB', timeout=1860, max_containers=1, retries=0)
def process(request_text: str, compiler: dict[str, bytes]) -> str:
    import os
    import subprocess
    import sys
    import tempfile
    with tempfile.TemporaryDirectory(prefix='oa-native-model-') as d:
        root = Path(d)
        for name, data in compiler.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (root / 'env/cuda.sh').chmod(0o700)
        req, res = root / 'request.json', root / 'result.json'
        req.write_text(request_text)
        subprocess.run([sys.executable, '-m', 'openassembler.modelfront.worker',
                        '--request', str(req), '--result', str(res)], cwd=d,
                       env=dict(os.environ, PYTHONPATH=d), check=True)
        return res.read_text()


@app.local_entrypoint()
def main(request: str, result: str, gpu: str = 'A100-40GB', budget: float = 900):
    from openassembler.transport import runtime_files
    response = process.with_options(gpu=gpu, timeout=math.ceil(budget) + 60).remote(
        Path(request).read_text(), runtime_files(native=True))
    Path(result).write_text(response)
