"""Retain deterministic CPU contract checks and unabridged regression diagnostics."""
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
TESTS=['tests/test_model_checkpoints.py','tests/test_model_optimizer_options.py','tests/test_model_operators.py','tests/test_source_diagnostics.py','tests/test_model_batch_broadcast.py',
       'tests/test_model_semantics.py','tests/test_model_graph.py','tests/test_model_frontend.py',
       'tests/test_model_runtime.py','tests/test_model_optimization.py','tests/test_model_tensor.py',
       'tests/test_model_batched_tensor.py','tests/test_model_kernels.py','tests/test_model_training_kernels.py']
if __name__=='__main__':
    import torch
    command=[sys.executable,'-m','pytest','-q',*TESTS]
    done=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    result={'schema':1,'gaps':['G34','G35'],'stage':'CPU graph/reference/encoding/emulator regressions',
            'python':platform.python_version(),'torch':torch.__version__,'command':command,
            'returncode':done.returncode,'stdout':done.stdout,'stderr':done.stderr,
            'passed':done.returncode==0,'hardware':'not run by this gate',
            'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [*(ROOT/'openassembler/modelfront').glob('*.py'),*(ROOT/n for n in TESTS)]}}
    (HERE/'cpu-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(done.stdout,end='');print(done.stderr,end='',file=sys.stderr)
    raise SystemExit(done.returncode)
