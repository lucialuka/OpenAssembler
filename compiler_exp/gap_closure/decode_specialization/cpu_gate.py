"""Run staged host specialization and cache custody regressions before promotion."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import openassembler.modelfront.decode as module

if __name__=='__main__':
    import pytest
    import torch
    candidate=HERE/'candidate-decode.py'
    exec(compile(candidate.read_text(),str(candidate),'exec'),module.__dict__)
    rc=pytest.main(['-q','tests/test_decode_specialization.py','tests/test_model_semantics.py'])
    result=dict(schema=1,passed=rc==0,python=sys.version,torch=torch.__version__,candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                test_sha256=hashlib.sha256((ROOT/'tests/test_decode_specialization.py').read_bytes()).hexdigest(),
                hardware='not_run',scope='CPU reference control doubles and existing model capture regressions; no native ownership or GPU execution claim')
    (HERE/'candidate-cpu-results.json').write_text(json.dumps(result,indent=2)+'\n')
    raise SystemExit(rc)
