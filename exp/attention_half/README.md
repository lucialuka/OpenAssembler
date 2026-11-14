# FP16 batched products and composed attention

Experiment-first extension of the OA Python model bridge, validated on A100.
See [RESULTS.md](RESULTS.md) for admitted paths, measurements and rejected cases.

`candidate.py` contains the experimental batched Tensor Core recipe and small
FP16-rounded scalar alternative. `campaign.py` checks independent CPU rounding,
forward/dX/dW, complete attention SGD, workspace poison/canaries, changed-input
CUDA Graph replay and profiler launch ownership. Production phases remove
experimental patches from the admitted path. Comparison controls are explicitly
labeled. `modal_run.py` pins CUDA 12.8.1, Python 3.12 and Torch 2.7.1/cu128 and
ships local OA source/data. It does not use Triton or Inductor for OA execution.

```sh
modal run exp/attention_half/modal_run.py --phase bmm
modal run exp/attention_half/modal_run.py --phase experiment
modal run exp/attention_half/modal_run.py --phase production-bmm
modal run exp/attention_half/modal_run.py --phase production
modal run exp/attention_half/modal_run.py --phase paired
modal run exp/attention_half/modal_run.py --phase cpu-tests
python3 -m openassembler cnr examples/native_attention_model --backend modal --verify --no-sass
.venv/bin/python -m pytest -q tests/test_model_batched_tensor.py tests/test_diagnostics.py
.venv/bin/python tools/check_docs.py
```

`runs/` holds local source snapshots, source hashes, full bundles, logs and raw
results and is ignored. Durable JSON outcomes and source identities are retained
under `evidence/`. A nonzero worker process is always failure, even if an earlier
child wrote a success result. Failed numerical, dispatch and harness attempts
are retained and distinguished from production validation.
