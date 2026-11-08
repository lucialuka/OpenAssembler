# CLI and reproduction map

## Native compiler smoke

```sh
python3 -m openassembler compile examples/vector_add.py --verify --out-dir output/vector
python3 -m openassembler run output/vector --backend modal --budget 120 --job-timeout 30
```

## GEMM and rowwise contracts

```sh
python3 -m openassembler gemm --m 1024 --n 4096 --k 1024 --dtype f16 --out-dir output/gemm
python3 -m openassembler run output/gemm --backend modal --budget 180 --job-timeout 60
python3 -m openassembler rowwise --op softmax --rows 4096 --width 512 --out-dir output/softmax
python3 -m openassembler run output/softmax --backend modal --budget 180 --job-timeout 60
```

These runner contracts use exact small-integer or analytic smoke inputs. The
broader normal-input A100 validators live under `docs/evidence/a100-mma` and
`exp/`.

## Reverse-engineering probes

```sh
python3 -m pytest -q tests/test_word.py tests/test_backend.py
PYTHONPATH=. python3 tools/lat_sweep.py --arch sm_80
python3 -m pytest -q tests/test_backend_tensor.py tests/test_gemm_codegen.py
```

The latency generator creates new probe artifacts; it does not reproduce an
archived GPU result without a separate Modal or A100 replay.

## Current experiment runners

| Question | Driver | Result location |
|---|---|---|
| Scalar control minima | `tools/lat_sweep.py`, `tools/gpu_modal.py` | `docs/evidence/a100-latency/` |
| HMMA dependency classes | `exp/tensor_scheduler/clocked_loop_probes.py` | `exp/tensor_scheduler/` |
| CUTLASS fragment schedule | `exp/tensor_scheduler/load_next_compute_current_modal.py` | `exp/tensor_scheduler/` |
| Grouped-N/Morton/m-fast | `exp/gemm_dispatch/grouped_n_modal.py` | `exp/gemm_dispatch/` |
| Reduction counters | `exp/weaknesses/reduction_overlap_ncu_modal.py` | `exp/weaknesses/` |
| Whole application capture | `openassembler cnr ... --frontend torch --backend modal` | application output manifest/report |

## Environment identity

The current A100 evidence uses NVIDIA A100-SXM4-40GB, CUDA 12.8.x, ptxas and
nvdisasm from the pinned toolkit, and the recorded driver version in each raw
JSON. Re-run on another device only with a new record; do not overwrite a
historical result with a different GPU or toolkit.

