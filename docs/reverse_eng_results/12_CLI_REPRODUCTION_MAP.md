# CLI and reproduction map

Commands below use Python 3.12+ from the repository's test environment and the
CUDA toolkit for real compilation. See [COMMANDS.md](../COMMANDS.md) for every
flag and [REPRODUCING.md](../../REPRODUCING.md) for the entry map. A compile or
emission check does not reproduce an archived hardware timing.

## Native compiler smoke

```sh
python3 -m openassembler compile examples/vector_add.py --verify --out-dir output/vector
python3 -m openassembler run output/vector --backend modal --budget 120 --job-timeout 30
```

## GEMM and rowwise contracts

```sh
python3 -m openassembler gemm --m 1024 --n 4096 --k 1024 --dtype f16 --out-dir output/gemm
python3 -m openassembler gemm --m 1024 --n 4096 --k 1024 --dtype f16 \
  --dispatch manual --out-dir output/gemm_reference
python3 -m openassembler gemm --m 128 --n 512 --k 512 --tile 64x64 \
  --copy-address-init reuse --pointer-mode direct --pointer-advance early \
  --out-dir output/gemm_copy_reuse
python3 -m openassembler gemm --m 1024 --n 4096 --k 1024 --dtype bf16 \
  --fragment-schedule load_next --out-dir output/gemm_load_next
python3 -m openassembler gemm --m 16384 --n 4096 --k 4096 \
  --cta-order grouped_n8 --out-dir output/gemm_grouped_n
python3 -m openassembler run output/gemm --backend modal --budget 180 --job-timeout 60
python3 -m openassembler rowwise --op softmax --rows 4096 --width 512 --out-dir output/softmax
python3 -m openassembler rowwise --op softmax --rows 4096 --width 32 --out-dir output/softmax_selected
python3 -m openassembler rowwise --op softmax --rows 4096 --width 32 \
  --dispatch manual --warps-per-cta 4 --out-dir output/softmax_reference
python3 -m openassembler run output/softmax --backend modal --budget 180 --job-timeout 60
```

`--dispatch auto` is the default for generated GEMM/rowwise commands. Current
exact tables route 34 tuned GEMM contracts plus eight measured baseline
retentions, and five rowwise geometry contracts; unmeasured or stale contracts
keep their reference. Explicit implementation flags select manual mode. No
ordinary compile command launches a hidden search.

`load_next` admits two M/N/K shapes, 1024×4096×1024 and 4096×1024×1024, for
both FP16 and BF16 under its other fixed source/resource restrictions.
`grouped_n8` changes the launch grid and requires N tile count divisible by
eight. Neither is selected by the current automatic table. The normal runner
uses exact small-integer or analytic smoke inputs. Broader A100 numerical and
performance validators live under `docs/evidence/a100-mma` and `exp/`.

## Current measured dispatch replay

The completed [November 11 replay](../../exp/dispatch_tuning/RESULTS.md) has
immutable selection, freeze and held-out records. Report the retained audited
decisions and check the current selectors without starting a GPU job:

```sh
.venv/bin/python -m exp.dispatch_tuning.promote \
  exp/dispatch_tuning/runs/a100-production-shapes-copyinit-v5-20261110
.venv/bin/python -m pytest -q tests/test_gemm_dispatch.py tests/test_gemm_promotion.py \
  tests/test_rowwise_dispatch.py exp/dispatch_tuning/test_promote.py
.venv/bin/python tools/check_docs.py
```

Reporting can reuse cached decisions when raw record/protocol hashes match.
For a fresh full archive/native-encoding audit, add
`--table exp/dispatch_tuning/runs/a100-production-shapes-copyinit-v5-20261110/audited-table.json`
to the promotion command. That writes an experiment table rather than changing
the installed one; see the [campaign commands](../../exp/dispatch_tuning/README.md#reproduce).

For new measurements, follow the bounded selection → freeze → independent
held-out sequence in the [campaign README](../../exp/dispatch_tuning/README.md)
under a new campaign name. The GEMM protocol uses a 2% per-round threshold and
16-launch graphs; the separate rowwise protocol uses 3% and 32-operation
graphs. Their thresholds, compiler identities and records are not
interchangeable. `gemm --autotune local|modal` is a separate explicit per-shape
search path, not an automatic table promotion.

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
| Exact GEMM selection/replay | `exp/dispatch_tuning/modal_campaign.py` | `exp/dispatch_tuning/runs/` |
| Rowwise CTA geometry | `exp/dispatch_tuning/rowwise_modal.py` | `exp/dispatch_tuning/rowwise_runs/` |
| Whole application capture | `openassembler cnr ... --frontend torch --backend modal` | application output manifest/report |

## Environment identity

The retained A100 evidence includes SXM4-40GB, SXM4-80GB and 80GB PCIe workers
with their recorded driver/toolkit identities. The completed production GEMM
campaign used CUDA 12.8 and ptxas/nvdisasm controls; rowwise selection/replay
used SXM4-40GB. Compare paired ratios within a worker/session and keep device
cohorts separate. Re-run on another device only with a new record; do not
overwrite a historical result with a different GPU or toolkit.

