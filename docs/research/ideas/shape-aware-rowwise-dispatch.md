# Shape-aware CTA geometry for cooperative rowwise kernels

Status: implemented

OpenAssembler's native FP32 rowwise softmax and population LayerNorm now use an
exact, measured CTA warp count for five A100 shape/operator contracts. Each warp
still owns one row and keeps the established lane reduction, arithmetic and
synchronization order. The change only groups more or fewer independent rows
inside a CTA. The default remains four warps for every unmeasured or stale
contract.

This is a win against OpenAssembler's own four-warp kernel on a small heldout
set. It is not evidence that OpenAssembler beats `ptxas`, cuDNN, or an end-to-end
transformer stack.

## Contract and implementation

The production kernel accepts contiguous row-major FP32 input/output buffers,
with four-byte alignment, a positive row count and width 1–1024. It implements
softmax or non-affine population LayerNorm; LayerNorm epsilon is part of the
exact key. Inputs and intermediate FP32 arithmetic must remain finite. Width
and row tails are supported. One full warp reduces each row with the existing
butterfly shuffle sequence; there is no shared-memory staging.

`--dispatch auto` checks the complete supported contract: operation, rows,
width, FP32 input/output/accumulation, epsilon, SM80, contiguous layout and
alignment. It also checks a rowwise-scoped compiler identity and the exact
emitted PTX body hash. No nearest-shape or inferred fallback exists. If a table,
identity, entry or body check fails, the plan uses four warps. `--dispatch
manual` keeps four warps unless `--warps-per-cta` explicitly selects 1, 2, 4 or
8. The checked runner records that plan and validates its launch grid and thread
count before running.

The exact production entries are:

| Operation | Rows × width | Warps per CTA | Heldout speedup over W4, three paired rounds |
|---|---:|---:|---:|
| Softmax | 129 × 1023 | 2 | 1.036×, 1.034×, 1.039× |
| Softmax | 4096 × 1 | 8 | 1.112×, 1.113×, 1.123× |
| Softmax | 4096 × 32 | 8 | 1.111×, 1.116×, 1.126× |
| LayerNorm | 4096 × 1 | 8 | 1.106×, 1.116×, 1.106× |
| LayerNorm | 4096 × 32 | 8 | 1.125×, 1.120×, 1.111× |

These results show that very narrow rows benefit from amortizing CTA scheduling
across more independent rows. The 129×1023 softmax case instead preferred two
warps. The experiment does not support generalizing those choices to nearby
widths or row counts.

## Measurement and correctness gates

The selection session covered 58 contracts: two operators, 29 shape points,
1/2/4/8 warps per CTA, three paired rounds and eight graph samples per round.
The frozen selection required a speedup of at least 3% in every selection
round. Seven candidates qualified and were replayed in a separate A100 session
with the same four-warp baseline, same-source `ptxas` controls, and the same
per-round 3% cutoff. Five passed. LayerNorm 129×1023/W2 reached only about
1.030× in one heldout round, and LayerNorm 4096×1024/W1 reached about
1.017×; neither is promoted despite passing numerical and memory-safety checks.

Both sessions used an A100-SXM4-40GB (SM80). Timings are resident-input CUDA
graph replays of 32 operations. Compilation, transfers, and training are
excluded. The records preserve device and session information; measurements
from other A100 variants must be compared within their own cohorts.

Every warp candidate was checked against FP64 formulas using five input
families: normal, zero, wide, large offset with small variance, and alternating
values. Full outputs, finite results, input immutability and guard regions were
checked. Poisoned graph outputs were replayed and checked. OpenAssembler
candidates also had to be bitwise identical to their four-warp OpenAssembler
control. Same-source `ptxas` candidates were retained and passed their full
numeric controls; ptxas bitwise equality is not required when its compilation
changes floating-point evaluation order. Disassembly and register/local-memory
resource gates were required before timing.

The selection/heldout records were made before the production launch parameter
was integrated. They retain historical compiler identity
`16199462600c00a60e675593b1d901a43b33c0f69f7c6f7957fabc3cd20d393c`. The
current production identity is independently computed from rowwise-reachable
compiler/data and the rowwise CLI and runner source slices. For every promoted
entry, current generated PTX is byte-identical to both archived selection and
heldout PTX. Re-encoding that PTX through the current native SM80 backend
matches the retained cubin word-for-word and register-for-register. The table
retains both sets of identities and the source, archive, cubin, SASS and harness
hashes; it does not relabel an old cubin as a newly built one.

## Reproduction

The ordinary compile path needs the CUDA toolkit; hardware experiments need an
authenticated Modal client and an A100. These commands use fresh session names
when re-running measurements:

```sh
modal run exp/dispatch_tuning/rowwise_modal.py \
  --session rowwise_selection_replay --rounds 3 --samples 8
modal run exp/dispatch_tuning/rowwise_modal.py \
  --session rowwise_heldout_replay --rounds 3 --samples 8 \
  --indices 17,23,24,46,52,53,57
```

The selection runner stores one record and a PTX/cubin/SASS archive per shape.
Freeze the exact winner before starting an independent heldout session; do not
choose a new winner after observing heldout timings. The checked-in frozen
selection and both completed cohorts are under
[`exp/dispatch_tuning/rowwise_runs`](../../../exp/dispatch_tuning/rowwise_runs/).

The offline dispatch, CLI, generator and launch-contract regressions run with:

```sh
.venv/bin/python -m pytest -q tests/test_rowwise_dispatch.py \
  tests/test_reduction_codegen.py tests/test_cli_rowwise.py
```

Try the selected narrow-row case and an unmeasured control:

```sh
openassembler rowwise --op softmax --rows 4096 --width 32 --emit-only --out-dir out_selected
openassembler rowwise --op softmax --rows 4096 --width 513 --emit-only --out-dir out_default
```

The first writes the exact heldout eight-warp plan. The second remains on the
four-warp default; no table entry is extrapolated. To force a policy for
diagnostics, pass `--dispatch manual --warps-per-cta 2`.

## Source and evidence

- Generator: [`reductions.py`](../../../openassembler/kernels/reductions.py).
- Exact selector and identity: [`rowwise_dispatch.py`](../../../openassembler/rowwise_dispatch.py).
- Production table: [`sm80_rowwise_dispatch.json`](../../../openassembler/kernels/sm80_rowwise_dispatch.json).
- CLI/checked annotation: [`rowwise.py`](../../../openassembler/rowwise.py); runner: [`runner.py`](../../../openassembler/runner.py).
- Selection: [`frozen-heldout.json`](../../../exp/dispatch_tuning/rowwise_runs/rowwise_selection_20261110/frozen-heldout.json) and per-case records/archives in the same folder.
- Heldout: per-case records and archives in [`rowwise_heldout_20261110`](../../../exp/dispatch_tuning/rowwise_runs/rowwise_heldout_20261110/).
- Test coverage: [`test_rowwise_dispatch.py`](../../../tests/test_rowwise_dispatch.py), [`test_cli_rowwise.py`](../../../tests/test_cli_rowwise.py), and [`test_reduction_codegen.py`](../../../tests/test_reduction_codegen.py).

The exact table is intentionally small. New shapes, widths, strides, dtypes,
operators, A100 cohorts and compiler identities require their own measured
selection and independent replay before automatic routing.
