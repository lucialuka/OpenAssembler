# Current-production A100 shape tuning

Updated November 11, 2026. This workspace has separate GEMM and rowwise
protocols; their thresholds, graphs and compiler identities are not
interchangeable.

This campaign revalidates the native policies that earlier studies found useful
on particular GEMM shapes. It uses the production `gemm_tune` implementation and
the ordinary annotated CLI compilation path; it does not rewrite experimental
PTX into an untracked kernel or infer performance from instruction counts.

The matrix has 42 exact FP16/BF16 contracts: 32 QKV/output/FFN contexts from the
role study, the earlier tall and square controls, the load-next tall shape, and
two shapes that require the 64×64 tile. Role labels supply context only. Every
kernel still computes unbatched contiguous `A[M,K] @ B[N,K].T`, FP32 output and
accumulation, alpha one/beta zero, and zero workspace. There is no claim that
these shapes cover every transformer, stride, tail, precision or library mode.

## Numerical, binary and timing gates

For each admissible native policy the production tuner:

- Compiles the checked PTX through the public compiler and retains PTX, native
  cubin/SASS and same-source ptxas cubin/SASS.
- Checks complete zero/integer/normal outputs, finite output, canaries, immutable
  inputs, same-source and baseline ptxas bits, and poisoned graph replay.
- Checks disassembly, register/local-memory resources, HMMA8 and drain15.
- Runs three paired rounds of at least eight samples with sixteen-launch CUDA
  graphs and shuffled native candidate order. A candidate needs at least 2%
  baseline improvement in every round; uncertain gains retain the baseline.

The wrapper then uses fresh buffers for a separate paired selected-OA,
same-source-ptxas and cuBLAS GemmEx comparison. This control phase has its own
full output/canary/input/bitwise checks and poisoned graphs. cuBLAS uses the
matched input/accumulation/output precision, alpha one/beta zero and disabled
reduced-precision reduction; it is not a tuned cuBLASLt search. Kernel timings
exclude compiler startup, transfers, packing, framework substitution and whole
training. Counter replay is outside this timing protocol.

## Independent selection and replay

Each case runs in a fresh subprocess. The first Modal application gathers the
selection data. Before the second application starts, `freeze.py` audits each
successful selection, freezes its exact winner/incumbent/source contract and
raw-record hash, and records the timestamp. Held-out promotion checks this
frozen choice, not a new choice selected after seeing replay results.

All ratios stay inside their worker/session. Distinct A100 variants are
retained as separate measured cohorts; their absolute times are never pooled.
A passing finite campaign establishes an exact measured policy, not a universal
fastest implementation or an unmeasured-shape guarantee.

## Durable evidence and recovery

Every case writes its JSON, source/binary archive, stdout and stderr into the
`openassembler-dispatch-evidence-20261110` Modal Volume before returning. The
local client immediately verifies/downloads each completed archive and result
under `runs/<campaign>/<session>/<M>x<N>x<K>_<dtype>/`. A transport interruption
can be resumed with the same command: complete local cases are reused, and a
completed remote checkpoint is recovered without repeating its GPU work.
Failures get explicit `failure.json` files and cannot become winners.

Session records retain actual Modal app/container/function-call identifiers,
compiler identity, generated source hashes, runtime/device information, raw
samples, recorded gate values and artifact hashes. The archive is checked by
the production record loader and the independent promotion auditor.

## Completed current campaign

The [November 11 results](RESULTS.md) cover all 42 exact contracts in selection
and independent frozen held-out sessions. The current table installs 34 tuned
policies and eight measured baseline retentions. Each split retains 500
candidate builds and 1,500 complete candidate/input checks. Same-source ptxas
is faster on every nominal-selected shape; these entries improve OA policy,
not the CUDA toolchain. Supplemental library controls use the session nominal
selection and must not be attributed to a different frozen policy.

The frozen campaign is `a100-production-shapes-copyinit-v5-20261110`. Its
per-entry decisions and retained PTX/cubin/SASS bundles, rather than the older
role-study or interrupted-run labels, authorize the installed table. Copy-address
reuse is selected in 20 exact contracts; Morton8 is selected in the two
16384×4096×4096 FP16/BF16 contracts. Combined, grouped-N and load-next candidates
remain available but are not selected by this frozen campaign. A measured
baseline retention means no frozen alternative cleared the admission policy;
it does not prove that a candidate can never win on that shape in another
controlled experiment.

`promote.py` reuses the production promotion audit and saves each exact decision
durably. A reporting invocation may reuse saved decisions when record/protocol
hashes agree; it does not recheck every archive in that cached path. Installation
with `--table` bypasses saved decisions, re-audits every source/binary archive
and current native encoding, and writes the table only after the whole frozen
campaign is complete.

## Reproduce

Use the authenticated Modal CLI from the repository root. Ordinary compilation
does not start this campaign implicitly.

```sh
.venv/bin/python -m pytest -q exp/dispatch_tuning/test_campaign.py
modal run exp/dispatch_tuning/modal_campaign.py \
  --campaign NEW-CAMPAIGN --session selection --start 0 --limit 4 --samples 8
.venv/bin/python -m exp.dispatch_tuning.freeze \
  exp/dispatch_tuning/runs/NEW-CAMPAIGN
modal run exp/dispatch_tuning/modal_campaign.py \
  --campaign NEW-CAMPAIGN --session heldout --start 0 --limit 4 --samples 8
```

Advance `--start` in bounded batches until all 42 selection cases complete,
freeze once, then run all 42 held-out cases. Use `--detach` when transport is
unreliable; completed cases are recovered from the evidence volume. Aggregate
the audited decisions with:

```sh
.venv/bin/python -m exp.dispatch_tuning.promote \
  exp/dispatch_tuning/runs/NEW-CAMPAIGN
```

To force the complete binary/source audit without replacing the installed
table, direct its resulting table to an experiment path:

```sh
.venv/bin/python -m exp.dispatch_tuning.promote \
  exp/dispatch_tuning/runs/NEW-CAMPAIGN \
  --table exp/dispatch_tuning/runs/NEW-CAMPAIGN/audited-table.json
```

Only a reviewed successful result should use
`--table openassembler/kernels/sm80_gemm_dispatch.json`. These commands perform
CPU audits, not new GPU measurements.

Use a new campaign name after changing source or the campaign protocol. Never
rewrite a raw selection or freeze file to make a later compiler appear current.
Only a successful independent promotion decision may update the production
exact-contract table.

## Separate rowwise CTA-geometry campaign

The rowwise study completed 58 operator/shape contracts (29 shapes for each of
FP32 softmax and non-affine population LayerNorm), testing 1/2/4/8 whole warps
per CTA. Its [frozen shortlist](rowwise_runs/rowwise_selection_20261110/frozen-heldout.json)
contains seven candidates that were at least 3% faster than W4 in every
selection round. A separate seven-case replay admitted only five: softmax
129×1023/W2, softmax 4096×1/W8 and 4096×32/W8, and LayerNorm 4096×1/W8 and
4096×32/W8. LayerNorm 129×1023/W2 and 4096×1024/W1 passed the numerical and
memory-safety checks but missed the frozen 3% heldout cutoff and remain W4.

This protocol uses 32-operation resident-input graphs, three rounds and eight
samples per round. Five input families, all output elements, finite values,
input immutability, canaries, poisoned graph replay, disassembly and resources
are checked. Candidate OA outputs must be bitwise identical to W4 OA;
same-source ptxas has a full numerical control. It is not the GEMM protocol's
2%/16-launch policy. Both rowwise sessions ran on A100-SXM4-40GB. The current
generator's PTX and native words were independently matched to both retained
sessions before installing the new launch-contract field.

The checked selector is in
[`rowwise_dispatch.py`](../../openassembler/rowwise_dispatch.py) and the
installed table is
[`sm80_rowwise_dispatch.json`](../../openassembler/kernels/sm80_rowwise_dispatch.json).
It matches exact operation, rows, width, epsilon, layout, dtype/alignment and
SM80 identity; unknown or stale entries use W4. See the
[canonical rowwise dossier](../../docs/research/ideas/shape-aware-rowwise-dispatch.md)
for all limitations and the integration/source continuity proof.

For a fresh measurement, use new session names and freeze winners before
running the independent shortlist. The original shortlist indices are shown
only to replay that fixed seven-case set:

```sh
modal run exp/dispatch_tuning/rowwise_modal.py \
  --session rowwise_selection_NEW --rounds 3 --samples 8
modal run exp/dispatch_tuning/rowwise_modal.py \
  --session rowwise_heldout_NEW --rounds 3 --samples 8 \
  --indices 17,23,24,46,52,53,57
.venv/bin/python -m pytest -q tests/test_rowwise_dispatch.py \
  tests/test_reduction_codegen.py tests/test_cli_rowwise.py
```

Rowwise checkpoints use the separate
`openassembler-rowwise-geometry-evidence` Modal volume. Each shape's JSON and
artifact archive is saved before its remote call returns. Existing frozen
records are retained unchanged; a fresh study must not inherit their admission
or compiler identity merely because its filenames match.
