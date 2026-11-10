# Current-production A100 shape tuning

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

## Reproduce

Use the authenticated Modal CLI from the repository root. Ordinary compilation
does not start this campaign implicitly.

```sh
.venv/bin/python -m pytest -q exp/dispatch_tuning/test_campaign.py
modal run exp/dispatch_tuning/modal_campaign.py \
  --campaign a100-production-shapes-20261110 --session selection --samples 8
.venv/bin/python -m exp.dispatch_tuning.freeze \
  exp/dispatch_tuning/runs/a100-production-shapes-20261110
modal run exp/dispatch_tuning/modal_campaign.py \
  --campaign a100-production-shapes-20261110 --session heldout --samples 8
```

Use a new campaign name after changing source or the campaign protocol. Never
rewrite a raw selection or freeze file to make a later compiler appear current.
Only a successful independent promotion decision may update the production
exact-contract table.
