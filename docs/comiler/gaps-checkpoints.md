# Complete owned-state native application checkpoints

Status: implemented

The [November 23 integration](../../compiler_exp/gap_closure/integration_20261123/README.md)
reruns the complete gate after wide-value backend repairs. The
[fresh full-state report](../../compiler_exp/gap_closure/checkpoints/hardware-20261123T082914Z.json)
passes 7,862 comparisons / 124,438 elements, nested optimizer and CLI gates,
and the [fresh IEEE report](../../compiler_exp/gap_closure/checkpoints/maximum-20261123T082454Z.json)
passes 80 comparisons / 1,360 elements. Both workers identify A100-SXM4-80GB.
The [current audit](../../compiler_exp/gap_closure/integration_20261123/checkpoint-final-audit.json)
checks all four backend hashes, unchanged model sources, 28 bundles, 383 native
comparisons and 17 exact checkpoint byte pairs. The earlier records below keep
their original source/device identities.

G35 identified that optimizer-only snapshots required callers to save weights
separately and omitted application progress. The native runtime and
`CompiledModel` now save a versioned, self-contained checkpoint for their exact
supported program: every captured model parameter and immutable FP32 buffer,
all optimizer state and step, CPU RNG state, and explicit host application data.
It uses JSON and raw tensor bytes; loading never unpickles or executes payloads.

## Ownership and compatibility

`Runtime.save_checkpoint(path, application={...})` and
`Runtime.load_checkpoint(path, device=..., restore_rng=True)` operate at a
completed optimizer-update boundary or after inference was explicitly discarded.
Outstanding backward tapes, pending gradients, replay-owned runtimes and wrong
CUDA streams reject. `CompiledModel.save_native_checkpoint` and
`load_native_checkpoint` provide the same contract with registered
`Parameter`/buffer ownership. Existing conventional autograd must finish and
prepared CUDA Graph replay must be closed first.

A checkpoint contains the exact native plan/source/cubin identities, all
captured model state, canonical optimizer policy, step and complete
`momentum`/`m`/`v`/`max_v` inventory. Restore requires matching program identities
and policy; it does not silently remap names, change precision or compile a new
program. A new runtime or CPU-loaded `CompiledModel` can initialize directly
from the checkpoint on an A100. The saved model bundle remains the executable
program dependency; the checkpoint is a complete state artifact, not a second
copy of the executable compiler/toolchain.

Raw little-endian FP32 bytes preserve signed zeros, infinities and NaN payloads
from live state. Legacy optimizer-only loading still requires finite snapshots;
the complete checkpoint validates nonnegative variance and maximum invariants
while preserving IEEE nonfinite state. This is state custody, not a claim that
all operators implement arbitrary NaN/Inf forward semantics. Maximize and
AMSGrad's native maximum have independent nonfinite state controls linked in
the optimizer dossier.

`application` must contain finite JSON values, for example:

```python
application = {
    "cursor": 12,
    "data_cursor": {"epoch": 1, "offset": 5},
    "scheduler": {"kind": "declared_step_table", "next_rate": 0.001},
    "scaler": None,
}
model.save_native_checkpoint("step-12.oa-checkpoint", application=application)
restored = model.load_native_checkpoint("step-12.oa-checkpoint", device="cuda")
next_step = restored["application"]["cursor"]
```

Python `random` and Torch CPU RNG state are included by default and restored
only after complete validation and device staging. The RNG runtime versions
must match. `include_rng=False` requires explicit `restore_rng=False` on load.
The current native program has no GPU random operator/scaler execution contract;
CUDA/NumPy/custom generator state, dataloader workers, file handles and arbitrary
host effects are not implicitly serialized. Scheduler/scaler/cursor data is
returned to the caller to consume; no scheduler or scaler behavior is invented.
The caller must serialize access to its native runtime and process-global CPU
RNG while saving/restoring; concurrent host RNG mutation is outside this boundary.

## Atomicity and stale bindings

Saving validates state first, writes a temporary file with a deterministic
manifest and raw tensor entries, flushes it, and atomically replaces the target.
A failed publication preserves the previous checkpoint. Restoring validates
format/version, exact program, complete inventories, canonical tensor paths,
shapes, dtypes, byte counts and SHA-256 before device allocation. Duplicate ZIP
members, duplicate JSON keys, unaccounted entries, path substitutions and
compressed data reject. The manifest is bounded to 1 MiB; raw tensor sizes are
derived from the admitted program, with no arbitrary executable object graph.

The loader allocates and copies a complete new set of guarded model/state
bindings, synchronizes staging, then commits. Allocation/transfer failure leaves
existing weights, moments, step, RNG and ownership lists unchanged. A CPU RNG
commit failure restores the previous RNG before changing bindings. Returned
application data is independent JSON, not borrowed mutable checkpoint storage.

Successful restoration replaces model/state bindings. Callers must fetch current
`runtime.weights` or module parameters again; old external tensor references
remain valid as their own old storage, and prior exposed runtime outputs are
invalidated. Superseded checkpoint and moment backings are retired, so repeated
loads do not grow the runtime's allocation/guard inventory indefinitely.
Ordinary `nn.Module.load_state_dict(assign=True)` replacements are refreshed
before a subsequent checkpoint save, even without another native call.

## Validation and retained failures

The [independent CPU candidate](../../compiler_exp/gap_closure/checkpoints/candidate.py)
compares uninterrupted and interrupted/resumed 24-update trajectories for SGD
momentum/Nesterov, Adam AMSGrad/maximize and AdamW AMSGrad. Complete model
parameter/buffer, optimizer, CPU RNG and application JSON bytes match exactly.
Its [record](../../compiler_exp/gap_closure/checkpoints/candidate-results.json)
uses Python 3.12.10, Torch 2.7.1 and seed 35052.

The [regressions](../../tests/test_model_checkpoints.py) include malformed
identity/inventory/hash/path/dtype/shape/RNG controls, IEEE payloads, deterministic
archive bytes, failed atomic publication, pending tape/gradient/replay boundary
rejections, partial RNG commit rollback, module binding replacement and repeated
ownership retirement. The latest [CPU gate](../../compiler_exp/gap_closure/checkpoints/cpu-results.json)
passes 566 model/optimizer/checkpoint regressions.

The first independent review reproduced stale public module bindings in a
checkpoint save: `load_state_dict(assign=True)` replaced the module state while
the runtime cache still referenced old tensors. The
[counterexample](../../compiler_exp/gap_closure/checkpoints/reviewer-stale-bindings.json)
is retained; save now refreshes validated canonical bindings and has CPU/GPU
positive controls. Review also identified old checkpoint allocation retention;
retiring the exact replaced backing subset fixes repeated loads while preserving
borrowed tensor references. These are host ownership fixes, with no change to
optimizer coefficients or tolerances.

The initial [30-update A100 gate](../../compiler_exp/gap_closure/checkpoints/hardware-20261121T134407Z.json)
passed on an actual A100-SXM4-80GB. A later
[rejected run](../../compiler_exp/gap_closure/checkpoints/hardware-20261121T135836Z.json)
retains the exact profiler/launch-ledger mismatch on the dedicated IEEE control:
initial forward activities were missing from the trace. The
[harness correction](../../compiler_exp/gap_closure/checkpoints/maximum-harness-review.json)
profiles the synchronized targeted optimizer-update region and keeps full-state
comparisons separate. Its [dedicated A100-SXM4-40GB record](../../compiler_exp/gap_closure/checkpoints/maximum-20261121T140047Z.json)
passes 80 all-state NaN/Inf classification and raw-bit restore checks. These
records identify different devices and are not pooled as timing evidence.

The [final reviewed full gate](../../compiler_exp/gap_closure/checkpoints/hardware-20261121T140736Z.json)
passes **7,862 full checks / 124,438 elements** on an actual
A100-SXM4-40GB, Python 3.12.1, Torch 2.7.1+cu126, CUDA runtime 12.6 and toolkit
12.8.1. Twelve cases cover all four view/reuse policies with three optimizer
families, 30 updates and resume after update 12. Public module resume, 17 exact
complete checkpoint byte pairs, 15 rollback/boundary controls, stable allocation
counts across repeated loads and registered-state replacement pass. The same
worker reruns the extended optimizer gate (2,612 comparisons / 336,948 elements)
and all four ordinary CLI optimizer families with six complete updates each.
The standalone IEEE classification gate remains a separate process: a second
attempt to append its profiler to the long worker again failed the exact
[ownership gate](../../compiler_exp/gap_closure/checkpoints/hardware-20261121T140349Z.json).
That failure is retained; its passing standalone result has identical final
production host/kernel hashes, checked explicitly by the audit.

The [final archive](../../compiler_exp/gap_closure/checkpoints/hardware-20261121T140736Z.zip)
has SHA-256 `60b8f46734a801bc5259ade56dcac4cde538838ab2855a0bba53e92f94ebb683`.
The [source/archive custody audit](../../compiler_exp/gap_closure/checkpoints/audit.json)
checks 28 native bundles, 383 current-versus-retained executable word/register
comparisons and all eleven changed production source identities against both
CPU and hardware records. All original successful and failed records remain
unchanged. No checkpoint speed or whole-workload speedup is claimed.

```sh
python compiler_exp/gap_closure/checkpoints/candidate.py
python compiler_exp/gap_closure/checkpoints/cpu_gate.py
modal run compiler_exp/gap_closure/checkpoints/run_modal.py
modal run compiler_exp/gap_closure/checkpoints/run_maximum_modal.py
python compiler_exp/gap_closure/checkpoints/audit.py
```

This closes the missing complete state artifact for the stated owned native
program. It does not claim generic Python-process checkpointing or unsupported
RNG/mutation/distributed/AMP/serving state; those remain separate application
contracts under G22–G24, G33, G38–G41 and G44.
