# Post-allocation pointer-copy forwarding

Status: tested-not-selected

The current backend already offers the equivalent direct-pointer stream. The
post-allocation rewrite adds no distinct code path or performance benefit and
is not an additional production pass.

The experimental transformation is functionally valid on its tested cases,
but produces exactly the already-implemented direct-pointer stream. It adds
no distinct optimization and is not a second additive speedup.

## Question and experiment

Direct pointer homes remove copies early, before physical register allocation.
Could a later transformation remove the same copies while retaining the old
allocation, improving performance or explaining why the tall case regresses?

The experiment runs normal allocation first, finds an adjacent MOV/MOV/LDGSTS
pattern, and forwards an existing aligned source pair directly to the transfer.
It updates the represented register reads and reruns dependency scheduling.
It does not delete a dependency simply because its copies disappeared.

All 18 FP16/BF16 shape cases produce **byte-identical complete native streams**
to direct-home mode, including scheduling controls and padding. The additional
162 zero/integer/normal method/output/canary checks pass. MLP products again
improve, square is neutral and the tall shape regresses, because the candidate
is the same executable stream, not an independently faster implementation.

## What the negative control establishes

For these workloads, the difference between legacy and direct mode is not a
different physical allocation induced by early pointer selection. Removing
copies and the resulting issue/dependency pace still change execution. The
control rules out one explanation for the tall regression; it does not prove
the exact cause of cache and scoreboard changes. Counter evidence belongs to
the [direct-pointer dossier](direct-pointer-homes.md).

## Reproduce

Use Python 3.12+, CUDA 12.8 tools and an authenticated Modal client, from the
repository root. Prepare the validated controls first:

```sh
PYTHONPATH=. modal run exp/weaknesses/gemm_pointer_homes.py --mode production --rounds 2
PYTHONPATH=. modal run exp/weaknesses/gemm_pointer_forwarding.py
.venv/bin/python -m pytest -q tests/test_tensor_pointer_homes.py
```

The second command applies the late experimental forwarding path, not a
production compiler pass. Default `mode=run` is the tested transformation;
`--mode production` would use the already-promoted direct selection instead.
The driver refuses a changed frozen legacy stream. Historical baseline caches
or reconstruction at the recorded commit may be needed; fresh binaries are
not evidence of old provenance merely because a folder has the old name.

## Source and evidence

- [Experimental forwarding and reader updates](../../../exp/weaknesses/gemm_pointer_forwarding.py).
- [All 18 stream identities and hardware results](../../../exp/weaknesses/gemm_pointer_forwarding-20261106T131732.426231Z.json).
- [Frozen controls](../../../exp/weaknesses/gemm_pointer_homes_static.json),
  [production direct selector](../../../openassembler/backend/tensor.py).

There is no user-facing forwarding flag to enable. A future late optimization
would need a genuinely different justified transformation, semantic/resource
checks and repeatable matched hardware benefit before integration.
