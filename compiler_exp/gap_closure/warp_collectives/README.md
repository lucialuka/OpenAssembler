# Full-warp shuffle research cycle

The [canonical dossier](../../../docs/compiler/gaps-collectives.md) defines G15,
the exact masks/clamps/participation domain, existing recovered encodings,
candidate experiments, independent references and remaining limits.

The independent lane-reference/PTX/SASS gate passes 64 fixture/optimization
combinations. The [final A100 report](candidate-a100-20261123T090825Z.json)
passes 1,792 launches, including aliased source/lane/destination operands,
reconvergence, low-five-bit offsets, one/four warps, complete guards/input/tails
and same-PTX ptxas controls. The [audit](promotion-audit.json) checks 3,670,016
bytes, all cubins/source identities and 64 native word/register comparisons.
The earlier 1,344-launch result and prior candidate snapshots remain retained.

The [policy continuity](../foundations_20261121/audit-20261123T091100Z.json)
preserves 49 installed policies and 98 original native sessions. The
[application audit](application-continuity.json) preserves 1,234 kernel
occurrences from earlier atomic/decode/checkpoint artifacts, retaining their
original source-exact numerical proofs and explicitly naming changed inputs.
It is a read-only executable comparison, with no new hardware/timing claim.

The full [supported](supported-suite.log) and
[compiler-only](compiler-only-suite.log) logs follow promotion. The dossier
separates native correctness, source tests and remaining participation work.

```sh
python compiler_exp/gap_closure/warp_collectives/offline.py
python -m pytest -q tests/test_backend_shuffles.py
modal run compiler_exp/gap_closure/warp_collectives/run_modal.py
python compiler_exp/gap_closure/warp_collectives/continuity.py
```

Production imports none of these experiment drivers. Full-mask/full-warp
preconditions remain explicit, and there is no performance or general
collective-synchronization claim.
