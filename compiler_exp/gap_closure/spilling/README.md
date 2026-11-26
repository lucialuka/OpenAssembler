# G17 bounded scalar spill cycle

The [canonical dossier](../../../docs/compiler/gaps-spilling.md) defines the
gap, private-slot and partial-definition rules, exact register/frame bounds,
independent experiments, production path and remaining grouped/predicate work.

The [baseline](baseline-pressure.json) records register-limit overflow and
actual exhaustion. The isolated candidate's independent loop-state reference
passes all four levels and normal/poisoned SASS. The first guarded-temporary
design failed conservative backedge liveness; its implementation and partial
failure log remain retained. It was stopped during the largest case and is not
a completed passing run. The corrected partial-definition lowering reloads the
old value before the guarded update, then preserves it on false predicates.

The final [expanded hardware](candidate-a100-20261123T095515Z.json) and
[tight-limit hardware](tight-a100-20261123T095932Z.json) pass 320 and 64 launches.
They cover 16/32/64/252 declared registers, automatic exhaustion recovery,
guarded loops and coexistence with user local objects. All complete state,
input, guard, tail, driver resource and frame checks pass. Separate
[full](promotion-audit.json) and [tight](promotion-tight-audit.json) audits
recheck 73,334,784 bytes and 24 exact native word/register/resource pairs.
The early 256-launch report uses the pre-automatic corpus; final admission uses
only the two source-exact audited gates above. During the early run the local
driver file changed before its final harness hash was computed, so that record
is not used as a source-exact promotion proof. Its uploaded compiler hashes,
archived PTX/output and raw result remain retained.

The [resource experiment](resources-20261123T100759Z.json) measures one matched
synthetic kernel, with thirteen randomized resident CUDA-event samples for each
OA/ptxas × explicit/automatic variant. Exact state passes first and after each
sample. The [audit](resource-audit.json) checks outputs/sample order/medians and
executable/source identities. The 64-register OA kernel is slower than its
252-register control in this domain. No automatic limit policy, occupancy
measurement or whole-application speedup is admitted.

The [policy audit](../foundations_20261121/audit-20261123T100450Z.json) preserves
49 policies/98 original sessions. The [application audit](application-continuity.json)
preserves 1,342 earlier native kernel occurrences and all original numerical
proofs. Full [supported](supported-suite.log) and
[compiler-only](compiler-only-suite.log) suites, docs and source/evidence hashes
are bound in the [validation record](validation.json).

```sh
python -m pytest -q tests/test_backend_spilling.py
python compiler_exp/gap_closure/spilling/offline.py
modal run compiler_exp/gap_closure/spilling/run_modal.py
modal run compiler_exp/gap_closure/spilling/run_tight_modal.py
modal run compiler_exp/gap_closure/spilling/run_resources_modal.py
python compiler_exp/gap_closure/spilling/continuity.py
```

Correctness drivers stage exact retained candidate modules, matching promoted
source. Production imports none of these experiment drivers. Wide groups,
predicates, slot reuse and general pressure/scheduling/cost policy remain open.
