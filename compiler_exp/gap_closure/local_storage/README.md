# G13 native static local storage

The [canonical dossier](../../../docs/compiler/gaps-storage.md) defines the
original gap, exact local-frame/address/initialization contract, oracle layout
observations, candidate, independent validation and remaining limits.

The [final A100 report](candidate-a100-20261123T093837Z.json) passes 704 launches
on A100-SXM4-40GB, including scalar bit/f32/i32/u32 transfers, static symbols and
32-bit pointers, predicates, loops, changed inputs, CTA-private state and
48/64/80/65,536-byte frames. Actual driver local resources match OA metadata.
Same-PTX ptxas uses its own optimized layouts. Full guards/tails/input/output
checks pass; no timing result is claimed.

The [audit](promotion-audit.json) rechecks 2,883,584 retained bytes, complete
source/binary/manifest/driver frame custody and 44 native word/register pairs.
The earlier 512-launch gate, baseline rejections, toolkit sweeps, a larger
harness's incorrect zero-donor-frame assertion, and the initial ledger fixture's
unsupported vector initialization remain retained. They are not relabeled as
final passing evidence. The final scalar fixture/harness corrections preserve
production scope and numerical gates.

The [policy audit](../foundations_20261121/audit-20261123T094137Z.json) preserves
all 49 measured policies and 98 original sessions. The separate
[application audit](application-continuity.json) preserves 1,298 prior native
kernel occurrences with unchanged original numerical proofs. Production now
imports its maintained layout module; it never imports these experiment drivers.

The [supported suite](supported-suite.log), [compiler-only suite](compiler-only-suite.log)
and [validation record](validation.json) bind final source checks and environment
versions. A focused invocation accidentally named a nonexistent `test_build.py`;
its collection log remains retained, and the corrected actual CLI/local/ownership
selection passes 88 tests. The final full suites use normal repository collection.

```sh
python compiler_exp/gap_closure/local_storage/oracle.py
python compiler_exp/gap_closure/local_storage/layout_sweep.py
python compiler_exp/gap_closure/local_storage/offline.py
python -m pytest -q tests/test_backend_local_storage.py
modal run compiler_exp/gap_closure/local_storage/run_modal.py
python compiler_exp/gap_closure/local_storage/continuity.py
```

G13 remains partial. Generic addressing, wider/vector local transfers, device
calls, spilling, module data placement and relocation/linking need separate
research and hardware gates.
