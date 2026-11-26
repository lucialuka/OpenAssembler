# Full-warp indexed, upward and downward shuffles

Status: implemented

## Gap and exact participation contract

G15 previously admitted only one full-mask butterfly shuffle. The native scalar
route now adds `shfl.sync.idx.b32`, `shfl.sync.down.b32` and
`shfl.sync.up.b32`. The instruction must be unpredicated, have one scalar result,
use literal member mask `0xffffffff`, and use clamp `31` for indexed/downward
or `0` for upward. The existing butterfly form and lowering stay intact.
Literal, parameter-backed and per-lane register source-lane/offset operands
are admitted; only their low five bits participate. Bits move unchanged.

All 32 lanes of each participating warp must execute the collective, and every
sourced lane must participate. Divergent paths may rejoin before the collective;
the lowering emits the existing recovered full-mask `WARPSYNC` sequence before
exchange. Partial masks/warps, predicated collectives, optional predicate
destinations and arbitrary clamp/segment masks remain rejected or outside the
caller contract. Successful scalar compilation does not prove arbitrary caller
participation or dynamically validate launch block dimensions.

The [NVIDIA PTX contract](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync)
defines the lane/segment/clamp and participation rules. The
[retained section](../../compiler_exp/gap_closure/warp_collectives/ptx-shuffle-reference.txt)
and [source identity](../../compiler_exp/gap_closure/warp_collectives/reference-source.json)
record the exact reference used. For the admitted unsegmented domain, indexed
reads select `b & 31`, downward reads select lane plus that offset, and upward
reads select lane minus that offset. An out-of-range source returns the calling
lane's original value.

## Solution and experiments

The [candidate](../../compiler_exp/gap_closure/warp_collectives/candidate.py)
uses existing recovered indexed/downward templates. Upward shuffle computes a
source lane, selects the calling lane when subtraction would leave the warp,
then uses indexed shuffle. It composes ordinary recovered S2R, bit-mask,
subtraction, comparison and select instructions. No encoder field, timing
constant or generated template was invented. The original butterfly selector
block is retained verbatim to preserve existing measured kernels.

The [offline gate](../../compiler_exp/gap_closure/warp_collectives/candidate-offline.json)
passes 64 fixture/optimization combinations: four modes × four operand domains
× four compiler levels. Independent integer lane arithmetic matches the PTX
interpreter and normal/poisoned SASS for offsets `0,1,17,31,32,33,0xffffffff`.
Fixtures reconverge after lane-dependent arithmetic and include a destination
that aliases both the source value and lane operand. This is a bounded semantic
simulation gate, not hardware synchronization evidence.

The [final A100 report](../../compiler_exp/gap_closure/warp_collectives/candidate-a100-20261123T090825Z.json)
passes **1,792 launches**: all 16 fixtures × four OA levels × OA/same-PTX ptxas
controls × 32/128-thread blocks × seven offset values. Every output byte,
unchanged input, allocation guard and inactive output tail passes. The
[first 1,344-launch gate](../../compiler_exp/gap_closure/warp_collectives/candidate-a100-20261123T090656Z.json)
predates the additional aliased-operand fixtures and remains a separate record.
The final worker identifies A100-SXM4-40GB, Python 3.12.1 and CUDA toolkit 12.8.1.

The [promotion audit](../../compiler_exp/gap_closure/warp_collectives/promotion-audit.json)
rechecks **3,670,016 retained bytes**, binary/source identities, corrupted output
rejection and 64 current-versus-archived native word/register comparisons before
and after exact candidate promotion. Production
[selection](../../openassembler/backend/ra.py),
[admission](../../openassembler/backend/contracts.py) and
[regressions](../../tests/test_backend_shuffles.py) import no experiment driver.
The focused shuffle/backend/loop selection passes 146 tests, including rejection
controls for masks, clamps, predicates, widths and unknown modes.

The first full suites retain one stale rejection: an earlier test still required
indexed shuffle to fail. The independent execution controls already cover that
new capability. Replacing the rejection with a partial-mask indexed shuffle
retains the unsupported boundary; the focused shuffle/reduction check passes
122 tests. Both initial failed full logs remain beside final reruns, rather than
being relabeled passing integration results.

The [545-form ledger](../../compiler_exp/gap_closure/warp_collectives/ledger-shuffles-20261123.json)
retains representative parse-through-encoding results. Its independent semantics,
toolkit, packaging, hardware and performance stages remain `not-run`; the
hardware corpus above has separate source identities. The
[application continuity audit](../../compiler_exp/gap_closure/warp_collectives/application-continuity.json)
records 1,234 exact native comparisons against earlier atomic, decode and
checkpoint hardware artifacts without rewriting their numerical/source proofs.
The new [policy audit](../../compiler_exp/gap_closure/foundations_20261121/audit-20261123T091100Z.json)
preserves all 49 measured policies across 98 retained sessions.

```sh
python compiler_exp/gap_closure/warp_collectives/offline.py
python -m pytest -q tests/test_backend_shuffles.py
modal run compiler_exp/gap_closure/warp_collectives/run_modal.py
python compiler_exp/gap_closure/warp_collectives/audit.py compiler_exp/gap_closure/warp_collectives/candidate-a100-20261123T090825Z.json /tmp/oa-shuffle-audit.json
```

G15 remains partial: votes/ballots, match/redux, partial participation, additional
barrier forms and general cooperative visibility still need independent
contracts and hardware evidence. This work does not expose new Python shuffle
intrinsics, change a measured policy or claim a shuffle/application speedup.

## CTA phase reconvergence

The [typed shared-array cycle](gaps-python-types.md#wide-shared-arrays-and-native-cta-reconvergence)
found a hardware-only conditional-writer failure that the functional models'
implicit reconvergence could not expose. Draining scoreboards and changing
shared layout failed. Predicating the writers or explicitly inserting BSSY/BSYNC
passed complete hardware outputs. Original failed and isolating runs retain
source/binary/harness identities and a 75,497,472-byte audit.

The ordinary scalar scheduler now identifies forward branch regions whose
**every path reaches the same next CTA BAR**, with no intervening barrier, exit
or backedge. Reverse dynamic programming proves this bounded path property
without recursion. It inserts BSSY B0 before the region and BSYNC B0 immediately
before BAR. B0 is reused only after the matching synchronization. Existing
labels remap to the inserted instruction at their original boundary; the encoder
resolves BSSY targets through its existing recovered branch routine. The
scheduler preserves six-cycle BSSY/BSYNC issue gaps and outstanding scoreboard
waits. Straight-line phases retain exact code.

The [encoding observations](../../compiler_exp/gap_closure/python_shared_wide/convergence-encoding.json)
bind real-corpus B0 words and **56 varied forward targets/control words** through
nvdisasm. The final twenty-source **1,280-launch A100 gate** includes nested
branches, if/else writers, repeated phases and 32/64-bit shared elements.
[Regressions](../../tests/test_pyfront_shared_wide.py) check long phases, label
collisions, token reuse and ineligible exit/backedge paths separately from
hardware. Existing async-copy/LDSM dependencies remain separate.

This proves the named CTA-phase repair. It does not make divergent barriers
valid, prove races/initialization, add counted/partial barriers, assign more B
indices, or establish general cyclic/irreducible control flow or SIMT call-stack
semantics. Paths outside the proven region retain their existing code and
support boundary; absence of an inserted pair is not a convergence proof.
