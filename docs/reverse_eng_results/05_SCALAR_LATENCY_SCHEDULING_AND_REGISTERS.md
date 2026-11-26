# Scalar latency, scheduling and registers

## Forward CTA convergence phases

The [Python shared-array finding](../compiler/gaps-collectives.md#cta-phase-reconvergence)
is distinct from barrier issue latency. Scoreboard/layout ablations failed;
conditional-writer predication or an explicit BSSY/BSYNC pair passed. The
ordinary scalar scheduler now inserts one B0 pair only where forward paths all
reach the same next CTA BAR without an exit, barrier or backedge. It preserves
outstanding waits and six-cycle convergence issue gaps. A final 1,280-launch
A100 gate covers nested/if-else writers, repeated phases and 32/64-bit elements.
The functional model's implicit reconvergence cannot establish native execution
dependencies; cyclic/irreducible flow remains separate.

## Shared 64-bit barrier dependency

The [shared-width hardware gate](../compiler/gaps-memory.md#scalar-shared-byte-halfword-and-64-bit-memory)
exposed wrong optimized 64-bit exchanges when the scalar model reduced BAR to
one cycle. Safe OA and same-PTX ptxas controls passed. The validated production
repair retains at least six cycles for barriers in ordinary LDS/STS.64 code,
including existing vector-two 32-bit exchanges. A final 1,216-launch A100 gate
and 76 exact production-native comparisons pass. This is a bounded correctness
repair, with no minimum-latency or performance claim.

## Corrected A100 latency sweep

The corrected 42-case sweep separates fixed-pipe RAW dependencies from
predicate-to-guard dependencies. The measured minima are:

* approximately 4 cycles for tested same-pipe ALU/FMA pairs;
* approximately 5 cycles for tested cross-class pairs;
* approximately 4 cycles from a predicate definition to tested `SEL` use;
* approximately 13 cycles from a predicate definition to tested guarded
  instructions and branches.

These are minimum safe distances for the tested pair and operand forms. They
are not a complete A100 latency table, and they do not describe HMMA issue rate.

## Scoreboards and WAR

Variable-latency loads, operand-collection instructions and asynchronous memory
operations use dependency scoreboards. A later writer of an address register
must wait for the prior reader to finish; dropping that read-WAR edge can
produce an apparently shorter but incorrect kernel. Scoreboard generations must
retire before a slot is reused.

## Register allocation

The compiler lowers PTX through virtual values, removes dead/copy-only values,
allocates aligned groups for wide and tensor operands, and then emits physical
register text. Current scheduling still has important ordering constraints:
physical choices can affect later movement, and the tensor path intentionally
keeps conservative source order. The experiment scheduler under
`exp/tensor_scheduler` explores scheduling over lowered blocks but is not a
production replacement.

## Production decision

The [bounded scalar spill route](../compiler/gaps-spilling.md) adds explicit
reload/store dependencies and final local-frame resource propagation after
ordinary allocation fails. It preserves existing latency/scoreboard rules and
all retained measured policy executables. Independent A100 and byte/state
checks precede promotion. One matched pressure-kernel resource study finds
tighter spilling slower; it does not select a global register/occupancy policy.
The subsequent aligned pair/quad and predicate extension passes another 320
A100 launches with self-guarded/negated loop state and complete private/output
checks. Slot reuse, cross-region scheduling and occupancy policy remain open.

The subsequent [virtual pressure pass](../compiler/gaps-spilling.md#pressure-aware-virtual-scheduling)
changes operation order before physical allocation under explicit dependency,
guard, fixed-boundary and memory-order constraints. Its selection requires
strictly fewer spill slots, no larger frame and non-increased local transfers
in every corresponding basic block. Paired A100 resident intervals improve on
the six new pressure fixtures; ptxas remains faster in those nominal controls.
An independent host code-generation study exposes added compile work. This
adds a bounded scheduling route, with no global occupancy or application claim.

The measured scalar classes are used in the scalar backend with conservative
fallbacks. A mined-latency scheduler promotion was rejected after full
`nvdisasm` validation produced invalid SASS on ordinary kernels. This is a
key reverse-engineering lesson: matching control fields on a probe does not
prove that an arbitrary reordered block is legal.

The November 11 GEMM table does not relax the scalar scheduler or dependency
rules. Its shape choices select already validated geometry, address
initialization, pointer placement or CTA traversal under exact contracts;
HMMA8/drain15, scalar RAW/WAR, async-copy visibility and allocation invariants
remain in force. The remaining instruction-count and scoreboard gaps are
scheduler/compiler research targets, not claims solved by dispatch.

