# Long-scoreboard reduction follow-up

This is an experiment plan for the 4096-row backward reduction gap. It does
not change the production reduction backend.

## Why FMA alone is insufficient

The existing FMA candidate is a useful first step. At width 513 it reduces
RMSNorm backward from 61 to 44 registers and from 238 to 205 generated
instructions, with about a 1.33--1.37x improvement over the unfused OA
template. Softmax backward improves about 1.24--1.35x. Same-source ptxas is
still faster, and the large-shape counter evidence shows long-scoreboard stalls
around 77--88% for OA versus roughly 24--29% for ptxas in the 4096x513 family.

The remaining problem is therefore load completion and reduction overlap. The
current generator loads all row chunks before local products and warp shuffles;
the FMA rewrite changes arithmetic liveness but does not create a deeper
load/reduction pipeline.

## Candidate variants

Test the following variants independently, then combine only the survivors:

1. **Load/FMA streaming.** Load a bounded window of row chunks, immediately
   accumulate the local FMA terms, and release the source fragments before the
   next window. Keep the full-width and tail masks unchanged.
2. **Balanced reduction tree.** Keep the five shuffle offsets and masks, but
   schedule the next independent local FMA or load between shuffle and its
   dependent consumer when the dependency graph permits it.
3. **Explicit shuffle read tracking.** Treat `SHFL` as an asynchronous reader
   for register reuse and preserve its write/read scoreboards across register
   allocation. Never remove a WAR edge merely because a shuffle is uniform.
4. **Synchronization sweep.** Compare the current WARPSYNC/SHFL protocol with
   a graph-derived protocol that removes only redundant barriers proven dead by
   full-lane participation and matching masks.
5. **Pressure-aware dispatch.** Choose the bounded window from register count,
   active-warps estimate and row width. A larger window is allowed only when it
   does not reduce the resident warp count.

## Required measurements

For every candidate at rows 4096 and widths 513, 512 and 1024, record:

* full output, canaries, unchanged inputs and declared FP32 tolerance;
* register count, static and dynamic instruction counts;
* DRAM bytes, global-load sectors and L2 hit rate;
* short and long scoreboard stalls, wait and barrier stalls;
* warp eligibility/occupancy and shared-memory traffic;
* paired OA/ptxas timing with two order-reversed rounds.

The target is not simply a lower long-scoreboard percentage. A candidate is
useful only if it lowers long-scoreboard time without increasing total time,
register pressure or numerical error. A reduction candidate that beats the
unfused OA template but still loses to ptxas remains experiment-only.

## Suggested implementation boundary

Keep the existing `exp/nn_round2/operator_candidates.py` FMA transformation as
the arithmetic control. Add a separate source generator under `exp/` for
load/FMA windows and schedule metadata. Do not monkeypatch
`openassembler/kernels/reductions.py` until the candidate has passed the full
counter and held-out-shape gate.

