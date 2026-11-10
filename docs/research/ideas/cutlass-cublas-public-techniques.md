# Public CUTLASS and cuBLASLt techniques for A100

Status: untested-or-hardware-unavailable

Status applies to the public technique transfer described here. It does not
claim access to or extraction of proprietary cuBLAS device code.

## Scope and source boundary

cuBLAS device kernels are proprietary. OpenAssembler can observe their public
API behavior, kernel names, counters, workspace requirements and timing, but it
should not treat a cuBLAS cubin as source code that can simply be copied into
the compiler. CUTLASS is public source and is the appropriate source for
reproducible algorithmic and pipeline designs.

The primary public references inspected for this dossier are:

* [CUTLASS efficient GEMM](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)
* [CUTLASS synchronization and pipelines](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/pipeline.md)
* [CUTLASS SM80 multistage mainloop](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/collective/sm80_mma_multistage.hpp)
* [cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/index.html)

## Transferable CUTLASS techniques

### Hierarchical tiling

CUTLASS separates CTA tiles, warp tiles, K groups and individual MMA
instructions. The loop order exposes independent work at multiple levels:

* CTA-level M/N/K tiles provide block-level parallelism and locality;
* warp-level M/N tiles provide register reuse and Tensor Core parallelism;
* K groups are fully unrolled into MMA instruction groups;
* the MMA instruction loop is the lowest tensor scheduling unit.

OpenAssembler already has a 128×128×64 CTA tile and four 64×64 warp tiles. The
missing part is a scheduler that sees the same hierarchy instead of receiving a
flattened source-order HMMA list after physical allocation.

### Shared-memory and register double buffering

CUTLASS's SM80 mainloop requires at least two `cp.async` stages. It prefetches
all but the last stage, waits for `Stages-2`, synchronizes the CTA, then uses a
register fragment pipeline to load the next shared-memory fragments while the
current MMA group executes. At the start of each K block it launches the next
global-to-shared copies, commits the async group, advances the circular stage
indices and continues tensor work.

This is a concrete design target for OA's dependency graph:

* `cp.async` stage writes are producer events;
* `wait_group` and CTA visibility are operand-ready events;
* LDSM/register loads for the next K block are independent of the current
  accumulator chain;
* HMMA groups consume the current register fragments;
* stage index and pointer updates are observer/address edges.

The mainloop should therefore be modeled as a circular pipeline with separate
global-copy, shared-load, register-fragment and HMMA stages. A global drain at
every HMMA run is a weaker approximation.

### Epilogue separation

CUTLASS keeps the mainloop accumulator organization optimized for Tensor Core
reuse, then runs a separate epilogue that exchanges or reorders values for
coalesced global stores and applies scaling, bias, activation or other visitor
operations. This matters for OA because forcing the accumulator layout to also
serve the final store can increase address arithmetic, register moves and
observer waits inside the tensor mainloop.

The next OA fused-epilogue experiment should compare a short mainloop with a
dedicated epilogue staging layout against the current direct accumulator-store
path. The epilogue must be measured as part of the full kernel.

### CTA rasterization and locality

CUTLASS exposes threadblock swizzles to increase reuse of tiles in the last
level cache. This is consistent with OA's Morton experiment: traversal changes
DRAM traffic and L2 hit rate without changing the HMMA count. The dispatch
decision should therefore include CTA traversal and tile shape, not only the
tensor schedule.

### Split-K and reductions

CUTLASS documents split-K and parallelized reductions for problems with too few
CTA tiles to occupy the GPU or with large K. These are algorithmic options with
workspace and reduction costs. OA should treat split-K as a shape-specific
dispatch candidate and include the reduction kernel, workspace traffic and
launch overhead in the timing contract.

## Transferable cuBLASLt behavior

The public cuBLAS documentation exposes several useful control concepts:

* `cublasLtMatmulAlgoGetHeuristic()` returns a ranked set of algorithms rather
  than committing to one universal kernel;
* the preference descriptor includes a maximum workspace size;
* the preference descriptor can mask reduction schemes;
* heuristic results can be cached because host-side heuristic work can take
  tens of microseconds;
* cuBLASLt may decompose a problem internally and combine subproblem results;
* alignment, epilogue selection, layout and workspace affect admissible
  algorithms.

These are dispatch and search ideas, not proprietary instruction sequences.
OA can reproduce the structure with a measured local manifest:

```text
shape/layout/dtype/alignment
    -> candidate tile/stage/traversal/epilogue/reduction policies
    -> static resource filter
    -> correctness gate
    -> paired A100 timing and counter sample
    -> cached shape-specific winner
```

The current public `nvidia-matmul-heuristics` documentation describes its
current heuristic coverage for Hopper and Blackwell rather than A100. OA should
therefore reuse the heuristic methodology and candidate metadata, not claim
that the current package directly supplies SM80 decisions.

## What should be integrated into OA experiments

The first experiment-only candidate family should add these independent knobs:

1. `stages`: two versus three only when full output and resource gates pass;
2. `register_k_blocks`: bounded register-fragment pipeline depth;
3. `hmma_chain_policy`: one, two or three independent accumulator chains;
4. `observer_policy`: targeted store/scalar/barrier delays versus fallback drain;
5. `cta_order`: m-fast, Morton and grouped traversals;
6. `epilogue_mode`: direct store versus dedicated staged epilogue;
7. `split_k`: disabled, bounded split-K with explicit reduction cost;
8. `workspace_budget`: a dispatch constraint, not an unmeasured assumption.

Each candidate must preserve the exact input/output precision contract and pass
full output, canary, input immutability, disassembly, register/shared-memory,
occupancy and counter checks. cuBLAS timing is a reference point; it is not a
source of copied device code.

## Current status

The public techniques explain why cuBLAS/CUTLASS can outperform the current OA
mainloop: they combine hierarchical tiling, a circular async-copy pipeline,
register-fragment buffering, shape-specific traversal, separate epilogues and
heuristic dispatch. OA currently has pieces of these mechanisms, but its fixed
geometry and source-order tensor scheduling do not expose them to one joint
cost model.

The immediate implementation boundary remains experiment-only. The next
candidate should add the CUTLASS-style register/K-block pipeline and targeted
observer policy to the existing OA dependency graph, then compare held-out A100
GEMMs and backward reductions against matched ptxas and cuBLAS controls.

## Source-inclusive A100 experiments

The experiment layer now tests two concrete public ideas without changing the
production compiler:

* `exp/tensor_scheduler/load_next_compute_current.py` alternates two fragment
  register windows and inserts the next K-fragment load between the current
  HMMA halves. It preserves the existing async-copy/barrier protocol and
  per-accumulator K order. On six FP16 A100 shapes it passed full outputs,
  canaries, poisoned graph replay, disassembly and resource checks. It beat
  the current OA schedule on five non-square shapes, by about 1–10%, but
  regressed 4096×4096×4096 by about 10% and used 24 more registers per thread.

* `exp/gemm_dispatch/cta_swizzle.py` and `grouped_n_modal.py` compare
  grouped-N, Morton-8 and m-fast traversal with the required launch-grid
  change. The tall 16384×4096×4096 FP16 case favored grouped-N and Morton by
  about 6–7% over m-fast; the wide and square cases were ties or regressions.

The records are
[`measured_dispatch_manifest.json`](../../../exp/gemm_dispatch/measured_dispatch_manifest.json),
the [load-next dossier](cutlass-load-next-compute-current.md) and the
[grouped-N dossier](grouped-n-cta-swizzle.md). These are shape-specific results.
They do not justify copying cuBLAS kernels or applying a universal traversal
rule. Alignment, workspace, BF16 and neighboring shapes remain separate
manifest entries until measured.

