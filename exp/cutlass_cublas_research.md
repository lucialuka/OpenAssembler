# Public CUTLASS and cuBLASLt design audit

This is a source audit for experiment planning. It does not extract or copy
closed cuBLAS binaries, and it does not modify OpenAssembler production code.

## CUTLASS SM80 mainloop

The public CUTLASS SM80 implementation exposes the tuning dimensions directly:

* threadblock, warp and instruction tile shapes;
* stage count and `cp.async` pipeline state;
* operand layouts and shared-memory multiplicand swizzles;
* warp-raked or warp-striped thread maps;
* 128-bit access widths and alignment;
* warp fragment ownership and accumulator policy;
* CTA swizzle and epilogue vector width.

The SM80 `MmaMultistage` path fills `Stages - 1` stages in its prologue,
commits asynchronous copies, waits with `cp_async_wait<Stages - 2>`, and uses
double-buffered warp fragments. During the mainloop it loads the next shared
fragments while issuing MMA on the current transformed fragments. The final
path fences and waits for all pending copies before the CTA synchronization.

Sources:

* <https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/threadblock/default_mma_core_sm80.h>
* <https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/threadblock/mma_multistage.h>
* <https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/threadblock/mma_pipelined.h>

This is a direct experiment target for OA: the current OA GEMM emits a group
of LDSM operations and then a group of HMMA operations. A CUTLASS-like
load-next/compute-current schedule can preserve each accumulator's K-order
dependency while increasing independent HMMA overlap.

## Concrete Ampere configurations

The public examples show that no single geometry is universal:

* TF32 example 14: SM80, 128x128x16 threadblock, 64x64x16 warp, 16x8x8
  instruction, four stages, identity CTA swizzle.
* Operand-reduction fusion example 23: SM80, 128x128x32 threadblock,
  64x64x32 warp, 16x8x16 instruction, four stages, identity swizzle width 8,
  128-bit aligned operands.
* Fprop mainloop fusion example 25: the same 128x128x32 / 64x64x32 /
  16x8x16 geometry with four stages and per-channel scale+bias+ReLU fused into
  the mainloop.
* Wgrad mainloop fusion example 26: the same geometry with five stages.
* GEMM+LayerNorm fusion example 37: 128x128x32 / 64x64x32 / 16x8x16 with
  stages 3 and 4 for the two GEMMs, fusing local mean and square-sum reduction
  into the first epilogue and normalization into the second GEMM.
* GEMM+Softmax example 35: 128x128x32 / 64x64x32 / 16x8x16 with three stages
  and an `ApplyShape` that controls the normalization mapping.

Sources:

* <https://github.com/NVIDIA/cutlass/blob/main/examples/14_ampere_tf32_tensorop_gemm/ampere_tf32_tensorop_gemm.cu>
* <https://github.com/NVIDIA/cutlass/blob/main/examples/23_ampere_gemm_operand_reduction_fusion/ampere_gemm_operand_reduction_fusion.cu>
* <https://github.com/NVIDIA/cutlass/blob/main/examples/25_ampere_fprop_mainloop_fusion/ampere_fprop_mainloop_fusion.cu>
* <https://github.com/NVIDIA/cutlass/blob/main/examples/26_ampere_wgrad_mainloop_fusion/ampere_wgrad_mainloop_fusion.cu>
* <https://github.com/NVIDIA/cutlass/blob/main/examples/35_gemm_softmax/gemm_softmax.cu>
* <https://github.com/NVIDIA/cutlass/blob/main/examples/37_gemm_layernorm_gemm_fusion/gemm_layernorm.cu>

OA should test these as shape-specific candidates. Previous OA three/four-stage
regressions mean that public CUTLASS stage counts are candidate seeds, not
promotion evidence.

## CTA traversal and epilogues

`GemmIdentityThreadblockSwizzle<N>` groups block indices into swizzled tile
clusters based on N-tile count. The no-padding variant selects the largest
swizzle width that divides the N tile count. This is a grouped traversal and
should be compared with OA Morton traversal using L2, DRAM and scoreboard
counters.

CUTLASS tensor epilogues rearrange accumulator fragments through shared memory.
Mixed-output specializations select vector widths and shared-load layouts that
avoid bank conflicts. OA should search epilogue vector width, shared-store
layout and bank-conflict cost jointly instead of treating output stores as a
fixed tail.

## cuBLASLt public control surface

NVIDIA's cuBLASLt documentation describes heuristic dispatch rather than a
single fixed algorithm. `cublasLtMatmulAlgoGetHeuristic()` selects an algorithm
using problem dimensions, GPU configuration and descriptor parameters. NVIDIA
recommends querying once and reusing the result; the library also provides a
thread-safe LRU-like heuristic cache with default capacity 8192 entries.

The public preference/configuration surface includes:

* maximum workspace bytes;
* minimum A/B/C/D alignment;
* reduction scheme mask;
* tile ID;
* stage ID;
* split-K count and reduction scheme;
* CTA swizzling;
* inner-shape ID and custom options;
* fused bias, ReLU, GELU, dReLU, dGELU and bias-gradient epilogues.

Source: <https://docs.nvidia.com/cuda/cublas/index.html>

The transferable OA feature is a persistent, shape/alignment/workspace-aware
candidate cache. This is an integration of public policy ideas, not extraction
of vendor kernels.

