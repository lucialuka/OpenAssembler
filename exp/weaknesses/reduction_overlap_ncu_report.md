# Reduction-overlap Nsight Compute results

This experiment profiles the source-inclusive OpenAssembler baseline and the streamed load/FMA overlap candidate on an NVIDIA A100-SXM4-40GB. Each case was rebuilt through the OA CLI, paired with a same-source `ptxas -O3` binary, and profiled on the third exact launch using Nsight Compute kernel replay. The child worker validated outputs, canaries, and input immutability before and after profiling.

Raw record: `reduction_overlap_ncu_20261108T061002.104721Z.json`.

All **12 cases** (RMSNorm backward and softmax backward; width 512, 513, 1024; 4 and 8 warps) collected all requested counter groups. Every OA and ptxas profile passed both full-output checks. Nsight reported 21 accepted metric names; barrier stall was zero in every row because these kernels do not use CTA barriers.

## Main findings

- The overlap candidate reduced OA dynamic instructions in every case by **10–16%** and registers by **12–31**.
- The reduction is mostly FMA work removed from the old per-chunk product/add chain; it does not change the numerical tolerance contract.
- OA long-scoreboard stalls remained very high (about **82–91%**) and usually stayed flat or increased slightly after overlap. The candidate lowers scalar work but does not solve the dominant memory/dependency completion latency.
- `ptxas` remains materially better on long-scoreboard stalls (about **19–49%** across these cases) and executes fewer instructions with fewer registers.
- Lower register counts increased or preserved eligible warps only modestly; the remaining gap is not explained by occupancy alone.
- DRAM and L2 traffic changed only a few percent. The candidate is therefore an instruction/liveness improvement, not a memory-locality solution.

## Per-case counters

| Op | Width | Warps | Method | Inst | FMA | Regs | Long SB % | Short SB % | Wait % | DRAM bytes | L2 hit % | Eligible warps |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rmsnorm_backward | 512 | 4 | baseline oa | 921600 | 532480 | 58 | 86.64 | 1.99 | 1.22 | 26915072 | 50.30 | 0.17 |
| rmsnorm_backward | 512 | 4 | baseline ptxas | 626688 | 331776 | 40 | 48.56 | 11.18 | 1.94 | 26200832 | 50.28 | 0.12 |
| rmsnorm_backward | 512 | 4 | overlap oa | 798720 | 409600 | 43 | 86.34 | 2.23 | 0.96 | 27186304 | 51.02 | 0.19 |
| rmsnorm_backward | 512 | 4 | overlap ptxas | 626688 | 331776 | 40 | 47.74 | 11.47 | 2.40 | 26072576 | 50.52 | 0.14 |
| rmsnorm_backward | 512 | 8 | baseline oa | 921600 | 532480 | 58 | 82.32 | 2.37 | 1.37 | 27106048 | 50.17 | 0.19 |
| rmsnorm_backward | 512 | 8 | baseline ptxas | 626688 | 331776 | 40 | 46.61 | 11.74 | 2.21 | 25986560 | 50.68 | 0.16 |
| rmsnorm_backward | 512 | 8 | overlap oa | 798720 | 409600 | 43 | 85.47 | 2.08 | 0.97 | 27013760 | 50.78 | 0.21 |
| rmsnorm_backward | 512 | 8 | overlap ptxas | 626688 | 331776 | 40 | 45.85 | 10.22 | 2.29 | 25823360 | 50.18 | 0.17 |
| rmsnorm_backward | 513 | 4 | baseline oa | 974848 | 561152 | 61 | 85.32 | 2.13 | 1.25 | 27439104 | 53.02 | 0.19 |
| rmsnorm_backward | 513 | 4 | baseline ptxas | 671744 | 352256 | 48 | 28.74 | 6.06 | 1.97 | 26003456 | 52.28 | 0.17 |
| rmsnorm_backward | 513 | 4 | overlap oa | 843776 | 430080 | 45 | 87.43 | 2.06 | 0.81 | 27517312 | 55.45 | 0.18 |
| rmsnorm_backward | 513 | 4 | overlap ptxas | 671744 | 352256 | 48 | 28.67 | 6.92 | 2.06 | 25956992 | 52.19 | 0.18 |
| rmsnorm_backward | 513 | 8 | baseline oa | 974848 | 561152 | 61 | 84.36 | 2.12 | 1.21 | 27488000 | 52.93 | 0.19 |
| rmsnorm_backward | 513 | 8 | baseline ptxas | 671744 | 352256 | 48 | 26.11 | 4.99 | 2.15 | 26589056 | 53.68 | 0.22 |
| rmsnorm_backward | 513 | 8 | overlap oa | 843776 | 430080 | 45 | 85.56 | 2.32 | 0.87 | 27323776 | 55.01 | 0.20 |
| rmsnorm_backward | 513 | 8 | overlap ptxas | 671744 | 352256 | 48 | 27.02 | 5.50 | 1.75 | 26281856 | 53.21 | 0.18 |
| rmsnorm_backward | 1024 | 4 | baseline oa | 1576960 | 991232 | 106 | 87.19 | 1.75 | 1.46 | 54762752 | 49.69 | 0.09 |
| rmsnorm_backward | 1024 | 4 | baseline ptxas | 1085440 | 593920 | 72 | 34.73 | 7.50 | 2.69 | 51831936 | 50.20 | 0.13 |
| rmsnorm_backward | 1024 | 4 | overlap oa | 1323008 | 737280 | 75 | 85.85 | 2.01 | 0.94 | 51736192 | 50.46 | 0.11 |
| rmsnorm_backward | 1024 | 4 | overlap ptxas | 1085440 | 593920 | 72 | 34.57 | 7.70 | 2.88 | 51564416 | 49.64 | 0.14 |
| rmsnorm_backward | 1024 | 8 | baseline oa | 1576960 | 991232 | 106 | 86.10 | 1.65 | 1.46 | 55053952 | 50.17 | 0.09 |
| rmsnorm_backward | 1024 | 8 | baseline ptxas | 1085440 | 593920 | 72 | 37.56 | 7.72 | 2.74 | 51968000 | 49.97 | 0.12 |
| rmsnorm_backward | 1024 | 8 | overlap oa | 1323008 | 737280 | 75 | 86.59 | 1.69 | 0.85 | 52047232 | 50.66 | 0.10 |
| rmsnorm_backward | 1024 | 8 | overlap ptxas | 1085440 | 593920 | 72 | 37.73 | 7.12 | 2.86 | 51939072 | 50.14 | 0.13 |
| softmax_backward | 512 | 4 | baseline oa | 610304 | 299008 | 53 | 89.22 | 1.35 | 0.89 | 26935168 | 50.45 | 0.11 |
| softmax_backward | 512 | 4 | baseline ptxas | 495616 | 229376 | 40 | 30.70 | 8.17 | 2.10 | 25818368 | 50.29 | 0.15 |
| softmax_backward | 512 | 4 | overlap oa | 548864 | 237568 | 41 | 91.53 | 0.97 | 0.67 | 26887552 | 50.79 | 0.09 |
| softmax_backward | 512 | 4 | overlap ptxas | 495616 | 229376 | 40 | 30.94 | 8.51 | 2.22 | 25815296 | 50.40 | 0.16 |
| softmax_backward | 512 | 8 | baseline oa | 610304 | 299008 | 53 | 87.07 | 1.36 | 1.04 | 26999296 | 50.10 | 0.10 |
| softmax_backward | 512 | 8 | baseline ptxas | 495616 | 229376 | 40 | 29.83 | 7.33 | 2.18 | 25898112 | 50.49 | 0.19 |
| softmax_backward | 512 | 8 | overlap oa | 548864 | 237568 | 41 | 87.54 | 1.08 | 0.82 | 26977664 | 50.84 | 0.12 |
| softmax_backward | 512 | 8 | overlap ptxas | 495616 | 229376 | 40 | 29.90 | 7.56 | 2.19 | 25854720 | 50.51 | 0.19 |
| softmax_backward | 513 | 4 | baseline oa | 651264 | 315392 | 56 | 89.87 | 1.21 | 0.76 | 27249408 | 53.82 | 0.12 |
| softmax_backward | 513 | 4 | baseline ptxas | 532480 | 245760 | 47 | 28.37 | 7.97 | 2.24 | 26114816 | 52.31 | 0.17 |
| softmax_backward | 513 | 4 | overlap oa | 585728 | 249856 | 43 | 90.05 | 1.23 | 0.79 | 27486976 | 55.65 | 0.11 |
| softmax_backward | 513 | 4 | overlap ptxas | 532480 | 245760 | 47 | 28.66 | 8.04 | 1.95 | 26055040 | 52.06 | 0.14 |
| softmax_backward | 513 | 8 | baseline oa | 651264 | 315392 | 56 | 86.87 | 1.37 | 0.97 | 27803904 | 53.03 | 0.12 |
| softmax_backward | 513 | 8 | baseline ptxas | 532480 | 245760 | 47 | 27.25 | 6.87 | 1.73 | 26346368 | 54.10 | 0.16 |
| softmax_backward | 513 | 8 | overlap oa | 585728 | 249856 | 43 | 89.37 | 1.10 | 0.74 | 27548032 | 55.12 | 0.11 |
| softmax_backward | 513 | 8 | overlap ptxas | 532480 | 245760 | 47 | 26.46 | 7.03 | 2.07 | 26418944 | 54.02 | 0.19 |
| softmax_backward | 1024 | 4 | baseline oa | 1069056 | 561152 | 101 | 89.64 | 1.22 | 1.24 | 55245824 | 49.81 | 0.05 |
| softmax_backward | 1024 | 4 | baseline ptxas | 888832 | 425984 | 72 | 18.79 | 6.76 | 1.92 | 52167552 | 49.66 | 0.11 |
| softmax_backward | 1024 | 4 | overlap oa | 942080 | 434176 | 73 | 88.62 | 1.18 | 0.81 | 51734400 | 50.37 | 0.06 |
| softmax_backward | 1024 | 4 | overlap ptxas | 888832 | 425984 | 72 | 18.73 | 6.61 | 2.12 | 52236032 | 49.76 | 0.12 |
| softmax_backward | 1024 | 8 | baseline oa | 1069056 | 561152 | 101 | 88.68 | 1.23 | 1.23 | 55189248 | 49.88 | 0.05 |
| softmax_backward | 1024 | 8 | baseline ptxas | 888832 | 425984 | 72 | 19.07 | 8.22 | 2.56 | 52169472 | 49.86 | 0.14 |
| softmax_backward | 1024 | 8 | overlap oa | 942080 | 434176 | 73 | 90.77 | 0.95 | 0.73 | 52353152 | 50.33 | 0.06 |
| softmax_backward | 1024 | 8 | overlap ptxas | 888832 | 425984 | 72 | 19.45 | 8.63 | 2.30 | 52443136 | 50.10 | 0.13 |

## Promotion decision

The candidate is **not promoted**. It passes correctness, disassembly, and resource gates, and it reduces OA instruction/register pressure, but the counter evidence shows the dominant long-scoreboard bottleneck remains. The next reduction experiment should combine this liveness reduction with a memory/reduction-tree schedule that increases independent loads and arithmetic while preserving the existing synchronization and numerical policy.

The measurement is diagnostic: Nsight replay timing is not used as the benchmark. Paired graph timings remain in the source-inclusive run records and still show `ptxas` ahead on every tested reduction case.
