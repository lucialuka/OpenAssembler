# Counters, performance and validation

## Counter interpretation

The most important A100 GEMM counter result is that OA and ptxas can execute
the same Tensor Core instruction count while differing substantially in total
instructions, waits, scoreboards and memory traffic. This moves the
optimization target from “add more HMMA” to “make tensor, memory and observer
work overlap without adding scalar overhead.”

The retained native GEMM counter records show, for example, OA around 287M
total instructions versus ptxas around 217M in one tall FP16 control while both
execute 134,217,728 tensor instructions. Morton can improve OA locality and L2
hit rate, yet still lose if scalar/control work and wait stalls remain high.

## Validation gates

Every candidate should report:

1. exact output or declared tolerance;
2. canary protection and input immutability;
3. poisoned CUDA graph replay when graphs are used;
4. valid disassembly and opcode/control checks;
5. registers, shared memory, occupancy and launch contract;
6. matched same-PTX ptxas output;
7. paired randomized timing on the same GPU;
8. counters when a stall or locality explanation is claimed; and
9. held-out shapes before any dispatch promotion.

## Comparison boundaries

* OA versus same-PTX ptxas measures backend/compiler gap.
* OA versus cuBLAS measures a library dispatch baseline and may use different
  kernels, tiling and workspace policies.
* TorchInductor/Triton application results measure the configured application
  frontend, not native OA SASS substitution.

The complete per-feature status remains in [the research catalogue](../research/README.md).

## November 9 current-source revalidation

The current source-path load-next confirmation passes every gate on FP16/BF16
1024×4096×1024 and 4096×1024×1024. It uses 24 more registers (216 to 240) and
measures roughly 8–10% lower OA time than the source-order schedule in paired
A100 rounds. Three held-out large FP16 shapes all regress, so the option is
exact-shape and opt-in. Grouped-N keeps 216 registers, passes the same gates and
improves repeated 16384×4096×4096 by roughly 6.9–7.3%; held-out tall FP16/BF16
cases improve while an 8192×4096×4096 control is neutral. These are layout and
fragment schedule results, not evidence that OA beats same-source ptxas.

## Role-matched FFN-up counters: November 9

The [current six-method replay](../../exp/shape_coverage/a100-role-gemm_20261109T111551.304398Z.json)
profiles FP16 1024×4096×1024 on one A100-SXM4-40GB. All full-output,
three-pattern, canary, input, same-policy and baseline-bitwise, poisoned-graph,
native manifest and encoded-control checks pass first. All 20 requested
counters were collected; raw CSV, units, queries and commands are retained.

| Method | Dynamic instructions | Tensor instructions | Registers | Tensor active % | Short scoreboard % | Long scoreboard % | Wait % |
|---|---:|---:|---:|---:|---:|---:|---:|
| OA source-order | 4,895,744 | 2,097,152 | 216 | 58.97 | 8.53 | 16.28 | 41.62 |
| OA direct + early | 4,371,456 | 2,097,152 | 216 | 62.00 | 9.85 | 12.77 | 42.37 |
| OA load-next / split-4 | 4,895,744 | 2,097,152 | 240 | 57.72 | 6.55 | 16.83 | 42.54 |
| OA split-6 | 4,895,744 | 2,097,152 | 236 | 57.32 | 6.35 | 17.67 | 42.09 |
| Same-source ptxas | 3,230,720 | 2,097,152 | 231 | 72.06 | 0.70 | 9.16 | 34.86 |
| Matched cuBLAS GemmEx | 5,417,984 | 2,097,152 | 220 | 73.12 | 3.61 | 1.80 | 32.64 |

Direct+early removes 524,288 dynamic instructions from the OA baseline but still
executes 35.3% more total instructions than ptxas. All methods report the same
global-load sector count, roughly 13.1–13.4 MB DRAM traffic, and a two-block
register occupancy limit. The register reduction from split-4 to split-6 does
not change that limit and does not improve paired timing.

Load-next reduces measured shared-load bank conflicts from 6,406 to 1,538;
split-6 records 1,354. Neither the smaller bank-conflict count nor register
count predicts the fastest method. cuBLAS has more dynamic instructions than
OA but better tensor activity, less long-scoreboard wait and faster FP16
timing. This supports optimizing overlap and non-tensor work together rather
than minimizing one counter in isolation. These observations are not a causal
decomposition of the remaining runtime.

Counters use deterministic finite operands in fresh driver contexts, one warm
call and one profiled call. All kernels inside a cuBLAS call are retained.
Paired timing separately uses normal inputs and four-launch CUDA graphs;
profiler replay duration is never used as benchmark time. Clocks were not
locked, stall percentages are not additive runtime shares, and no cross-device
absolute timing comparison is made.

## Small-grid geometry counterexample

The [128×512×512 FP16 counter study](../../exp/shape_coverage/a100-small-tile-counters_20261109T113502.971650Z.json)
passes all 15 full-output method/pattern checks and collects all 20 requested
metrics in six profiles. It compares separate 128/64-tile source-inclusive OA
and ptxas cubins, with the correct 64/32 KiB shared-memory launches and matched
cuBLAS controls.

| Counter | OA tile128 | OA tile64 | ptxas tile64 |
|---|---:|---:|---:|
| Dynamic instructions | 42,512 | 75,328 | 40,640 |
| Tensor instructions | 16,384 | 16,384 | 16,384 |
| Registers / theoretical register block limit | 216 / 2 | 88 / 5 | 84 / 5 |
| Active warps % | 6.23 | 6.11 | 6.18 |
| Short / long scoreboard % | 8.48 / 18.23 | 12.92 / 21.43 | 16.27 / 34.35 |
| Wait % | 50.00 | 40.43 | 21.03 |
| Global-load sectors | 32,768 | 65,536 | 65,536 |
| DRAM bytes | 683,136 | 669,440 | 668,032 |
| L2 hit % | 47.13 | 62.75 | 61.73 |
| Shared-load bank conflicts | 0 | 0 | 0 |

The faster small OA tile does more total work and staging, while expanding the
grid from four to 16 CTAs. Its resource ceiling improves, but measured active
warps do not increase. The data support more grid parallelism and shorter
per-CTA work as explanations, not a demonstrated occupancy increase or a bank
conflict fix. Extra staging traffic is mostly cached in this resident workload.

The same-geometry compiler gap remains: OA tile64 has 58,944 non-tensor
instructions versus ptxas's 24,256, about 2.43×. Its lower long-scoreboard
percentage does not make it faster; wait stalls and instruction overhead also
matter. cuBLAS launches 64 CTAs and executes 243,200 total instructions while
winning paired timing. See the [canonical tile dossier](../research/ideas/small-grid-tensor-cta-tiles.md)
for timing, exact scope, held-out evidence and negative cases.

