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

