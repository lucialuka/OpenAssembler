# Matched ML task benchmark suite

## Question and boundaries

This suite compares OpenAssembler's verified SM80 cubins, direct Triton kernels,
NVIDIA ptxas compiled from each exact OA PTX operation contract, and a PyTorch
CUDA operator schedule using NVIDIA libraries on five bounded synthetic tasks. It does not
claim these five tasks represent whole-model training generally.

Task definitions are frozen in tasks.py. Each arm uses the same shapes, seeded
initial FP32 weights, target, FP32 or explicitly named FP16-RNE-input/FP32-
accumulation contract, output, loss, gradient update and iteration count.
The PyTorch CUDA operator schedule uses an explicit straight-through RNE conversion for the half
contract. TF32 is disabled for FP32 eager checks. Triton uses IEEE dot for
FP32 and binary16 dot for explicitly rounded inputs. The standalone Triton code
does not call Inductor. No vendor result enters OpenAssembler's production
runtime.

The matched NVIDIA compiler arm compiles OA-emitted PTX with ptxas, preserving
identical operations, signatures and launch shapes. This isolates “OA backend
versus ptxas for the same PTX.” The PyTorch CUDA arm manually executes the same
logical plan through eager framework operators, including explicit forward and
reverse operator schedules; matrix products dispatch through installed NVIDIA
CUDA libraries. It is not stock module/autograd execution, a full nvcc CUDA C++
model-compilation path, or a cuBLAS algorithm-search benchmark. Its matrix and
reduction algorithms may differ from the OA launch plan. A direct cuBLAS wrapper
attempt failed API status checks and is excluded. These two controls provide
useful NVIDIA toolchain/library points of comparison, not a benchmark of every
NVIDIA compiler, autotuner or model runtime.

Every arm passes four changed-input complete-state checks against CPUReference
(or FP32 CPU Module/autograd where contracts match), including output, loss,
all input/parameter gradients, updated weights, immutable inputs/targets and
allocation guards. OA additionally passes raw profiler-category exact-name/count
ownership. Numerical tolerance is fixed before timing. Failed arms are retained
and never timed as accepted.

Each steady-state invocation is warmed, captured into a four-step CUDA graph,
then measured in 15 shuffled paired samples per round for three rounds. This is
resident-input stream time and excludes CPU data generation, H2D transfers,
capture, setup and build. A separate uncaptured host-call distribution includes
Python dispatch and synchronization. A separate 15-sample timing includes CPU
input/target copy, graph replay and output/loss delivery. H2D is timed separately.
Per-part timing is independently paired: forward, loss/seed, backward and
update. Component medians are not summed to manufacture a total. Setup reports
capture, cubin bundle, runtime initialization and other arm warmup separately.
Profiler per-kernel durations are diagnostic traces, not timing distributions.

Device variants are retained separately; do not pool A100-SXM4-40GB, 80GB PCIe,
or any other A100. Reproduction uses the exact pinned GPU worker and source hashes
in each run result.

## Tasks

1. Tiny regression MLP (17×8): small-grid FP32 training, launch/host overhead.
2. Dense regression FP32 (128×128): FP32 forward/dX/dW, MSE and SGD.
3. Dense regression FP16 RNE products (128×128): explicit Tensor Core training;
   reports both packing and matrix stages.
4. Attention training (B=2, sequence=7, hidden=8, two heads): q/k/v, permutations,
   batched QKᵀ and PV products, softmax, projection, residual, affine LayerNorm,
   tanh GELU, FFN, MSE, backward and update.
5. Projection inference (128×128): three shared-input Q/K/V projections and add,
   FP16-RNE inputs with FP32 accumulation/output; reports no training phase.

This is not an optimized transformer serving workload: no masking, embeddings,
dropout, dynamic sequence batches, KV paging or a serving runtime.

## Interpreting a table

Every table must show actual GPU model, math/dtype, batch and dimensions. Report
paired medians/round ranges and preserve samples. The component breakdown must
include layout/pack, GEMM/BMM, reductions/activation, loss, backward/update,
launch/graph, and transfers when present. Label framework-level compilation and
setup separately from steady execution. If an arm fails any correctness or
ownership gate, exclude all its speed values from admitted comparisons.
