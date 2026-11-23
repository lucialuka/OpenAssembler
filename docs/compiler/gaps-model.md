# Batch-broadcast matrix products and regression contracts

Status: implemented

G25 identified that arithmetic broadcasting already worked while `matmul` batch
axes still required equal shapes. G51 independently identified a stale test that
expected supported `x + bias` to reject. These are different failures: the stale
test did not demonstrate a broadcasting implementation defect.

## Contract and solution

`torch.matmul`, `@`, and tensor `matmul` now accept contiguous positive-shape
FP32 operands of rank two through four, with matching matrix inner dimensions
and compatible trailing batch axes. For example, `[2,1,2,3] @ [1,4,3,5]` produces
`[2,4,2,5]`. The existing exact-shape plan and SM80 target remain explicit.

The frontend materializes each expanded operand through a bit-preserving
`broadcast_to` copy, flattens batch axes into the existing equal-batch BMM, and
restores the output shape. The backward graph computes each existing BMM
orientation and reduces replicated dimensions with `sum_to`. Shared parameter
uses still accumulate separately. Copies do not round values or replace them
with arithmetic identities, so signed-zero bits survive the expansion.

`torch.bmm` retains its rank-three, equal-batch contract. Vector/dot products,
batch-broadcast rank above four, noncontiguous external inputs, empty dimensions, mutation via
`out=`, and mixed dtypes remain outside this route. The pre-existing equal-batch
flattening route retains its broader input-rank admission. Tensor storage remains FP32.
Explicit `f16_rne_f32_accum` independently rounds the expanded matrix operands
and each derivative product as before; flattened batch <=32 and logical matrix
dimensions <=512 remain its separate recipe limits. Every buffer retains the
existing 16,777,216-element limit and CUDA grid bounds. There is no dynamic
specialization or new encoding.

Materializing expanded operands costs memory and launches. This is semantic
coverage, with no claimed performance improvement. Strided zero-copy expansion,
fusion and scalable broadcast reductions remain work under G24/G30/G46.

## Research cycle and retained evidence

The [candidate](../../compiler_exp/gap_closure/model/candidate.py) first compared
broadcast/flatten/BMM algebra and both input derivatives with independent eager
`torch.matmul`, using arbitrary output seeds. It independently enumerated flat
source indexing against `Tensor.expand` on eight shape pairs. Its
[CPU record](../../compiler_exp/gap_closure/model/candidate-results.json) uses
Python 3.12.10, Torch 2.7.1 and seed 25051. Production lowering followed this
candidate gate.

The [native regression tests](../../tests/test_model_batch_broadcast.py) cover
all four view/reuse policies, both expanded operands, missing leading axes,
shared-weight accumulation, keyword operands, rejected `out=` effects and
unsupported ranks/dtypes/contracts. Broadcast copies also pass the encoder and
SASS/PTX interpreters with poisoned registers and signed-zero bit patterns.
The [retained CPU gate](../../compiler_exp/gap_closure/model/cpu-results.json)
passes 270 model/diagnostic regressions on Python 3.12.10/Torch 2.7.1. CPU
algebra and emulation are distinct from the hardware result below.

The [A100 production gate](../../compiler_exp/gap_closure/model/hardware-20261120T153924Z.json)
passes **72 full-tensor comparisons / 3,280 elements** on NVIDIA
A100-SXM4-40GB, Python 3.12.1, Torch 2.7.1+cu126, CUDA runtime 12.6, and CUDA
toolkit 12.8.1. It covers seven broadcast shape pairs, shared weight derivatives
under all four storage policies, and an explicit-half outer-batch product.
There are 24 forward/backward runs with changed inputs, poisoned workspaces,
passing canaries and immutable caller inputs/weights, plus 12 independent Kineto
kernel-name/count ownership audits. FP32 controls use eager CPU autograd; the
half case uses the declared independently rounded per-product CPU reference.
The unchanged gate is rtol=4e-5, atol=5e-6; maximum observed absolute error is
4.76837158203125e-7. This gate measures correctness and ownership, without timing.

The [artifact archive](../../compiler_exp/gap_closure/model/hardware-20261120T153924Z.zip)
retains plans, PTX/cubins, ledgers, traces and complete compared tensors. Its
SHA-256 is `9fd831aa5a1d6e27846818a80cd35f0454acf9d554e96cc68528fe8f4d0f7db0`.
The report records compiler/harness hashes. The first remote attempt failed
before executing the compiler because the harness assumed its local directory
ancestry. The [startup failure](../../compiler_exp/gap_closure/model/hardware-startup-failure.json)
is retained separately and is not counted as a hardware correctness result.

## Regression maintenance

The [source diagnostics test](../../tests/test_source_diagnostics.py) now pairs
a genuine `add(alpha=2)` rejection with a positive bias-broadcast capture and
CPU output check. Sigmoid still supplies a separate unsupported-operator source
location control. This repairs the stale G51 assertion without weakening a
numerical test. Input and CPU-reference diagnostics name declared FP32, int64
index and Boolean mask types; model parameters and arithmetic remain FP32.
Deterministic emulator seeding and exact-division verification have their own
verification dossier and do not follow from these model checks.

Reproduce from the repository root in a supported Torch environment:

```sh
python compiler_exp/gap_closure/model/candidate.py
python compiler_exp/gap_closure/model/cpu_gate.py
modal run compiler_exp/gap_closure/model/run_modal.py
```

The Modal runner requests an A100 and a Python 3.12/CUDA 12.8.1 image with
Torch 2.7.1. Actual device/runtime identities in its result are authoritative.
Neither this bounded extension nor its test count closes the remaining model
operator, layout, dtype, state, distributed or serving gaps G21–G41.
