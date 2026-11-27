# Native sigmoid and static tensor reductions

Status: implemented

G21 identified the small framework operator whitelist. This extension admits
sigmoid and FP32 `sum`/`mean` through the ordinary graph/compiler/runtime path.
The larger operator coverage gap remains open; convolution, indexing, arbitrary
activation families and general framework execution do not follow from these
three operators.

## Admitted contract

| Operation | Supported forms | Explicit boundary |
|---|---|---|
| Sigmoid | `torch.sigmoid`, `nn.Sigmoid`, tensor `.sigmoid()` and functional sigmoid; contiguous exact FP32 shape; backward from saved output | Approximate exp2/reciprocal with flushed subnormal exponential tails; no dtype conversion, `out=` or in-place operation |
| Sum/mean | `torch.sum`/`torch.mean` and tensor methods; static integer `dim`, tuple/list of distinct axes, or `None`; Boolean `keepdim`; rank 0..4 FP32 | Exact positive shapes; no integer promotion, dtype conversion, `out=`, dynamic axes, named axes or empty input dimensions |
| Reduction axis normalization | Negative axes normalized by rank; an empty axis tuple/list reduces all dimensions, matching Torch; scalar axes 0/-1 produce a scalar | Duplicate and out-of-range axes reject; output shape is checked against normalized axes |

Optional explicit `dtype=torch.float32` preserves the admitted dtype. Tuple/list
**axes** select one tensor reduction result; a tuple of model **outputs** retains
the existing output-container and explicit-seed semantics. A scalar input has
rank zero and one element; it is distinct from a zero-element tensor.

Existing 16,777,216-element buffer and launch bounds remain. Storage, reductions,
seeds and gradients remain FP32 under either existing model math policy; the
explicit half policy continues to apply only to matrix operands. Finite input
and finite FP32 reduction intermediates remain required. Serial accumulation
order can differ from eager Torch's reduction tree. No claim of bitwise equality
with Torch or a universal numerical error bound is made.

## Candidate, validation and promotion

The [independent candidate](../../compiler_exp/gap_closure/operators/candidate.py)
first tested stable sigmoid algebra and reduction normalization into
`sum_to`/reshape/scaling against eager CPU Torch and arbitrary-seed derivatives.
Its [19-case record](../../compiler_exp/gap_closure/operators/candidate-results.json)
uses Python 3.12.10, Torch 2.7.1 and seed 21052. Production integration followed
that algebra gate.

Reduction lowering preserves reduced axes as size-one dimensions, uses the
existing deterministic `sum_to`, applies reciprocal element count for `mean`,
and restores the requested shape with an immutable view/copy. Backward expands
upstream gradients with the existing bit-preserving `broadcast_to`; graph
joins and shared parameters retain explicit gradient accumulation.

The new sigmoid kernel evaluates `z=exp(-abs(x))` and selects `1/(1+z)` or
`z/(1+z)` by sign. It clamps the absolute argument to 104 before exp2 scaling,
keeping intermediates finite for extreme finite FP32 inputs. Native exp2 and
reciprocal remain approximate; exponential subnormal tails flush to zero.
Backward uses `dy * (y * (1-y))` with the saved output. These policies are
explicit and do not assert exact exceptional-value Torch semantics.

The [regression suite](../../tests/test_model_operators.py) covers both Torch and
tensor reduction entrypoints; scalar, integer/negative/list/tuple/empty axes;
all four view/reuse policies; all four sigmoid entrypoints; extreme ±1e30 and
143-element tails; shared-parameter and multioutput gradients; and typed/effect
rejections. Sigmoid PTX encodes and its SASS emulator results match an independent
Python-math reference with poisoned registers. Reduction kernels also encode
and match the PTX interpreter under register poisoning. The
[combined CPU gate](../../compiler_exp/gap_closure/operators/cpu-results.json)
passes **501 tests** in Python 3.12.10/Torch 2.7.1.

The [November 21 A100 gate](../../compiler_exp/gap_closure/operators/hardware-20261121T122827Z.json)
passes **416 full-tensor comparisons / 30,576 elements** on NVIDIA
A100-SXM4-40GB, Python 3.12.1, Torch 2.7.1+cu126, CUDA runtime 12.6 and CUDA
toolkit 12.8.1. It executes 200 forward/backward runs with changed inputs,
poisoned workspaces, passing allocation canaries and immutable external inputs
and weights, plus 100 independent Kineto ownership audits. All four storage
policies cover reductions including widths 33 and 513, sigmoid entrypoints and
combined tuple-output/shared-bias derivatives. The unchanged gate uses
rtol=4e-5, atol=5e-6. Maximum absolute difference is 1.1444091796875e-5, on a
513-element sum; the combined relative/absolute criterion passes.

The [ordinary model repository](../../compiler_exp/gap_closure/operators/repo/model.py)
combines Linear, sigmoid, mean and sum. Public CLI compilation and saved replay
each pass 35 full output/loss/gradient/weight checks across five SGD updates,
with zero vendor compute. Both complete reports, plans, PTX/cubins, native
ledgers, traces and full compared tensors remain in the
[archive](../../compiler_exp/gap_closure/operators/hardware-20261121T122827Z.zip).
Its SHA-256 is `471a5b3713754b6f1e0072debd6f91668c89ea7dfad85e5d99e2edac7473f8d7`.
The [identity audit](../../compiler_exp/gap_closure/operators/audit.json) checks
101 bundle plan/source/binary identities, archive/harness identities, and that
the three changed production operator modules match both CPU and hardware
source records. No performance improvement is inferred from correctness or
profiling activities; expanded reductions remain a scalar correctness baseline.

## Retained failures and maintenance

The [first CPU regression record](../../compiler_exp/gap_closure/operators/cpu-initial-results.json)
retains 494 passes and one stale `nn.Sigmoid` rejection after the capability was
added. That rejection now uses unsupported `nn.SiLU`; the separate source-line
negative test uses unsupported `torch.sin`. New positive sigmoid tests pair
those changes with explicit numerical and derivative checks. This does not
weaken a numerical gate. The first encoded-sigmoid harness also called a
nonexistent emulator `read_bytes` helper; its
[failure record](../../compiler_exp/gap_closure/operators/harness-failures.json)
explains the corrected word-read API. No hardware failure or tolerance change
was hidden by either maintenance correction.

Reproduce from the repository root in a supported Torch environment:

```sh
python compiler_exp/gap_closure/operators/candidate.py
python compiler_exp/gap_closure/operators/cpu_gate.py
modal run compiler_exp/gap_closure/operators/run_modal.py
python compiler_exp/gap_closure/operators/audit.py
```

The Modal command requests an A100 and a Python 3.12/CUDA 12.8.1 image with
Torch 2.7.1. Actual device/runtime metadata in the record is authoritative.
The fixed November 21 audit references the retained hardware record; a future
replay produces a new dated record and does not replace its predecessor.


## Convolution coverage

The [Conv2d extension](gaps-convolution.md) adds FP32 NCHW/OIHW groups=1 forward,
input/weight/bias gradients and ordinary CNN training/replay. Its independent
A100 gate and public CLI example extend G21 into compact convolutional models.
The operator reference lists the supported static attributes and remaining
coverage. [Operation reference](../reference/supported-operators.md)
