# Naive scalar matrix-multiplication baseline

CLI examples below run inside the shell started with `openassembler`; enter
commands such as `compile`, `run` and `cnr` without repeating the executable name.
Homebrew, Python, Modal and SSH setup commands run in your regular terminal.

Status: tested-not-selected

The example remains source for a compiler baseline. Its retained byte-exact
comparison fails, and it is not the validated tensor-core `gemm` path.
No admitted speedup follows from its timing.

## Idea and implementation

Assign one thread to each output element and loop over the inner dimension
using scalar FP32 arithmetic. The source multiplies contiguous `A[M,K]` by
`B[K,N]` to return `[M,N]`, with no shared tiling or tensor-core matrix
instructions. This simple form exercises loops, address arithmetic and
accumulation; it intentionally leaves substantial data reuse/performance work.

The recorded test is only 32×32×64. Its storage layout differs from the native
tensor generator's `C=A@B.T` contract. They cannot be compared by exchanging
binary names without also matching layouts, precision and mathematical work.

## Measured outcome

The November 2 A100-SXM4-40GB study checks byte equality against ptxas in three
shuffled rounds. Scalar matmul fails in all rounds; the first shown output has
OA byte display `f6b39cbf` versus ptxas `f7b39cbf`. Best-round medians are about
0.02867 ms OA and 0.01434 ms ptxas, but they are diagnostic only because the exact
gate fails. The record does not independently bound FP64-reference error or
prove the mismatch is harmless.

This is not evidence against the separate native tensor implementation, nor
evidence that a low-rank algorithm is needed. Different arithmetic contracts
and source mappings answer different questions.

## Reproduce

Requirements: Python 3.12+, CUDA 12.8 tools, and a compatible GPU/driver or
authenticated Modal client for execution. From the repository root:

```sh
compile examples/nn_ops.py --kernel matmul --out-dir output_scalar_matmul
.venv/bin/python tools/nn_study.py --dry-run --out-dir output_nn_study
.venv/bin/python tools/nn_study.py --backend modal --budget 180 --rounds 3 --reps 20 \
  --out-dir output_nn_study
# Separately, the validated full-tile tensor route:
gemm --m 128 --n 128 --k 128 --dtype f16 --out-dir output_tensor_gemm
```

The eight-family driver supplies the actual small matmul buffers and layout;
the generic scalar auto-harness does not infer M/N/K. A fresh test uses the
current compiler; inspect its correctness status before any performance claim.
Do not pool best-of-round numbers with later full-tile graph timings.

## Source and evidence

- [Naive function](../../../examples/nn_ops.py),
  [study specs](../../../tools/nn_study.py),
  [recorded exact failure](../../evidence/nn-study-a100-20261102.json).
- [Current tensor generator](../../../openassembler/kernels/gemm.py),
  [native GEMM hybrid evaluation](native-hybrid-mlp.md).

An independently declared mathematical accuracy test and a tuned scalar-only
mapping would be needed to change this baseline's status. The production
tensor route has its own full-output/canary contract and is the relevant
current MMA implementation.
