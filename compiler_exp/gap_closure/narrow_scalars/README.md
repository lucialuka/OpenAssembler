# CUDA-emitted narrow scalar register operations

Status: two actual original CUDA controls reject at 16-bit comparisons after
narrow entry admission; no candidate or production change.

The [baseline](baseline.json) reuses the exact independently compiled original
CUDA/PTX/cubin/listing/tool logs from the Boolean ABI study and verifies every raw
artifact hash. Both original entries now parse their parameter lists. The scalar
kernel rejects at `setp.ne.s16`; the Boolean-array kernel rejects at
`setp.eq.s16` and subsequently contains `selp.u16`. This is a scalar operation
admission boundary, without a current numerical failure or general CUDA-support
claim. The original sources are unchanged and no toolkit result is rewritten.

The next G11 solution must define narrow register input/output semantics rather
than assume the high bits of a physical 32-bit register already represent the
PTX signed/unsigned value. Comparisons need low-width signed/unsigned extraction;
selection and copies need the exact declared bit result and alias/guard snapshots.
Any added conversion/arithmetic/bit/shift form needs an explicit truncation,
extension, overflow, shift-count, immediate and modifier contract. Unsupported
forms remain explicit. Emulator agreement is insufficient if both paths assume
the same missing width rule.

Original CUDA and handwritten PTX controls must cover poisoned upper bits,
signed extremes, immediate/copy/alias/guard/loop cases and complete byte outputs
against independent references, actual ptxas and A100. New templates are unnecessary
only if validated existing 32-bit extraction/selection operations preserve the
whole narrow contract. Full source/native/resource, prior executable, measured
policy, installed/remote and dependency-matrix gates remain required before promotion.

```sh
python compiler_exp/gap_closure/narrow_scalars/probe.py
```
