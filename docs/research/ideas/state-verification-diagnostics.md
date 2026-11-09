# Complete application-state verification diagnostics

Status: implemented

The worker now explains every observed failed state group while retaining the
original numerical gate. This is a reporting/correctness feature, not a repair
of the BF16 training mismatch or a speed optimization.

## Problem and design

The old application verifier stopped at the first tensor mismatch. That hid
later failed parameters and made it difficult to tell an output problem from
optimizer amplification. The new verifier still calls the original
`torch.testing.assert_close` with the same reference direction, dtype/shape/
layout checks, tolerance policy and `equal_nan=False`, but completes comparison
of all observed state entries before raising an error.

`StateVerificationError` is a `RuntimeError` subclass with structured JSON
diagnostics; the worker includes them in its failed response. A failure still
aborts publication. The change does not alter the model, optimizer, epsilon,
precision defaults or tolerance. It also does not silently reinterpret retained
`.grad` as raw gradients: clipping, scaling or accumulation belong to the
application and cannot be inferred by a generic worker.

The report retains exact checked/failed counts by outputs, parameters,
gradients, optimizer and other state. Detailed output is bounded: up to eight
failed tensors, eight logical index samples each and 256-character key samples.
It includes elementwise mismatch counts, chosen tolerances, finite-pair L2
norms/absolute differences and nonfinite counts. Undefined/overflowing
statistics use JSON `null`. Structured diagnostic JSON is capped at 12,000
characters by omitting details while preserving totals. A statistics error
cannot turn the original comparison failure into a pass.

## Validation and findings

Eighteen tests cover simultaneous group failures, separate tolerance policies,
BF16/FP32/integer/complex/bool values, metadata/type errors, NaNs, matching
infinities, noncontiguous logical indices, empty tensors, missing keys,
thousands of failure names and worker failure/no-publication behavior. All 18
pass in both the local PyTorch environment and pinned 2.7.1 GPU image.

An unchanged A100-SXM4-40GB wide BF16 run checks 127 tensors after 29 updates:

| Group | Checked | Failed |
|---|---:|---:|
| Observed outputs | 2 | 0 |
| Parameters | 25 | 2 |
| Retained clipped gradients | 25 | 0 |
| Optimizer tensors | 75 | 0 |

Layer-zero QKV bias fails three entries at 680/786/965, max absolute 0.0064439;
layer one fails index 603, max 0.0067420. All lie in the key-bias section 512:1024.
An independent loop using the old assertions fails exactly those two keys.
Thus admission is unchanged, while the user can see the complete failure.
Passing final optimizer tensors within tolerance does not prove matching
optimizer histories. The [underlying sensitivity](adam-epsilon-sensitivity.md)
and [unsuccessful cast repair](bf16-cast-preservation.md) remain separate findings.

## Reproduce

Run from the repository root with a host PyTorch environment for CPU tests,
and an authenticated Modal client for the pinned A100 diagnostic:

```sh
python3.13 -m pytest -q tests/test_torchfront_numerics.py
PYTHONPATH=. python3.13 -m exp.weaknesses.bf16_verification \
  --out exp/weaknesses/reproduced-bf16-fixture.json
modal run exp/weaknesses/bf16_verification_modal.py \
  --out exp/weaknesses/reproduced-bf16-verification.json
```

The CPU fixture deliberately perturbs synthetic bias/optimizer entries; it is
not a training measurement. The GPU driver retains the original seed 1729,
batch 8, sequence 128, width 512, eight heads, two layers, vocabulary 512,
BF16 autocast,29 updates, AdamW and `rtol=0.05, atol=0.005` gates.
It also runs an independent original-assertion loop. These are diagnostic runs;
failed numerical comparisons remain failures even if the driver successfully
saves a report.

## Source and evidence

- [Worker comparison and structured error](../../../openassembler/torchfront/worker.py),
  [numerical regression tests](../../../tests/test_torchfront_numerics.py).
- [CPU fixture](../../../exp/weaknesses/bf16_verification.py),
  [recorded synthetic fixture](../../../exp/weaknesses/bf16_verification_cpu.json).
- [Pinned GPU driver](../../../exp/weaknesses/bf16_verification_modal.py),
  [127-tensor A100 result](../../../exp/weaknesses/bf16_verification_a100.json).

The worker only checks observed model outputs and final retained state for the
configured workload. Full intermediate/trajectory equivalence, unobserved
branches and a BF16 accuracy repair remain outside this feature.

## Experimental per-update trajectory: November 9

The [new experiment-only ledger](../../../exp/whole_stack/trajectory_gate.py)
requires an explicit ordered checkpoint list. Missing, duplicate or reordered
checkpoints cannot produce a pass. It clones named parameters, registered
buffers, explicitly missing gradients, raw or clipped gradients, all optimizer
state entries and parameter-group settings, using parameter names rather than
process-local IDs. The caller supplies the existing numerical comparator; the
ledger changes neither arithmetic nor tolerances. Tensor snapshots are released
after each comparison and compact diagnostics are retained. This copy-heavy
path is excluded from timing and is not integrated into native graph dispatch.

The [A100 BF16 application diagnostic](../../../exp/whole_stack/first_divergence.py)
uses eager PyTorch and the same AOTAutograd/Inductor training route as the
earlier study. It observes initial state plus five boundaries per update:
separately computed full pre-update logits, original forward loss, raw
backward, clipped backward and post-AdamW state. The first 29 batches match
the old diagnostic; update 30 repeats its final batch. Seed 1729, epsilon
1e-8, precision and `rtol=0.05, atol=0.005` gates remain fixed.

The [30-update raw result](../../../exp/whole_stack/runs/bf16-first-observed-divergence-20261109T111158.926633Z.json)
records all 151 required boundaries on an A100 80GB PCIe with PyTorch
2.7.1/CUDA 12.8. Inputs remain unchanged. The trajectory fails 53 checkpoints;
this counts repeated observations, not 53 distinct broken operators.

| Observed group | First boundary outside its gate | Finding |
|---|---|---|
| Full logits | Before update 3 | 2/524,288 BF16 elements fail; relative L2 0.002881 |
| Parameters | After update 22 | Two QKV bias tensors fail one key-bias entry each, at indices 965 and 537; max absolute 0.005915 / 0.005558 |
| Raw and clipped gradients | None in the 30-update record | Inside the declared gate, not bitwise equal |
| Adam moments, step counters and settings | None in the 30-update record | Passing stored state does not prove an identical optimizer trajectory |

At update 30, parameter failures remain in the same QKV tensors, with 5 and 2
entries outside the gate. The observed full-logit failure at update 3 is not
proof of the first causal arithmetic divergence: no every-operation comparison
was performed, and the extra full-logit path is a separate diagnostic forward.
The result is **not native OA execution**, does not repair the BF16 failure,
and supplies no performance result. Native composition still needs this style
of check with its own exact route and numerical contract. RNG histories,
unobserved Python attributes and arbitrary intermediate activations are outside
the snapshot scope.

```sh
modal run exp/whole_stack/first_divergence.py --steps 30
python3.13 -m pytest -q exp/whole_stack/test_trajectory_gate.py \
  tests/test_torchfront_numerics.py
```

The 11 new ledger tests and 18 existing application-state tests pass locally.
The first Modal attempt failed before training because a flat remote import
could not resolve the local `exp` package; the standalone image definition
fixed that startup problem. That attempt supplies no numerical evidence.
