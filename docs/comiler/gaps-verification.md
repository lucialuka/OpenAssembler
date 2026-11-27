# Exact-division verification and reproducible emulator samples

Status: implemented

## Gap and proposed solution

G01 retains a fixed mismatch for numerator `0x8147ec48` and denominator
`0x21dea823`. The frozen ptxas SASS emulator produces `0x9ee5dc9c`; the PTX
interpreter and a separate integer-ratio binary32 round-to-nearest-even reference
both produce `0x9ee5dc9d`. The initial quotient is correct; the correction
residual rounds to the smallest positive subnormal, and the final FMA changes
the quotient by one ULP. The emulator's hypothesis for `FCHK` selects the fast
path for these inputs. This offline trace does not establish the hardware's
`FCHK` predicate or MUFU approximation.

The closure experiment replays the frozen cubin, fresh same-source ptxas and
OA code on A100. Two frozen-binary controls force the fast and slow paths;
a third captures the actual `FCHK` predicate. A separately written integer-ratio
reference rounds the exact quotient without calling emulator/interpreter math.
The corpus includes the original failure, adjacent mantissas, exponent sweeps
and fixed-seed finite random operands. The hardware gate checks complete
outputs, input immutability and output guards; it makes no timing claim.

G51 additionally used process-randomized `hash(probe)` values for emulator
sampling. The maintained regression now derives a stable 64-bit SHA-256 seed
from each probe name and prints seed, trial, lane, all inputs and scalar
parameters when it fails. The corpus test's numerical comparison is unchanged.

## Validation and production repair

The [initial A100 report](../../compiler_exp/gap_closure/verification/a100-20261120T152605Z.json)
identifies Python 3.12.1, CUDA 12.8.1 and an NVIDIA A100-SXM4-40GB. All 3,309
finite quotient cases pass for the frozen cubin, same-source fresh ptxas, OA
safe and forced-slow implementations. The forced-fast negative control has 707
wrong outputs, including the retained counterexample. The predicate diagnostic
reports hardware `FCHK=true` for that case. Guards and input immutability pass
for every variant. The diagnostic's low output word replaces a quotient
correction and is deliberately excluded from numerical implementation gates.

This isolates the retained failure to the emulator's slow-path decision. The
experimental small-dividend guard fixes all nine emulator mismatches in the
same 3,309-case corpus. Production `FCHK` now selects the slow path below
dividend exponent -102; the other existing exponent/special checks remain.
The [current emulator replay](../../compiler_exp/gap_closure/verification/local-heldout-validation.json)
compares every quotient with the independent exact reference and retains all
outputs. An always-running [frozen SASS fixture](../../tests/data/emu/p_div_rn_f32.sass.txt)
supports fixed failure and adjacent-value regressions even without generated
corpus directories. Normal and poisoned-register runs agree. The emulator,
backend-arithmetic and generic-verifier checks pass 212 tests locally.

This closes the named G01 counterexample and its retained neighbors. It does
not reconstruct every hardware `FCHK` decision: the initial replay still has
nine missed hardware-slow decisions and four extra emulator-slow decisions,
with correct final quotients in each. MUFU remains approximate in the emulator.
The guard is an observed repair within the measured domain, not a proof of
complete ISA conformance.

The [held-out A100 replay](../../compiler_exp/gap_closure/verification/a100-20261120T153212Z.json)
extends the corpus to 3,409 cases and runs OA `safe`, `mined`, `sched` and `opt`.
All four OA levels, fresh same-PTX ptxas, the frozen cubin and forced-slow
control match every exact quotient; guards and input immutability pass. The
100 new boundary controls vary both signs, exponents -104 through -100,
minimum/maximum and neighboring mantissas, and two denominators. Their hardware
predicates agree with the new -102 guard. The current emulator also matches
all 3,409 quotients. The remaining 13 predicate differences are confined to the
earlier corpus and do not change these division outputs.

## Experiments and retained evidence

All new drivers and outcomes are in
[compiler_exp/gap_closure/verification](../../compiler_exp/gap_closure/verification/).

- [Offline replay driver](../../compiler_exp/gap_closure/verification/offline.py)
  and [pre-change trace](../../compiler_exp/gap_closure/verification/offline-before.json)
  retain the fixed failure and its instruction-level state; the
  [post-change trace](../../compiler_exp/gap_closure/verification/offline-after.json)
  retains the corrected slow path.
- [Independent reference and corpus](../../compiler_exp/gap_closure/verification/probe.py)
  define the exact rounding and sample domains.
- [A100 driver](../../compiler_exp/gap_closure/verification/run_modal.py)
  uses CUDA 12.8.1 and Python 3.12. Diagnostic cubins have explicit labels and
  must never be counted as numerical implementations.
- [Emulator validation driver](../../compiler_exp/gap_closure/verification/validate.py)
  verifies the hardware archive hash and compares its entire frozen input corpus.
- [Initial harness startup failure](../../compiler_exp/gap_closure/verification/startup-failure.json)
  records a remote module-path error before any hardware measurement.
- [Sample reproducibility](../../compiler_exp/gap_closure/verification/deterministic-seeds.json)
  checks processes with `PYTHONHASHSEED=1` and `9173`. The 136 emulator tests
  passed before the fixed-counterexample regressions were added. These seed
  checks are independent of the numerical G01 hardware gate.

## Reproduction and boundaries

From the repository root:

```sh
.venv/bin/python compiler_exp/gap_closure/verification/offline.py /tmp/oa-g01.json
.venv/bin/python -m pytest -q tests/test_emu.py
modal run compiler_exp/gap_closure/verification/run_modal.py
.venv/bin/python compiler_exp/gap_closure/verification/validate.py compiler_exp/gap_closure/verification/a100-20261120T153212Z.json /tmp/oa-g01-validation.json
```

The offline replay requires the frozen generated PTX/SASS corpus. The hardware
replay additionally requires the frozen cubin, authenticated Modal access and
an A100 allocation; it stages compiler sources and retains their hashes. These
checks address the named finite FP32 division forms and deterministic samples,
not all IEEE operations, arbitrary SASS semantics, native division performance,
or general compiler support.

General FCHK/MUFU conformance and scalar/backend coverage remain open beyond
these gates.
