# Typed low64 multiplication: research to production

Status: production integrated; complete source-exact evidence retained. General G03/G11 and the original 51-gap program remain open.
The canonical [wide scalar dossier](../../../../docs/compiler/gaps-wide.md)
contains the user-facing contract.

## Gap and solution

The [original unsigned CUDA kernel](cuda.cu) emits `mul.lo.s64`; production
rejected that form. The earlier [backend-only experiment](../README.md) passed
760 A100 launches but left typed Python multiplication explicitly rejected.
This cycle implements both source routes. `baseline-*.py` retain the preceding
production frontend, selector and scalar admission sources.

The exact candidate uses four recovered IMAD/IMAD.HI operations. For
`a = a0 + 2^32*a1` and `b = b0 + 2^32*b1`, the low word is `low32(a0*b0)` and
its high word is `high32_unsigned(a0*b0) + low32(a1*b0) + low32(a0*b1)` modulo
2^32. Signed and unsigned low64 products have the same bits. Fresh result
binding preserves input aliases, snapshots and false guards. No table or new
encoding field changes. Python `i64/u64` now emits the exact `mul.lo` form for
`*` and `*=`, retaining explicit fixed-width casts and mixed-type diagnostics.

## Experiments and validation

- [Baseline](baseline.json): all 35 source configurations reject, including
  sixteen Python routes. [Baseline squares](baseline-squares.json) retain four
  additional `x*x` source/parameter rejections.
- [Offline gate](candidate-offline.json): 2,520 complete PTX/normal/poisoned SASS
  executions. [Square gate](candidate-squares-offline.json): 288 more. References
  use independent arbitrary-precision products modulo 2^64; full guards, input
  immutability and inactive output tails are compared. Sources cover both integer
  types, register/immediate/parameter values, both aliases, outer/negated guards,
  Python arrays/literals/augmented assignment/helpers/loops/casts and squares.
- [Toolkit audit](oracle-audit.json): all 35 original PTX/CUDA/Python sources
  accepted by actual ptxas, with 157 raw artifact hashes. The
  [square toolkit audit](oracle-squares-audit.json) accepts four more and retains
  twenty raw artifact hashes.
- [Base A100 gate](candidate-a100-20261123T212929Z.json): 1,400 launches.
  [Square A100 gate](candidate-squares-a100-20261123T213402Z.json): 160 launches.
  Each uses original/same-emitted-PTX ptxas controls plus four OA levels, two
  seeds, lengths 0/1/259/384 and three 128-thread CTAs. Exact source/PTX/cubin/
  disassembly/manifest/input/output archives are hashed by each report.
- [Production base audit](production-base-audit.json) and
  [production square audit](production-squares-audit.json): complete independent
  byte/reference, source custody, frontend regeneration, native words/registers,
  zero local frames and ownership. The [combined audit](promotion-audit.json)
  retains both gates separately: **1,560 launches, 15,974,400 bytes, 156 native
  comparisons**. No tolerance or timing is pooled. Corrupted-output controls reject.
- [Focused regressions](targeted-suite.log): **140 pass**, including the existing
  typed wide suite. The stale `x*x` rejection becomes a real unsupported-power
  assertion; accepted square behavior has independent golden cases and hardware.
- [Frontend continuity](frontend-continuity.json): fifteen earlier source files,
  62 exact unchanged PTX kernels and unchanged rejection diagnostics.

Both full suites pass: **3,465 supported tests** (Python 3.12.10/Torch 2.7.1,
zero skips) and **2,980 compiler-only tests with fifteen skips** (Python 3.14.5,
no Torch). The [installed wheel audit](wheel-audit.json) checks 129 source/data
identities, 156 complete native outputs and eighty exact Python PTX regenerations.
The [policy audit](../../foundations_20261121/audit-20261123T213840Z.json) retains
all 49 policies and 98 original sessions. The [prior-native replay](application-continuity.json)
passes **1,934 earlier kernel occurrences** with exact current words/registers,
binding all preceding numerical/ownership proofs and archives. The
[final integration record](validation.json) records exact final sources, suites,
independent numerical/native evidence, retained failures and live/exported docs.

`first-square-audit-path-failed.log` retains a harness invocation with a guessed
report timestamp; the corrected invocation uses the actual saved report. It
never modifies or reinterprets a hardware result. Actual toolkit aliases such
as `IMAD.SHL` are retained in native listings; exact words and full bytes pass.
Production imports no experiment modules. This is a correctness result, with
no claim of FP64 arithmetic, wide/high products, division/remainder, arbitrary
Python arithmetic or improved speed.

```sh
python compiler_exp/gap_closure/wide_multiply/promotion/offline.py --candidate
python compiler_exp/gap_closure/wide_multiply/promotion/offline_squares.py --candidate
python compiler_exp/gap_closure/wide_multiply/promotion/oracle_probe.py
python compiler_exp/gap_closure/wide_multiply/promotion/audit_oracle.py
modal run compiler_exp/gap_closure/wide_multiply/promotion/run_modal.py
modal run compiler_exp/gap_closure/wide_multiply/promotion/run_squares_modal.py
python compiler_exp/gap_closure/wide_multiply/promotion/audit.py compiler_exp/gap_closure/wide_multiply/promotion/candidate-a100-20261123T212929Z.json /tmp/low64-base-audit.json
python compiler_exp/gap_closure/wide_multiply/promotion/audit_squares.py compiler_exp/gap_closure/wide_multiply/promotion/candidate-squares-a100-20261123T213402Z.json /tmp/low64-squares-audit.json
python -m pytest -q tests/test_backend_wide_multiply.py tests/test_pyfront_wide.py
```
