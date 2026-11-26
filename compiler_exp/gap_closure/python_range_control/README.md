# Finite range counter overflow: next G06 cycle

Status: retained initial watchdog/guard study. The subsequent
[original-source cycle](promotion/README.md) promotes finite range exhaustion and
runtime loop else with separate original-source/hardware/integration validation.

Constant nonunit i32 range steps can wrap before the loop comparison observes
exhaustion. A mathematically finite range near INT_MIN/INT_MAX then repeats
unexpected values. This is the fixed-width counter risk in the supplied G06 exit
gate. Dynamic steps and range/while else remain separate open control contracts.

The probe uses an explicit eight-trip watchdog, preserving a finite safe test even
for the buggy lowering. Independent Python range values determine complete sum,
trip count and last-target bytes. Four boundary cases cover positive/negative
overflow, INT_MAX positive step and INT_MIN negative step; ordinary and empty
ranges act as capability controls. PTX and encoded emulators share the faulty
frontend here, so agreement between them alone is insufficient.

The isolated solution checks whether the mathematical next counter would leave
i32 before applying a nonunit step. Because start/stop are constrained to i32,
such a value has necessarily crossed stop. Branching to exhaustion preserves the
last target and avoids wrapping without changing the admitted source types or
raising resource limits. Unit steps retain their established lowering.

Original helper/caller/nested control, overflow-safe snapshots, exhaustion else,
diagnostics, source/native ownership, actual toolkit and independent A100 gates
are still required before any promotion. No experiment code enters production.

```sh
python compiler_exp/gap_closure/python_range_control/replay_baseline.py
python compiler_exp/gap_closure/python_range_control/candidate.py
python compiler_exp/gap_closure/python_range_control/probe.py --candidate
```

The [baseline](baseline.json) reproduces four counterexamples and two controls
in 72 bounded emulator observations; the [candidate](candidate-offline.json)
corrects all 72. The final [A100 record](candidate-a100-20261124T060602Z.json)
and [audit](candidate-audit.json) retain 30 candidate numerical passes,
16 baseline OA counterexamples, eight baseline OA controls and six explicitly
observational baseline ptxas results, with 9,120 audited bytes and 48 exact native
outputs. The first [failed A100 attempt](candidate-a100-20261124T060330Z.json)
assumed ptxas would reproduce the baseline wrap; it instead matched the Python
reference. All six retained ptxas results match Python while OA still reproduces
four boundary counterexamples. The toolkit/native/emulator divergence remains
unresolved and is not hidden by a relaxed candidate gate or universal PTX claim.
The [canonical dossier](../../../docs/compiler/gaps-range-control.md) retains the
remaining promotion gates. Production is unchanged.

After promotion, [frozen replay](baseline-replay.json) checks all 72 exact original
watchdog observations without rewriting their report. The initial `probe.py`
remains retained as the original harness; baseline reproduction uses the frozen
replay command above, since current production now corrects the counter defect.
