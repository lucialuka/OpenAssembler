# Comparison-chain evaluation: next G06 research cycle

Status: retained first-stage condition-only hypothesis. The subsequent
[original-source cycle](promotion/README.md) promotes validated typed/effectful
chains and original expression/source/native ownership.

The initial production frontend rejected original `a < b < c` conditions. Expanding them to two source
comparisons duplicates the middle evaluation, which can change atomic effects or
read snapshots. The isolated candidate evaluates the left and first comparator,
stores the intermediate value once, branches out on false, then evaluates the
next comparator only on the surviving path. Existing typed comparison and branch
polarity rules remain in use; no predicate Boolean ISA operation is introduced.

The probe has separately computed outputs and per-thread atomic counters. It
checks ordinary/negated chains, skipping a final atomic when the first comparison
is false, and evaluating a middle atomic once. All input/output/counter/guard
bytes are compared through four PTX/normal/poisoned levels. This is finite CPU
evidence only. Original rejection, complete mixed-type/literal/NaN/infinity/zero
semantics, nested helper/while/and/or combinations, source/build/native ownership,
independent toolkit and A100 gates remain required before production. First-class
Boolean values and arbitrary effectful Python expressions remain separate work.

```sh
python compiler_exp/gap_closure/python_comparison_chains/candidate.py
python compiler_exp/gap_closure/python_comparison_chains/probe.py
```

The [baseline](baseline.json) retains the original production rejection.
The [atomic probe](candidate-offline.json) passes twelve complete executions;
the [typed domains](domains.json) pass sixty more, including NaN, infinities,
signed zeros, wide boundaries, not/unequal/while/and combinations and five actual
ptxas controls. The [A100 gate](candidate-a100-20261124T111942Z.json) passes thirty
original condition-chain launches (six source cases, ptxas plus four OA levels).
The [independent audit](candidate-audit.json) checks 11,080 full input/output/
atomic-counter/guard bytes and 24 exact native word/register outputs, original
source identity/native ownership and corrupted-output rejection. These finite
results do not promote the candidate. Mixed contextual literals, broader nested
effects, inactive tails/multiple CTAs, dependency/source graph, normal remote and
wheel/public interface regression gates remain open.

The [canonical dossier](../../../docs/compiler/gaps-comparison-chains.md) retains
the current experimental contract, evidence and outstanding promotion gates.

After promotion, [frozen rejection replay](baseline-replay.json) checks the
original baseline using its frozen compiler without overwriting historical
reports or asserting that current production still rejects chains.
