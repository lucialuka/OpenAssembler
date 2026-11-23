# November 21 integration of the supplied-gap research

This continues the original 51-gap scope through four bounded production
extensions. It does not close the full program. Earlier experiments, wrong
results, rejected fixtures, stopped audits and superseded successful records
remain beside their corrections.

Final [validation](validation.json): **1,686 passed / zero skips** in the
supported Python 3.12.10 / Torch 2.7.1 environment; the separate compiler-only
Python 3.14.5 run passes **1,294 with 12 explicit skips**. Final source hashes,
raw logs, hardware report/archive identities and both continuity audits are
bound in that record.

| Area | Production result | Independent evidence |
|---|---|---|
| G11/G12 memory and predicate values | Global v2 loads, v4 stores, scalar byte/halfword/64-bit stores; predicate homes; packed/unpacked snapshots and binary64 immediates | Final [memory report](../memory_widths/a100-20261121T124129Z.json): 224 launches / 458,752 output bytes; OA four levels and same-PTX ptxas |
| G04/G06 helper control flow | Typed pure scalar helper branches/range/while/break/continue with conservative initialization and terminal return | [Final helper report](../python_control/hardware-20261121T124655Z.json): 15 launches / 960 exact elements, guards/immutability |
| G21 model operators | FP32 sigmoid and static rank0..4 sum/mean with exact axis/layout/derivative contracts | [Operator report](../operators/hardware-20261121T124125Z.json): 416 tensor comparisons / 30,576 elements, 100 ownership audits; CLI compile/saved run each 35 checks over five SGD updates |
| G48 repeated profiling | Configurable final verification iterations, per-contract samples/distributions/resources, separately labeled profiler-free step arrays | [Profiler report](../profiling/hardware-20261121T124038Z.json): inference/direct/graph/default/unverified routes, 108 audited launches, 96 reference checks; no extra training updates |

The canonical [gap register](../../../docs/compiler/gap-register.md) links each
contract dossier. The current scalar ledger has 496 representative forms, all
selecting, with 490 register-backed and 482 parameter-backed encodes; its
numerical/toolkit/hardware evidence is explicitly separate from the named
28-case memory hardware corpus. No syntax/encoding count is a percentage of
arbitrary workloads.

## Frozen code and evidence custody

[compiler-before.zip](compiler-before.zip) and its [identity map](compiler-before.json)
retain the compiler at the start of this continuation, including the earlier
uncommitted repairs. The input repository and work were not reset or committed.

The [final policy audit](audit-20261121T124658Z.json) proves unchanged native
words/controls/labels/registers/exits for all 44 GEMM and five rowwise policies
against frozen sources and 98 original hardware-session artifacts. Dispatch
identities advance only after those checks. Historical policy choices, times,
contracts and source/binary evidence remain unchanged.

The model-native continuity audit records current source and archived execution
identities for 106 bundles / 866 kernel occurrences, including complete
instruction controls, registers and EXIT metadata. This permits reuse of the
named earlier hardware gates after the final immediate-only backend repair;
it does not assert that new hardware or profiling was performed. The final
validation record links the latest immutable audit and exact source hashes.

All actual hardware gates use NVIDIA A100-SXM4-40GB, Python 3.12.1 and CUDA
toolkit 12.8.1. Model gates use Torch 2.7.1+cu126 / CUDA runtime 12.6.
The supported local suite uses Python 3.12.10 / Torch 2.7.1 on macOS15.6.1
arm64 with toolkit calls through Colima. The Python 3.14.5 environment has no
Torch and is recorded separately with explicit skips. No performance ratio is
claimed by the correctness, identity or profiling-accounting gates.

## Retained failures

- [Initial full suite](supported-suite-initial.log): 1,652 pass / 13 failures,
  all installed-policy identity checks during backend changes, before rebinding.
  This is not a final stable-source validation.
- [Audit correction](continuity-review.json): the first audit cached the current
  tensor selector while swapping scalar modules; it was stopped before table
  publication. The next attempt exposed an immutable configuration-class
  mismatch, retained in its failure log. The corrected driver reloads all frozen
  modules and reconstructs the matching configuration type.
- Another audit was stopped after review demonstrated packed/unpacked aliases
  changed supposedly saved values. Independent [reviewer proofs](../memory_widths/reviewer-snapshot-negative.json)
  retain exact before-code and expected/actual bytes. The selected candidate
  fixed both, passed hardware, and was then extended to legal binary64 literal
  operands after an independently retained internal-error reproducer.
- Memory candidate/toolkit fixture failures and earlier successful candidates
  remain in `memory_widths/`; they are not substituted for the final 28-case
  gate. Updated capability tests retain genuine unsupported rejection controls.

## Reproduction

Run from the repository root with the supported Python/Torch dependencies:

```sh
python -m pytest -q
python compiler_exp/gap_closure/memory_widths/validate.py compiler_exp/gap_closure/memory_widths/a100-20261121T124129Z.json /tmp/oa-memory-audit.json
python compiler_exp/gap_closure/integration_20261121/hardware_continuity.py --help
python tools/check_docs.py
```

`continuity.py --update` is an explicit identity mutation and requires every
retained policy comparison to pass first. Without `--update` it is read-only.
The old reporting-only continuity/426-form audit scripts require their earlier
recorded sources; their strict drift checks must not be bypassed or their raw
snapshots regenerated over the new compiler state.
