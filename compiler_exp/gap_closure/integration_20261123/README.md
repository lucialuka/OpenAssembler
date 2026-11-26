# November 23 foundation integration

The current cycle adds bounded GPU ordered/shared atomics, repairs two wide-value
lifetime defects, and integrates the earlier wide Python/buffer/primitive/ABI/
build/optimizer/checkpoint work into the supplied 51-gap register. Original
exit gates remain intact; broad gaps stay partial or open.

## Source and execution checks

- [Supported suite](supported-suite-final.log): 2,710 passed, no skips,
  Python 3.12.10 / Torch 2.7.1.
- [Compiler-only suite](compiler-only-suite-final.log): 2,253 passed,
  14 explicit skips, Python 3.14.5 without Torch.
- [Policy continuity](../foundations_20261121/audit-20261123T082155Z.json):
  all 44 GEMM and five rowwise policies match frozen HEAD and 98 retained native
  executables, including full controls, labels, exits and register counts.
- [Atomic ordering dossier](../../../docs/compiler/gaps-atomics.md): two final
  candidate gates pass 472 and 768 launches on separately identified A100
  variants. Complete retained bytes and current native generation are audited.
- [Fresh checkpoint/optimizer audit](checkpoint-final-audit.json): 28 current
  hardware bundles, 383 native comparisons and 17 exact checkpoint byte pairs.
  The full gate passes 7,862 comparisons / 124,438 elements and separately reruns
  the optimizer and ordinary CLI policies. The IEEE gate adds 80 comparisons /
  1,360 elements on another A100-SXM4-80GB worker.

The [validation record](validation.json) binds logs, source and evidence hashes.
The [gap register](../../../docs/compiler/gap-register.md) links each canonical
dossier; experiments and failures remain under their feature directories.
No performance claim follows from these correctness gates.

## Interrupted and rejected observations

The initial [supported run](supported-suite-first.log) was intentionally
interrupted after 799 passes to avoid overlap with dispatch-table rebinding;
it is not a completed suite. The stable run above started after the audit.
The [old checkpoint executable check](checkpoint-continuity.log) rejected
binary reuse after the selector fix. Fresh hardware was required and passed;
the old result was not relabeled current. The first
[checkpoint transport](checkpoint-hardware.log) lost its localhost proxy;
the [retry](checkpoint-hardware-retry.log) ran without those proxy environment
variables and saved a complete successful report/archive. A failed transport
without a retrieved report is not hardware evidence.

```sh
python -m pytest -q --disable-warnings
python compiler_exp/gap_closure/checkpoints/audit.py --output /tmp/oa-checkpoint-audit.json
python tools/check_docs.py
```

Policy rebinding is an explicit audited operation. Reproducing its source
continuity driver needs the named frozen Git commit and original retained
policy archives; ordinary compilation performs no hidden search or remeasurement.
