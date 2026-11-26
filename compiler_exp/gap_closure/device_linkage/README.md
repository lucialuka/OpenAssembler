# Separate CUDA translation-unit device linkage baseline

Status: retained original research baseline; explicit scalar source-library
resolution has since been validated and promoted in the
[promotion study](promotion/README.md). General native device linkage remains open.

The [direct scalar function dossier](../../../docs/compiler/gaps-device-functions.md)
implements one-module static specialization. This next G16/G42 baseline starts
with two actual CUDA translation units: `entry.cu` declares a device helper;
`helper.cu` defines it. `oracle_probe.py` invokes the original CUDA 12.8 commands
for relocatable PTX/device objects and then `nvcc -dlink -cubin`.

`oracle-probe.json` records source/command/tool-log/artifact hashes and the exact
production compiler inputs. Local toolkit compilation/linking succeeds. The
linked cubin retains two executable sections, `.text.linked_probe` and
`.text.helper`. The entry's standalone PTX instead contains an unresolved
`.extern .func` declaration. The recorded pre-promotion OA rejects it explicitly with
`unresolved device function helper` before generation. No donor helper bytes
may silently become native OA-owned execution.

This observation defines the next experiment: resolve cross-module scalar
symbols and original dependency identities, compare original linked controls
with an OA implementation at complete output/resource/ownership boundaries,
and retain recursion/duplicate/ABI/visibility/unresolved rejections. A native
linker, whole build graph and arbitrary libraries cannot be inferred from one
successful toolkit command. Raw `.ptx`, `.o`, `.cubin`, `.sass` and all logs
remain beside the report.

```sh
python compiler_exp/gap_closure/device_linkage/oracle_probe.py
```

## Isolated solution and independent A100 controls

`merge_probe.py` tests this exact two-source shape by resolving its external
prototype to the visible definition. The first attempt changed the prototype
to bare visibility; actual ptxas rejected the mismatch. Its source and logs
remain in `failed-visibility-*`. Matching visible prototype/definition passes
independent complete CPU output/guard checks, all four native encodings and
local toolkit acceptance. This script deliberately asserts one known symbol;
it is not a production linker or a general source parser.

`research-a100-20261123T182610Z.json` passes **48 launches** on actual
A100-SXM4-40GB: original separate-unit nvcc objects/device link, merged-source
ptxas, and OA safe/mined/sched/opt, each at four active lengths and two seeds.
The worker builds original `.cu` inputs and both linked/merged controls itself.
Three CTAs of 128 threads compare every output/guard/input byte with independent
wrapping32 arithmetic. Original and merged-control binaries remain distinct;
this is not an assertion of equal executable layout or resources.

`research-audit.json` checks **196,608 complete retained bytes**, four exact
OA-native word/register comparisons, original sources, worker-emitted and
merged PTX, module ownership and archive/harness/source identities. Actual
worker Python 3.12.1/CUDA 12.8.1 toolchain metadata are retained. No performance
claim is made. `run_modal.py` and `audit.py` reproduce these research gates.

Before production promotion, this exact-source hypothesis still needs general
module/visibility/signature resolution, deterministic duplicate/unresolved
rules, original library/dependency custody, malformed/unsupported diagnostics,
normal source/CLI transport integration and full resource/ownership/continuity
validation. This research does not establish native object relocation or
arbitrary device-library ABI support.

The subsequent [promotion](promotion/README.md) implements explicit source
libraries through normal CLI/remote builds, deterministic symbol rules and
per-unit dependencies after a separate 576-launch A100 gate. The raw baseline
and research-only observations above keep their original meanings.
