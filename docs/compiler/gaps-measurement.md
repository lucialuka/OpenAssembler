# Compilation and cache attribution

Status: implemented

G47 found that four coarse model-worker times hid native generation, toolkit
startup/container construction, packaging, disassembly and cache validation.
The solution records actual wall intervals around those operations. Attribution
is a measurement feature; it does not change generated instructions or establish
a compile-speed improvement.

## Contract and production path

[Source builds](../../openassembler/build.py) attach `compile_stage_ms` and
`compile_total_ms` to each manifest entry. The non-overlapping stages are source
hashing, frontend conversion, PTX parsing, backend generation, verification,
ptxas/container construction, packaging/ownership and disassembly. A named
`unattributed` remainder closes the source wall-time accounting. An omitted
operation has zero time; failed operations retain elapsed time before the
failure. Backend generation still groups selection, allocation, scheduling and
encoding. Toolkit intervals include tool availability/startup and the Colima
hop where applicable. Output publication, discovery and transport lie outside
this per-source total.

[Model caching](../../openassembler/modelfront/model_cache.py) reports identity
and source generation, lock wait, entry validation, restore/weight save and
validation, native build/weights, publication/validation, and cleanup/validation.
On a miss, `source_builds` retains nested scalar-module source timings before
the donor manifest is removed. On a hit it is empty and native-build time is
zero. Historical cold-build times are never replayed as current cache-hit work.
File inventory and unmeasured interpreter overhead remain outside these named
cache intervals; the enclosing worker bundle interval includes them.

The [model worker](../../openassembler/modelfront/worker.py) additionally records
source unpacking, CPU example generation, source metadata, repository discovery,
independent native builds and optional application validation, alongside the
existing factory/capture/planning/bundle times. `compile_worker_total_ms` bounds
them. Application validation can contain GPU execution when explicitly requested;
it is separate from compiler work. Cache phases are nested in the bundle
interval and source phases are nested in native build. Summing all three levels
would double-count work.

These records do not yet attribute request transport, remote image/container
startup and queueing, final artifact archive/publication, each internal compiler
pass or separate tensor-recipe builds. Those remaining G47 exit gates stay open.

## Experiment, validation and limits

The [before probe](../../compiler_exp/gap_closure/measurement/before.json) records
an actual binary-only compile without attribution. The [after probe](../../compiler_exp/gap_closure/measurement/after.json)
records the same source and the new phases. Both deployed cubins have SHA-256
`623c91d119019f6b6332f8161a7673f2390a6a0cfb726563681689daa487d07e`.
The single samples describe attribution, not a latency comparison.

The [cache experiment](../../compiler_exp/gap_closure/measurement/cache-results.json)
retains three independent cold directories followed by validated warm hits,
all phase samples and binary identities. Every cold/hit pair keeps the same
native cubin. The source-defined seven-element ReLU plan includes backward;
no GPU execution or timing is inferred from this compilation experiment.
[Regression tests](../../tests/test_compile_measurements.py) additionally check
failure attribution, non-overlapping source totals, skipped disassembly,
cache-hit versus miss work, and propagation through the ordinary model worker.
The four new tests and 48 related source/cache/runtime tests pass in Python
3.12.10, Torch 2.7.1 on macOS 15.6.1 arm64 with CUDA 12.8 tools through Colima.

The [continuity audit](gaps-continuity.md) verifies every installed GEMM and
rowwise policy against frozen generation and retained selection/heldout binary
artifacts before rebinding the tables' compiler identities. Measurement values,
PTX contracts, numerical gates and selected policies remain their original
research evidence.

From the repository root:

```sh
python compiler_exp/gap_closure/measurement/probe.py /tmp/oa-source-phases.json
python compiler_exp/gap_closure/measurement/cache_probe.py
python -m pytest -q tests/test_compile_measurements.py tests/test_ptx_entries.py tests/test_model_cache_tools.py tests/test_model_runtime.py
```

The probes use exclusive output creation to preserve retained records. Copy a
probe or select a new record path when rerunning the cache experiment. Compiler
source hashes identify the exact reporting code used by each stored result.
This work supplies no GPU performance comparator or complete G48/G49 benchmark.
