# Python versus a compiled host implementation

Status: untested-or-hardware-unavailable

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Measure whether parsing, optimization, encoding or autotuning takes enough host time to matter, then optimize that specific work. Host implementation language is separate from the native code executed by the GPU.

## Findings and implementation decision

The [compilation attribution contract](../../compiler/gaps-measurement.md) now
records source-build, cache and model-worker phases with actual cold/hit
samples and unchanged native binaries. Remote startup/transport and individual
backend-pass costs remain outside its scope. These measurements provide no
evidence for a language rewrite or compile-speed improvement.

OpenAssembler is Python tooling that emits/loads native cubins; the A100 executes SASS, not a Python interpreter per HMMA instruction. No complete compile-stage profile has shown that a C++ rewrite would materially improve application runtime. No rewrite was made.

Frequent JIT, uncached startup and large autotuning searches can make host latency important, but that requires evidence. A whole-language migration could change code generation and introduce correctness differences without improving the measured GPU bottleneck.

## Method and validity checks

Proposed profiling separates frontend parsing, graph construction, allocation, scheduling, encoding, ptxas/container invocation, cold/warm cache behavior and total application startup. Compare identical generated instruction streams and device workloads before/after any host optimization.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
.venv/bin/python -m openassembler compile examples/vector_add.py --out-dir output/host-profile-control
.venv/bin/python -m pytest -q tests/test_cli_compile.py tests/test_integration.py
```
These commands provide a reproducible current compile control; they do not constitute the missing stage-by-stage latency study. Use a profiler in a new `exp/` driver and record cold/warm runs explicitly. There is no production C++ backend switch.

## Code and retained evidence

- [Build orchestration](../../../openassembler/build.py)
- [Native tensor backend](../../../openassembler/backend/tensor.py)
- [Output publication/manifest code](../../../openassembler/project.py)
- [CLI integration tests](../../../tests/test_integration.py)

## What remains open

If a measured hot loop dominates, optimize its data structures or move just that loop to compiled code. Keep device code and precision/performance comparisons identical, and report host startup separately from steady GPU execution.
