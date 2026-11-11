# Adaptive shared-memory budgeting

Status: untested-or-hardware-unavailable

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Choose how much shared memory to devote to each operand, pipeline stage and epilogue for a particular shape. The goal is enough overlap and reuse without consuming the capacity needed for more active CTAs.

## Findings and implementation decision

The production native GEMM has two checked fixed-layout families: a
128×128×64 CTA tile with 64 KiB dynamic shared memory and a 64×64×64 tile with
32 KiB. Both use 128 threads and two shared-memory stages. The current
[exact shape selector](shape-aware-native-gemm-dispatch.md) can choose between
these families; 22 of the 42 independently replayed contracts select tile64.
That is bounded selection of complete kernels, not a general memory allocator.

A measured policy that independently chooses operand, stage and epilogue
budgets has not been implemented. Historical three/four-stage and larger-tile
candidates tested other fixed designs and did not win their measured controls.
Changing only the launch byte count without regenerating the layout would be
incorrect. A100's physical banks are not arbitrarily reconfigured by this idea.

## Method and validity checks

A proposed policy must jointly choose CTA/warp tile, stage count, operand/epilogue storage, occupancy and allocation size. First prove that all shared objects are disjoint and in bounds through fill/steady/drain phases. Then check full outputs/canaries, registers/local memory, actual launch resources and paired timing. Include any tuning/startup cost in the application comparison.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

There is no adaptive-budget CLI switch. Establish the explicit 128×128 control
and compare the checked tile64 family:

```sh
.venv/bin/python -m openassembler gemm --m 1024 --n 4096 --k 1024 --dispatch manual --out-dir output/smem-budget-control
.venv/bin/python -m openassembler gemm --m 1024 --n 4096 --k 1024 --tile 64x64 --out-dir output/smem-budget-tile64
.venv/bin/python -m pytest -q tests/test_gemm_codegen.py tests/test_cli_gemm.py
python3 -m json.tool docs/evidence/a100-mma/phase2/summary.json
```
The first command's manifest reports the selected shared-memory requirement. The historical stage tests are evidence about those fixed candidates, not a tested adaptive policy.

## Code and retained evidence

- [Fixed kernel allocation/launch contract](../../../openassembler/kernels/gemm.py)
- [Driver shared-memory opt-in](../../../openassembler/gpu/cuda_driver.py)
- [Historical stage/resource results](../../evidence/a100-mma/phase2/summary.json)
- [Deeper pipeline outcomes](deeper-pipelines-and-tile-shapes.md)

## What remains open

Extend the existing bounded tile/dispatch search to independently generated
stage and epilogue storage plans. More buffering must earn its register,
occupancy and control cost; the deepest pipeline is not assumed fastest.
