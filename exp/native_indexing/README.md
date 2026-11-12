# Runtime integer indexing on A100

The November 12 repair adds `F2I` to the production scheduler's asynchronous
source-reader protection. Integer division/remainder arithmetic is unchanged.
The former variable-latency write scoreboard protected the conversion result,
but did not stop a following instruction from overwriting its source before
operand collection. The original minimal `i % cols` reproducer fails even in
`safe` mode without that protection.

The first source-identical ablation run retains four modes per optimization
level: original controls, F2I read protection only, I2F read protection only,
and both protections. It tests the two simple index kernels at seven shapes,
three repeats, all four optimization levels and exact-source ptxas controls.

| Controls | safe failures / 42 | mined | sched | opt |
|---|---:|---:|---:|---:|
| Original | 28 | 26 | 30 | 30 |
| I2F read protection only | 27 | 27 | 29 | 30 |
| F2I read protection only | 0 | 0 | 0 | 0 |
| Both protections | 0 | 0 | 0 | 0 |
| Same-source ptxas | 0 | 0 | 0 | 0 |

The safe stream at byte offset 0x120 is `F2I ... R7,R5`, followed by
`MOV R5,0`. The failing conversion sets W5 with no read barrier; the isolated
repair also sets R0, and the MOV waits on R0. Instruction order and arithmetic
operands remain identical. Changing I2F alone does not repair it. The complete
native cubins, disassembly and control maps are retained in `builds/baseline/`;
the source at that snapshot contains the two simple index kernels only.

Two independent fixed-source A100-SXM4-40GB calls each pass 432 checks covering
156,477,312 result values, including ptxas controls. The expanded source adds
unsigned and signed quotient/remainder pairs on arbitrary input arrays with
guarded tails. Inputs cover zero numerators, quotient boundaries, signed
extremes, large unsigned values, random words and power-of-two/non-power-of-two
divisors, including negative signed divisors. Divide by zero and signed
`INT_MIN/-1` quotient overflow are excluded from the stated contract. Every
result is checked against exact host integer arithmetic; all output guards and
input-immutability checks pass. These are correctness measurements, not timing
claims. Raw failures are preserved in `runs/baseline-20261112T014909Z.json`;
the first two passing records are `runs/fixed-20261112T015403Z.json` and
`runs/fixed-20261112T015533Z.json`.

Two further complete A100 calls with alternating output poison also pass all
432 checks each: `runs/fixed-20261112T020024Z.json` and
`runs/fixed-20261112T020151Z.json`. The latter records the exact harness hash.
The former reports A100-SXM4-80GB; the latter reports A100-SXM4-40GB. Device
metadata takes precedence over the requested Modal tier and times are not pooled.
The former returned its complete result successfully but the local Modal client
timed out waiting for final app logs; that log-transport warning is preserved
in the task transcript and did not truncate the returned checks. Neither run
changes the hardware/numerical contract. `manifest.json` hashes retained files.

Build/run from the repository root with CUDA 12.8, Python 3.12+ and an
authenticated Modal client. A fresh tag is required; existing build snapshots
are never overwritten.

```sh
PYTHONPATH=. .venv/bin/python exp/native_indexing/build.py --tag my-fixed-replay
modal run exp/native_indexing/run_modal.py \
  --payload exp/native_indexing/builds/my-fixed-replay/payload.json
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_runtime_indexing.py
```

To reproduce the causal ablations after the production fix, use a different
fresh tag and add `--ablations`. The experiment temporarily removes F2I from
the scheduler's reader list only while building its explicitly labelled old
control; it never writes production code. The current driver alternates three
output poison bytes across repeated launches to distinguish complete writes
even when an expected result is all-one bits.

`audit_dispatch.py` independently re-encodes retained selection and held-out
native artifacts after the scalar fix. It checks all GEMM candidate plans,
and all four warp geometries for the five installed rowwise contracts, before
refreshing table compiler identities. Historical timing results and raw records
remain untouched; byte-identical machine text/registers justify continuity,
not a new performance claim.

The completed `dispatch-continuity.json` records 1,000 GEMM candidate/session
checks (all 500 candidates in both sessions), and 40 rowwise geometry/session
checks (four geometries for all five installed contracts in both sessions).
All native words, controls, register counts and generated sources agree.
The 42 GEMM and five rowwise actual automatic selectors were also checked.

The final timestamped continuity audit also records the unchanged hashes of
every raw result/archive and checks all six fixed Tensor Core word/register
goldens. It fingerprints the standalone ownership validator because build
acceptance belongs to the compiler identity, although that validator does not
change machine instructions. `ownership-smoke/` and `ownership-indexing/`
retain compile-only ledger checks; neither directory is a GPU measurement.
