# Hardware-counter-guided tensor analysis

Status: validated-not-integrated

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Measure what the GPU actually waits for: tensor activity, shared bank traffic, short/long scoreboards, barriers, dynamic instructions and cache/DRAM behavior. Use these to choose the next experiment rather than guessing from SASS length.

## Findings and implementation decision

Retained matched profiles show equal tensor work but different overhead and stalls. One early direct/same-PTX comparison reported tensor activity 59.43% versus 78.26%, short-scoreboard stalls 13.79% versus 1.15%, and 90.85M versus 69.62M executed instructions. Later schedule profiles improved tensor activity, and November 5 repeats examined Morton locality.

These are validated measurements and research harnesses, not an integrated automatic compiler cost model. They do not conclusively isolate register-bank or operand-collector causes. Newer direct-pointer and whole-stack counter analyses have their own dossiers.

## Method and validity checks

Validate identical outputs and tensor work first, query available NCU metrics for the installed version, retain raw rows and resource/device identity, and collect each method on matching shapes. NCU replay changes caches and may run many passes; use separate CUDA-graph timings for performance claims.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
PYTHONPATH=. modal run exp/a100_counters/run.py --mode preflight
PYTHONPATH=. modal run exp/a100_counters/run.py --mode build-current
PYTHONPATH=. modal run exp/a100_counters/run.py --mode validate-cublas-ncu
```
Read the harness options before scheduling cloud work. NCU collection needs the profiler and permissions in its remote image. A failed collection is not a zero-counter result, and preflight itself can request an A100.

## Code and retained evidence

- [Retained counter runner](../../../exp/a100_counters/run.py)
- [Earlier counter/resource follow-up](../../evidence/a100-mma/phase3/profile-followup.json)
- [November 5 counter records](../../../exp/a100_counters/counter-validation_20261105T171913.378083Z.json)
- [Newer direct-pointer work](direct-pointer-homes.md)

## November 7 NN arithmetic and tensor counter round

The isolated [round-two counter worker](../../../exp/nn_round2/counter_worker.py)
profiles twelve matched controls on an A100-SXM4-40GB: OA/ptxas GELU and RMSNorm
backward, plus wide/tall GEMM pointer/order controls. A separate four-profile
run compares the confirmed [local-FMA backward candidates](fused-backward-reductions.md).
All sixteen subprocesses return successfully and pass complete output,
output-canary and input-nonmutation checks before and after the profiled launch.
Each process uses the exact binary from an already admitted experiment record,
checking its SHA-256. It makes three matching kernel launches; Nsight's exact
function-name filter skips the first two and profiles the third. Other
initialization/reference kernels are outside that filter.

The profiler uses kernel replay, `--cache-control none` and
`--clock-control none`. Its warnings about uncontrolled caches/clocks are
retained. Observed SM clocks differ materially between profiles, so the raw
profile duration is **not** another performance speedup measurement. Timing
admission comes from the separate paired graph experiments.

### Recovering the original parser failure

The [first raw counter record](../../../exp/nn_round2/runs/counters-20261107T143821.759059Z.json)
originally says `completed_with_failures`: its parser expected a long table
with `Metric Name`/`Metric Value` rows, but Nsight 2025.1 emitted a wide table
with a header, units row and one profiled-kernel row. All twelve child proofs
and profiler exits had already succeeded. The
[derived recovery record](../../../exp/nn_round2/counter_recovered_20261107T143821.759059Z.json)
parses that retained stdout, verifies kernel/binary/proof identity, and recovers
22 of 23 requested fields for all twelve profiles. No GPU rerun or original
record modification was needed.

Nsight can also return exit status zero while warning that a requested metric
is invalid. The requested `smsp__inst_executed_pipe_sfu.sum` is absent and must
not be interpreted as zero. The corrected worker rejects those query warnings.
Launch register/shared-memory attributes can still appear automatically in raw
output even though the hardware-metric query does not accept their names.
The additional wide-table columns are device, launch, profiler and metric
rollup metadata, not evidence of additional benchmark kernels.

The [four-profile FMA record](../../../exp/nn_round2/runs/counters-20261107T144534.159647Z.json)
uses the corrected parser and succeeds directly. Its supported XU instruction
counter is a combined named execution-pipe observation, not an isolated SFU
utilization metric. The requested XU-active-cycle spelling is unavailable.
The [derived counter summary](../../../exp/nn_round2/counter_summary_20261107.json)
keeps every profile's requested fields and source identities.

### What the tensor profiles support

For the wide `1024×4096×1024` FP16 product, OA legacy/direct execute approximately
4.896/4.371 million dynamic instructions, versus 3.231 million for same-source
ptxas `m_fast`. Tensor-pipe instruction counts match at 2,097,152. OA
short-scoreboard percentages are 8.64/8.76%, versus 0.82% for ptxas. This is
evidence that pointer-copy removal leaves substantial instruction/dependency
overhead; it does not isolate one operand-collector cause. Banks are not
uniformly conflict-free in these newer profiles; raw shared-bank counts and
wavefront denominators are retained rather than replacing earlier evidence.

For the tall `16384×4096×4096` product, traversal changes memory behavior:

| Method / traversal | DRAM bytes, decimal GB | L2 sector hit | Long-scoreboard stalls | Tensor-pipe active |
|---|---:|---:|---:|---:|
| OA legacy `m_fast` | 3.050 | 64.17% | 13.91% | 75.14% |
| OA legacy Morton8 | 1.091 | 85.80% | 10.12% | 80.06% |
| ptxas `m_fast` | 4.535 | 52.21% | 57.02% | 50.45% |
| ptxas Morton8 | 1.222 | 83.43% | 3.65% | 85.50% |

All four execute 134,217,728 tensor-pipe instructions. These observations support
a cache/locality and memory-wait explanation for the measured tall order gap.
They do not establish that OA's general instruction scheduler is faster than
ptxas: a different source traversal can substantially improve the ptxas control
too. Observed ptxas clocks are about 1.407 GHz for `m_fast` and 1.087 GHz for
Morton8, which is another reason not to turn these replay durations into paired
benchmark ratios. See [Morton CTA ordering](morton-cta-order.md) and
[joint tensor options](joint-tensor-options.md) for independently timed controls.

### What the local-FMA profiles support

These are FP32, non-affine input-gradient operators, width 513 and 4096 rows.
RMSNorm uses four warps per CTA; softmax uses eight. The candidate contracts
local multiply-plus-sum terms into explicit FMA without removing shuffle
synchronization or changing the predeclared mathematical tolerances.

| Operator / version | Registers per thread | Dynamic instructions | Active-warps percentage | Eligible warps/cycle | Long-scoreboard stalls |
|---|---:|---:|---:|---:|---:|
| RMSNorm dX baseline | 61 | 974,848 | 39.69% | 0.21 | 83.67% |
| RMSNorm dX local FMA | 44 | 839,680 | 53.50% | 0.20 | 84.96% |
| Softmax dX baseline | 56 | 651,264 | 42.14% | 0.12 | 88.68% |
| Softmax dX local FMA | 43 | 585,728 | 54.71% | 0.13 | 87.57% |

RMSNorm loses 13.9% of dynamic instructions and 17 registers; softmax loses
10.1% and 13 registers. Their measured active-warps percentages rise by
13.81 and 12.57 percentage points. Global traffic remains essentially unchanged:
about 27.56 MB for RMSNorm and 27.35–27.42 MB for softmax, with L2 hit rates near
53%. Shared-load bank conflicts are zero in both scalar candidates. XU counts
are unchanged at 4096 for RMSNorm and zero for softmax.

This supports reduced local arithmetic and live-register pressure as useful
changes, with more active warps. It does **not** show that occupancy alone fixes
issue throughput: RMSNorm eligible warps and issue-active percentage slightly
decrease, and long-scoreboard stalls remain dominant. Nor does it show a
shared-bank or special-function fix. The performance gains and their independent
confirmation are in the [FMA dossier](fused-backward-reductions.md); a schedule
or load-coalescing follow-up still needs its own correctness and timing gates.

### Reproduce the new profiles and recovery

The payload planners verify retained admission records and binary hashes before
submission. They intentionally reject a changed binary; a fresh build requires
fresh full-output evidence rather than silently borrowing an old gate.

```sh
# Prepare and profile the twelve admitted baseline controls.
.venv/bin/python exp/nn_round2/counter_plan.py
PYTHONPATH=. modal run exp/nn_round2/campaign.py --phase counters \
  --payload exp/nn_round2/counter_payload.json

# Prepare the four independently confirmed local-FMA controls.
.venv/bin/python exp/nn_round2/counter_fma_plan.py
PYTHONPATH=. modal run exp/nn_round2/campaign.py --phase counters \
  --payload exp/nn_round2/counter_fma_payload.json

# Read-only recovery/analysis of the retained records; no GPU is requested.
PYTHONPATH=. .venv/bin/python exp/nn_round2/counter_recover.py
PYTHONPATH=. .venv/bin/python exp/nn_round2/counter_analysis.py
.venv/bin/python -m pytest -q exp/nn_round2/test_counter_worker.py
```

Each profile has a 90-second process-group deadline and each metric query a
five-second deadline; at most twelve profiles are accepted per payload. Raw
stdout/CSV, stderr, unsupported metrics, timeout/failure status, complete child
proofs, device identity and commands are retained. This research has not become
an automatic compiler cost model.

### Current-source counter revalidation: November 9

The counter harness was rebuilt from the current checkout and profiled four
full-output tall FP16/BF16 controls on an A100-SXM4-80GB. Nsight Compute 2025.1
accepted all requested metric names and collected both OA and same-PTX rows.
Each row passes zero, integer and normal full-output checks, canaries and
bitwise ptxas comparison. The raw record is
[`counter-validation_20261109T064116.119142Z.json`](../../../exp/a100_counters/counter-validation_20261109T064116.119142Z.json).

The current records again show equal tensor instruction counts (134,217,728)
but roughly 287.1–287.6M OA instructions versus 217.4–218.0M ptxas
instructions. OA m-fast has about 9.8–9.9% short-scoreboard and 11.2–12.1%
long-scoreboard stalls; ptxas m-fast has about 1.5% and 30.7–32.4%.
Morton lowers OA long scoreboard to about 10.1% and raises tensor-pipe active
percentage to about 80.1%, while ptxas Morton remains around 85.5% tensor active
and 3.9–4.0% long scoreboard. Shared-bank conflict counts are nonzero and
traversal-dependent for both compilers, so the earlier zero-conflict comparison
must not be generalized. The m-fast timing in this replay is not a promotion
claim because the same run reports order and clock sensitivity and ptxas
remains the stronger baseline under other paired campaigns.

## What remains open

Turn reproducible counter diagnoses into narrowly controlled candidate changes. Counter percentages are correlated observations, not by themselves a causal model or a general performance prediction.
