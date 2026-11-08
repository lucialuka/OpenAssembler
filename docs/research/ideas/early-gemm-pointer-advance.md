# Moving next-tile pointer updates ahead of GEMM compute

Status: implemented

The result is conditional: moving these address updates helps the tested wide
forward GEMM but hurts the tall control. It remains an experiment, including
the combination with direct pointers. There is no production compiler change
or general speedup claim.

## Idea in plain language

Each GEMM block copies the next operand tile into shared memory while working
on the current tile. The old source updates sixteen input addresses after the
current tensor computation and its final barrier. This experiment performs
those updates immediately after issuing and committing the next tile's copies,
before the current tensor computation.

The copies still read their original addresses before any update. The current
computation uses shared-memory fragment addresses, not these next-tile global
pointers. Moving independent address work may make better use of the pipeline;
it may also make a warp wait earlier for asynchronous address consumption and
lose overlap. Timing must decide which outcome occurs on each shape.

The experiment moves instructions; it does **not** remove copies, barriers,
copy waits, matrix operations or proven dependency handling. Native generation
retains HMMA issue gap eight, epoch drain fifteen, operand reuse and the
asynchronous-copy/LDSM address read/write protections. The compiler's ordinary
source-order tensor scheduler still handles the changed source.

## Methods and validity checks

Two shapes were chosen before measurement: 1024×4096×1024, where the existing
direct-pointer option helped, and 16384×4096×4096, where it hurt. FP16 ran first;
the wide positive result justified a separate BF16 follow-up on the same pair,
retaining the unfavorable control. Both GPU calls used an A100-SXM4-40GB,
108 SMs, driver 13030, PyTorch 2.7.1 and CUDA 12.8.

For each precision and shape, the methods are:

- Original native OA with legacy pointers, and with direct pointers.
- Relocated native OA with legacy pointers, and with direct pointers.
- ptxas compiling the original PTX, and ptxas compiling the relocated PTX.
- FP32-output cuBLAS `GemmEx` algorithm 99, using FP32 compute and disallowing
  reduced-precision reduction.

Inputs are contiguous FP16/BF16 `A[M,K]` and `B[N,K]`; output is FP32
`C=A@B.T`, accumulated in FP32. No narrowed output or different arithmetic
policy is used to admit a result. These timings exclude packing/transposes,
allocation, compilation and full-network work.

CPU checks require **the complete PTX instruction/operand multiset to remain
identical** after relocation. The early and original native variants also have
the same 216 registers and 1600 words with legacy pointers, or 1472 words with
direct pointers. Static MMA, shared-fragment-load, copy and barrier counts are
unchanged. Native words/control fields and critical operands round-trip through
`nvdisasm`. Passing these structural checks alone does not prove hardware
correctness.

Hardware checks use zero, small-integer and normal inputs. Every output is
compared with a complete FP64 reference outside timing; native output must also
match ptxas compiling its **own exact source bitwise**. Every method checks
NaN-poisoned output replacement, output canaries and unchanged input buffers.
Poisoning after graph capture and validating after replay confirms captured
work actually writes the complete result.

Across the FP16 and BF16 calls, **84/84 full method/input checks pass**,
covering **2,994,733,056 output elements**. Another **28/28 poisoned graph
replays pass**, covering 998,244,352 output elements. These repeated checks
are not independent statistical replications. The worst relative L2 error
against FP64 is 6.61e-6. Each precision has one worker call with three timing
rounds, not multiple independent-process replications of that precision.

## Measured results and decision

Each round contains eleven shuffled paired samples, with five launches per
CUDA graph. The first round can select a fixed option; the table reports the
two later evaluation rounds. Ratios are **original time / relocated time**:
above one favors relocation. Timings are paired within each actual worker.

| Shape / input precision | Legacy-pointer evaluation ratios | Direct-pointer evaluation ratios |
|---|---:|---:|
| 1024×4096×1024 FP16 | 1.070× / 1.075× | 1.018× / 1.015× |
| 1024×4096×1024 BF16 | 1.073× / 1.076× | 1.023× / 1.021× |
| 16384×4096×4096 FP16 | 0.994× / 0.992× | 0.952× / 0.959× |
| 16384×4096×4096 BF16 | 0.994× / 0.995× | 0.921× / 0.919× |

On wide BF16, early direct mode takes about **78.85 µs**, versus
**80.49–80.69 µs** for the already optimized direct control and
**63.69–63.90 µs** for relocated-source ptxas. The default library control
takes 83.15–83.35 µs in that worker. This supports a small additional native
improvement on the chosen shape; it does not beat the stronger ptxas control.
Wide FP16's incremental gain over direct mode is only about 1.5–1.8%.

The approximately 7% legacy-relative gain is larger, but the already available
direct-pointer control must remain visible. Comparing only against legacy
would overstate how much new progress remains after existing tuning.

The tall result rejects unconditional early advancement, especially in direct
mode. The per-option winner selected on round zero for the tall case does not
remain better in the evaluation rounds: the original legacy control is faster.
The experiment therefore also demonstrates why a fastest-looking tuning sample
must have a held-out check.

Both original and relocated `m_fast` ptxas tall controls take approximately
3.49–3.52 ms in these workers. That slow traversal is not the strongest CUDA
baseline: the main campaign's Morton ptxas and cuBLAS controls are faster than
the selected OA tall kernel. The original `m_fast` and Morton cubins are
byte-for-byte identical to the retained November 5 controls; new compiler code
is not an explanation for a changed absolute timing. Allocation, cache, clock
and worker conditions require separate counter-controlled investigation.

No scheduling cause is claimed from these timings alone. In particular, the
test does not prove an operand-collector or shared-bank bottleneck, and unchanged
instruction count does not imply unchanged dynamic overlap.

## Reproduction and commands

Use the compiler test environment, CUDA 12.8 tools and an authenticated Modal
client from the repository root. Compilation and every artifact stay under
`exp/nn_round2`. The dedicated GPU launcher below is the tested route.

```sh
.venv/bin/python -m pytest -q exp/nn_round2/test_tensor_suite.py \
  exp/nn_round2/test_tensor_followup.py
.venv/bin/python exp/nn_round2/tensor_suite.py build --profile full
.venv/bin/python exp/nn_round2/tensor_followup.py \
  --base-manifest exp/nn_round2/tensor_build_20261107T142912.570956Z.json \
  --dtype f16
python3 exp/nn_round2/tensor_followup_run.py \
  --payload exp/nn_round2/tensor_followup_payload_20261107T143207.740187Z.json \
  --rounds 3 --samples 11
.venv/bin/python exp/nn_round2/tensor_followup.py \
  --base-manifest exp/nn_round2/tensor_build_20261107T142912.570956Z.json \
  --dtype bf16
python3 exp/nn_round2/tensor_followup_run.py \
  --payload exp/nn_round2/tensor_followup_payload_20261107T144004.242308Z.json \
  --rounds 3 --samples 11
.venv/bin/python exp/nn_round2/tensor_analyze.py \
  exp/nn_round2/runs/tensor-followup-20261107T143807.885774Z.json
.venv/bin/python exp/nn_round2/tensor_analyze.py \
  exp/nn_round2/runs/tensor-followup-20261107T144047.961466Z.json
```

The timestamped paths replay/analyze the retained snapshot. Fresh builders print
new manifest/payload names; substitute those names for a new experiment. Loaders
refuse compiler-source or binary drift. The shown local compiler environment
and the Modal-client Python can differ; the remote image uses its recorded
Python/PyTorch/CUDA runtime.

Two launcher attempts failed before GPU execution: a local nested Modal
function was rejected, followed by a Python 3.10/3.11 serialized-function
mismatch. The corrected top-level worker produced the completed records.
These failures remain in the raw logs, and neither counts as a hardware test.

## Code and evidence

- [Relocation and payload builder](../../../exp/nn_round2/tensor_followup.py)
- [Structural and policy CPU gates](../../../exp/nn_round2/test_tensor_followup.py)
- [Tested top-level Modal launcher](../../../exp/nn_round2/tensor_followup_run.py)
- [Complete frozen GPU validator](../../../exp/nn_round2/tensor_suite.py)
- [FP16 payload](../../../exp/nn_round2/tensor_followup_payload_20261107T143207.740187Z.json)
  and [BF16 payload](../../../exp/nn_round2/tensor_followup_payload_20261107T144004.242308Z.json)
- [FP16 raw result](../../../exp/nn_round2/runs/tensor-followup-20261107T143807.885774Z.json)
  and [BF16 raw result](../../../exp/nn_round2/runs/tensor-followup-20261107T144047.961466Z.json)
- [FP16 held-out summary](../../../exp/nn_round2/tensor_followup_summary_20261107T143807.json)
  and [BF16 held-out summary](../../../exp/nn_round2/tensor_followup_summary_20261107T144047.json)
- [Summary generator](../../../exp/nn_round2/tensor_analyze.py)
- [Existing direct-pointer option](direct-pointer-homes.md) and
  [joint option study](joint-tensor-options.md)

## What remains open

Use counters to explain the favorable and unfavorable cases before generalizing
the placement rule. Repeat the small incremental direct-mode gain independently
and test other forward/dX/dW shapes, complete training state and whole-step
cost. Promotion would need an explicit shape/contract policy, regression tests
and a validated ordinary compiler path. No such production promotion is made
in this research round.
