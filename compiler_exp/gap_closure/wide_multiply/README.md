# Low 64-bit multiply research

Status: retained backend-only baseline. The expanded [promotion cycle](promotion/README.md)
validates and promotes both backend and typed Python source routes. Results below
are the original narrower candidate gate, retained unchanged.

The original [CUDA source](cuda.cu) is a simple unsigned 64-bit elementwise
product. Actual nvcc emits [mul.lo.s64](cuda.ptx), which the recorded baseline
rejected in [baseline-cuda.json](baseline-cuda.json). This is a concrete remaining
G11 boundary after wide selection. The signed spelling of the PTX low product
has the same modulo-2^64 bits; the source-level workload uses unsigned C++.

The [candidate](candidate.py) computes a low32 product, unsigned high32 carry
and two low32 cross terms using four existing recovered IMAD/IMAD.HI words.
Fresh low/high results preserve aliases, snapshots and guarded destinations.
No guessed field or encoding table changes. `baseline-ra.py` and
`baseline-contracts.py` retain the exact preceding production sources.

The [offline candidate](candidate-offline.json) passes 72 rows / 1,296 complete
PTX/normal/poisoned SASS executions: u64/s64, aliases in both operands, false and
negated guards, parameters, left/right/both immediates, two seeds and three
lengths. References use independent arbitrary-precision Python multiplication
modulo 2^64, including full input/output guards and unchanged tails/inputs.
The [baseline](baseline.json) preserves eighteen source rejections; its replay
explicitly loads the frozen baseline, so rerunning after promotion cannot turn
a forced assertion into a fabricated backend rejection. The pre-replay records
are also retained. The [toolkit probe](oracle-probe.json) accepts all eighteen original typed PTX
sources. The [A100 candidate gate](candidate-a100-20261123T211046Z.json) passes 760 launches:
eighteen original PTX cases plus the actual unsigned CUDA source, original
ptxas control and four OA levels, lengths 0/1/259/384, two seeds, three 128-thread
CTAs. Its archive retains exact original/emitted sources, cubins, disassembly,
manifests and complete input/output bytes. The [candidate audit](candidate-audit.json)
checks 7,782,400 independent bytes, 76 native word/register/resource matches,
source custody, ownership and a corrupted-output rejection control. The subsequent [promotion cycle](promotion/README.md) adds typed Python
source/parameter squares, full regressions, installed packaging, suites and
prior code/policy continuity. Hardware/integration gates are separate work; functional agreement does not establish native IMAD.HI timing or production
support. No performance claim is made.

```sh
python compiler_exp/gap_closure/wide_multiply/probe.py
python compiler_exp/gap_closure/wide_multiply/probe.py --candidate
python compiler_exp/gap_closure/wide_multiply/oracle_probe.py
```
