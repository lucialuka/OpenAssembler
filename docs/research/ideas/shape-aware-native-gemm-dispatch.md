# Shape-aware native GEMM dispatch and on-device tuning

Status: implemented

This status covers the checked native policy selector, explicit GPU tuning and
their bounded GEMM contracts. It does not mean OpenAssembler beats ptxas or
cuBLAS, or that every neural-network operation has an autotuner.

## Idea in plain language

A kernel that wins on a short output projection can lose on a large QKV
product. The compiler should preserve several correct implementations and
select between them using the actual shape and precision, rather than deleting
a useful alternative because it lost one earlier experiment.

The normal `gemm` command now uses an exact measured policy table. A current
matching entry selects its recorded native implementation; a missing, stale or
ambiguous entry keeps the reference policy. An explicit implementation option
selects manual mode for reproducible experiments. Optional A100 tuning compiles
and checks the admitted alternatives, then compares them on the selected GPU.
Ordinary compilation performs no hidden GPU job or network search.

## Contract and candidate space

The operation remains contiguous, unbatched `C=A@B.T`, with FP16/BF16 inputs,
FP32 accumulation/output, alpha one, beta zero, no tails and no workspace. The
selection key includes M/N/K, dtype, architecture, layout, accumulation/output
precision, epilogue, alignment, batch count, workspace and tail policy. The API
requires 16-byte input and eight-byte output alignment. A semantic role such as
“QKV” does not override a different storage or precision contract.

Representable native candidates include the reference stage-pair mainloop,
the combined-loop control, Morton8/grouped-N traversal, direct pointers,
earlier pointer advancement, their admitted direct/early combination, bounded
load-next fragments and a 64×64 CTA family. Tile64 now has independently
selectable copy-address reuse, direct pointer homes and early advancement. The
existing dispatch table fingerprint predates this expanded candidate set and
is stale, so the selector currently falls back to the reference for every
shape. A 42-contract production-source A100 campaign is testing the expanded
policy set. Only independently held-out promotions will repopulate automatic
dispatch. `--autotune` can test admissible choices on the current A100.
`GemmPlan` rejects incompatible combinations before a build. Dispatch does not
silently replace kernels with ptxas, cuBLAS or an Inductor graph.

The 128×128 CTA uses four 64×64 warp tiles and 64 KiB shared memory. The
64×64 CTA uses four 32×32 warp tiles and 32 KiB. Both use 128 threads, two
stages, explicit async visibility and the protected HMMA8/drain15 controls.
M/N must fit the chosen full-tile family; K is a positive multiple of 128.
General strides, transposes, TF32, split-K and unsupported stage/fragment
combinations are not admitted merely by adding them to a search list.

## Dispatch integrity and evidence

The built-in selector requires a unique exact contract, a promoted entry with
passed gates, current compiler/data identity and the selected generated PTX
body hash. It falls back to the reference on any mismatch. The checked PTX
annotation and ordinary manifest describe the actual selected launch; exact
recompilation retains it. `gemm-dispatch.json` records the decision, scope and
selected plan beside the artifacts.

The preceding [32-case role study](../../../exp/shape_coverage/a100-role-gemm_20261109T113850.802215Z.json)
establishes why policy selection is needed: tile64 improves 18 cases and loses
14, although all 1,296 input/method and 432 graph checks pass. It supplied
historical candidates, not current automatic rows. The earlier integrated
pilot on 128×512×512 FP16 found tile64/direct at 9.15 µs; same-source ptxas and
cuBLAS were faster at about 7.04 and 6.30 µs.

The first 12-shape production replay was interrupted and had no return bundle;
it is excluded. Its replacement uses the expanded source identity and includes
copy-address reuse. Each case retains raw JSON and PTX/cubin/SASS before the
next case starts. Automatic entries remain disabled until current selection
and independent held-out promotion gates pass.

The table is a finite set of measurements, not a universal model of A100
performance. An unknown shape can have a faster unmeasured alternative. Even
a known shape can change with clocks, toolkit, cache residency and device
variant. Per-device tuning is the explicit way to test the current alternatives
instead of pretending that one old median is a permanent optimum.

## Optional on-device tuning

`--autotune local|modal` enumerates the current admissible native policies for
the shape. It compiles them through the real CLI, loads the OA and same-source
ptxas cubins, checks disassembly/resources and validates every complete output
on zero, integer and normal inputs. It retains guards, input immutability,
finite outputs, same-source and baseline bitwise comparisons, poisoned CUDA
graphs and HMMA/drain controls. A failed gate aborts admission.

Timing uses shuffled paired candidate order, at least three rounds and at
least eight samples per round. Sixteen-operation graphs amortize host launch
overhead. A candidate must beat the reference by at least 2% in every round;
uncertain gains retain the reference. Among qualifying candidates, selection
uses the geometric mean of round medians. This is a bounded measurement of
resident-input kernels, not packing, model-state integration or a training step.

An explicitly saved tuning record has a companion `.artifacts.zip` containing
the generated PTX, native/ptxas cubins and disassembly. Reuse checks the current
compiler, exact contract, current candidate sources, recorded gates and raw
paired samples. A GPU tuning result remains scoped to its recorded A100 worker
and timing protocol; it is not evidence that an arbitrary imported claim is
true or that a different GPU will repeat the ranking.

The public Modal path was exercised on the same 128×512×512 FP16 shape. Seven
admissible candidates passed all output, bitwise, canary, input, graph,
disassembly and resource gates; three rounds of eight paired samples selected
tile64/direct. It measured 9.22–9.28 µs versus 19.90–19.97 µs for the 128×128
reference. This second timing record retains 35 PTX/cubin/SASS files and was
successfully reused to compile the selected policy. The durable
[tuning record](../../../exp/instruction_overhead/a100-production-autotune_20261110.json)
and [binary/source archive](../../../exp/instruction_overhead/a100-production-autotune_20261110.artifacts.zip)
are the reproduction evidence. Same-shape ptxas and cuBLAS remain faster, as
reported in the separate production pilot above.

## Reproduction and command set

Native compilation requires Python 3.12+ and the CUDA tools. Local tuning also
requires the optional supported PyTorch/CUDA runtime and an A100. Modal tuning
uses a separately authenticated Modal client. Kernel tuning incurs compilation,
validation and timing costs; those costs are not included in a steady kernel
speedup.

```sh
openassembler gemm --m 128 --n 512 --k 512 --out-dir output/auto
openassembler gemm --m 128 --n 512 --k 512 --dispatch manual --out-dir output/control
openassembler gemm --m 128 --n 512 --k 512 --tile 64x64 --out-dir output/manual64
openassembler gemm --m 128 --n 512 --k 512 --autotune modal \
  --tuning-record output/tuning/small-output-f16.json --out-dir output/tuned
openassembler gemm --m 128 --n 512 --k 512 \
  --tuning-record output/tuning/small-output-f16.json --out-dir output/reused
```

Read [COMMANDS](../../COMMANDS.md#gemm-checked-native-a100-tensor-gemm) for every
flag and the [small-tile dossier](small-grid-tensor-cta-tiles.md) for geometry
findings and negative controls.

## Code and remaining questions

- [Policy table and selector](../../../openassembler/gemm_dispatch.py)
- [Bounded tuning and record validation](../../../openassembler/gemm_tune.py)
- [Checked plan generation](../../../openassembler/kernels/gemm.py)
- [PTX annotation and build publication](../../../openassembler/gemm.py)

Expand the table only with complete source/binary identities, paired held-out
cases and the applicable numerical/resource checks. Revalidate a rejected
policy on a genuinely different contract rather than treating its old failure
as universal. General dependency-aware tensor reordering, complete native
training-state integration and library-level coverage remain separate work.
