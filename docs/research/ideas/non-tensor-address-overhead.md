# No-op address instruction removal

Status: tested-not-selected

Status applies to the stated experiment and contract, not to every possible
address-lowering strategy.

## Idea in plain language

The GEMM PTX generator emitted eight `add.u32 %r30,%r30,0` instructions in the
tested full-tile mainloop. They do not change the address value, so removing
them is a safe instruction-selection candidate. The experiment asks whether
fewer scalar instructions also reduce A100 time or scoreboard pressure.

## Method

`exp/tensor_scheduler/non_tensor_overhead.py` built three FP16 GEMMs on an
A100-SXM4-40GB:

* 512×2048×512;
* 2048×512×512; and
* 1024×4096×1024.

For each shape it compared the ordinary OA PTX, a transformed OA PTX with the
eight no-op adds removed, and the matching `ptxas` controls. Each cubin passed
`nvdisasm`, full finite-output checks, and bitwise equality against its ptxas
control. The run record is
`exp/tensor_scheduler/non_tensor_overhead_20261108T044028.978463Z.json`.

## Findings

The OA candidate removed exactly eight SASS instructions per kernel and kept
the same declared register count of 216. The matching ptxas stream was
unchanged, which confirms that ptxas already eliminated these no-ops.

| Shape | OA baseline | OA candidate | ptxas control | Candidate timing result |
|---|---:|---:|---:|---|
| 512×2048×512 | 1,600 instructions | 1,592 | 1,704 | 0.020992 → 0.020992 ms |
| 2048×512×512 | 1,600 | 1,592 | 1,704 | 0.020787 → 0.020890 ms |
| 1024×4096×1024 | 1,600 | 1,592 | 3,168 | 0.088576 → 0.088986 ms |

The candidate is numerically correct and reduces static instruction count, but
it does not improve paired timing. The likely explanation is that these eight
instructions are outside the dominant tensor/memory critical path or are
absorbed by existing issue gaps. This rejects the narrow rewrite as a useful
standalone optimization, while supporting a broader address-CSE and scheduling
effort that also changes live ranges, pointer homes, reuse and overlap.

## Reproduction

```bash
modal run exp/tensor_scheduler/non_tensor_overhead.py
```

The experiment remains under `exp/`; no production compiler code was changed.

