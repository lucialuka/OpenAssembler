# Tensor instruction overhead studies

This directory tests changes that reduce scalar instructions around an SM80
tensor mainloop. Each candidate keeps the established HMMA control value,
post-chain drain, LDSM visibility/address dependencies, async-copy groups,
barriers, K order and full-output contract. One exact tile64/direct policy has
since passed a current production-source A100 pilot and is enabled through the
exact GEMM dispatch table; the copy-address source rewrite remains experimental.

## Copy-address reuse

For the 64×64 CTA, each 16-byte vector copy maps thread `t` and chunk `c` to
vector `t + 128*c`. The chunk changes the logical row by `16*c`, leaves the
column and low-three-bit shared swizzle bank unchanged, and therefore lets the
emitter derive later global pointers from the first pointer by adding
`c*32*K` bytes. The additions are 64-bit so pointer carry is preserved.
`address_certificate()` exhaustively checks threads, chunks, several K sizes,
CTA offsets, pointer bases near 32-bit carry boundaries, and shared addresses.

For 128×512×512, the source-preserving experiment compiles through the native
SM80 tensor selector and encoder with 88 registers. Reusing copy addresses plus
the existing direct-home lowering reduces the native stream from 723 to 597
instructions, while retaining 128 HMMA instructions, 64 LDSM instructions,
HMMA8 and sixteen post-HMMA drain NOPs. That is a static instruction reduction,
not a predicted runtime percentage. The per-policy SASS and compiler resources
must be checked for each shape before using this approach.

## A100 evidence

`a100-pointer-init_20261109T121351.605036Z.json` is a one-shape pilot with
retained PTX, cubin and SASS for each method. All full output, FP32 reference,
same-source ptxas bitwise, baseline ptxas bitwise, canary, immutable-input and
poisoned-graph checks passed. On its A100-SXM4-40GB worker, 128×512×512 FP16
measured approximately 9.73 µs for legacy tile64 and 8.90 µs for address reuse,
direct pointer homes and early pointer advance (about 8.5% lower). Matched
tile64 ptxas was about 7.04 µs and cuBLAS about 6.40 µs. Thus the candidate
improved OpenAssembler's own tile64 path but did not beat either NVIDIA control.

`a100-pointer-geometry-summary_20261109.json` summarizes twelve same-session
timing rows comparing tile64 and tile128 over six shapes and two input dtypes.
Each row in the remote log was emitted only after its full correctness gates.
Across three rounds, tile64 beat tile128 by a wide margin for the small
128×512×512 and 128×1536×512 shapes, by about 21–31% for 1024×1024×1024, and by
about 40% for 1024×512×512. Tile64 lost by about 17% on 1024×4096×1024 and by
about 48–55% on 4096×2048×8192. Matched tile64 ptxas and the cuBLAS call remained
faster than the corresponding tile64 OpenAssembler result in these cohorts.

The geometry run was submitted detached. Its per-case timing lines survived in
the local log, but the caller was stopped before its return payload could write
the full gate records, compiler/source identity, or retained cubin/SASS files.
Treat these twelve rows as a useful routing hypothesis, not promotion evidence.
The synchronous production-source pilot
`a100-pointer-init_20261109T232622.533492Z.json` separately validates the
128×512×512 FP16 tile64/direct policy with retained binaries and complete gates.
Its result is reflected by one exact row in
`openassembler/kernels/sm80_gemm_dispatch.json`; it does not validate the
other shape rows. The copy-address source rewrite remains experimental.

For the production pilot, direct pointer lowering removes 64 MOV instructions
from the 64×64 native SASS listing (728→664), with HMMA 128, LDSM 64, sixteen
drain NOPs and 88 registers unchanged. Across three rounds, tile64/direct is
9.15–9.18 µs versus 9.60–9.66 µs for tile64/legacy and 18.94–19.01 µs for the
best tile128 OA control. Same-source tile64 ptxas is 7.04 µs and cuBLAS is
6.30–6.34 µs. This is a bounded OA-vs-OA win, not a CUDA-toolchain win.

A subsequent twelve-shape production replay attempt ended when Modal stopped
the app before returning results. It emitted no usable raw result, artifact
bundle or gate record and is excluded from promotion.

The public CLI tuning path was then exercised synchronously on the same exact
shape. `a100-production-autotune_20261110.json` records the seven admissible
candidates, three numerical families, bitwise/canary/input checks, poisoned
graphs, resources, hashes and 3×8 paired timings. It selected tile64/direct at
9.22–9.28 µs versus 19.90–19.97 µs for the 128×128 reference. The matching
`.artifacts.zip` contains 35 PTX/cubin/SASS files; the record was reused and
the selected kernel rebuilt through the CLI. ptxas (about 7.04 µs) and cuBLAS
(about 6.30 µs) remained faster. This validates the tuner and the exact
production policy, not other shape rows.

The run protocol and failed attempts are recorded in
`tile-geometry-replay.json`; the pre-run matrix is in
`tile-geometry-protocol.json`. `a100_pointer_init_modal.py` compiles each
candidate through the native selector and container splice, compares against
same-source ptxas and cuBLAS, and writes checked PTX/cubin/SASS artifacts when
the full response is returned.

## Reproduction

```sh
.venv/bin/python -m pytest -q exp/instruction_overhead/test_candidates.py
python3 -m modal run exp/instruction_overhead/a100_pointer_init_modal.py \
  --pilot --rounds 3 --samples 12
python3 -m modal run exp/instruction_overhead/a100_pointer_init_modal.py \
  --rounds 3 --samples 12
```

Omit `--detach` so the local runner can retain the returned evidence. Timing is
for isolated GEMM graph replay; it excludes framework composition, packing,
backward/optimizer state and whole-training steps. A tile policy should be
selected against the best applicable OA shape policy, not compared only with
one default tile.
