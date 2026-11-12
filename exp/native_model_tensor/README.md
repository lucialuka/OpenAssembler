# Native Tensor Core model recipes

`openassembler/modelfront/kernels_tensor.py` implements explicit
`f16_rne_f32_accum` recipes for full-tile linear forward, input gradient and
weight gradient. Graph tensors and master weights remain FP32. Each matrix
operand is rounded to binary16 round-to-nearest-even by OA-generated packing
code, optionally transposed, before OA's existing FP32-accumulating GEMM.
Forward bias uses a separate FP32 native add stage. Cast derivatives follow the
declared straight-through policy. This mode is an explicit numerical contract;
it is not a silent substitution for the FP32 scalar graph.

Recipes refresh every packed operand on each invocation, including updated
training weights. They retain two aligned FP16 workspaces, exact stage bindings,
launch geometry and source/binary/text hashes. Each module is independently
encoded by OA with the conservative tensor dependency policy and has exactly
one owned executable entry. NVIDIA ptxas supplies the donor/reference cubin;
reference functions are only executed in this experiment's comparator arm.
The production model runtime loads only ledger-owned OA modules.

The first A100-SXM4-40GB record,
[`tensor-linear-v1-20261112T021135Z.json`](runs/tensor-linear-v1-20261112T021135Z.json),
passes 81 complete checks: nine forward/dX/dW recipes across three shape
families, three input families (zero, small integer, normal), three repeats.
It checks complete results against a FP64 oracle of the declared half operands,
exact packing bits, input immutability and surrounding output/workspace guards.
Normal results require relative L2 below 2e-5; integer/zero cases are exact.

The stronger repeat,
[`tensor-linear-v2-20261112T021349Z.json`](runs/tensor-linear-v2-20261112T021349Z.json),
again passes all 81 checks and additionally requires bitwise output equality
with ptxas compiling the exact same source for each stage. Random raw FP32-bit
conversion stress includes signed zero, subnormals, rounding ties, overflow,
infinities and signed canonical quiet NaNs. CUDA graph destinations are
poisoned after capture and required to be fully replaced by finite outputs.
Twenty-seven separate paired native/ptxas timing rounds measure the complete
pack+GEMM+optional-bias pipeline, five invocations per graph, ten shuffled
samples per method/round. Compilation, allocation, FP64 checking and transfers
are outside timing. OA remains slower than the matched same-source ptxas
pipeline in this record; it supports no toolchain-superiority claim.

The requested Modal tier is A100-40GB; each raw record reports the actual device
and runtime. Never pool absolute times across device variants. No whole-model
convergence, arbitrary-shape support or fused-attention performance follows
from these isolated recipe measurements. The production CLI's complete-model
tests are separate evidence.

Reproduce from the repository root with CUDA 12.8, Python 3.12+ and an
authenticated Modal client. New build tags cannot overwrite old snapshots.

```sh
modal run exp/native_model_tensor/run_modal.py --tag my-linear-replay
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_model_tensor.py
```

`builds/TAG/payload.json` retains every source, launch, recipe and compiler hash,
OA cubin and matching ptxas donor/reference. The first driver failed during
Modal import because the mounted script path had fewer parents than a local
repository path. [`failures/startup-path.json`](failures/startup-path.json)
records that setup failure; the app was stopped, and it is not a GPU result.
The corrected driver reused the identical compiled payload.

`address_reuse.py` stages a bounded experimental optimization: the second
packed element reuses the first element's 64-bit pointer with a constant
displacement, retaining both exact conversions and asynchronous dependencies.
It is not imported by production code. `--address-cse` builds original and
candidate recipes together. Full outputs, exact packing and same-source ptxas
checks precede timing; the paired address ablation shuffles original OA,
candidate OA, original ptxas and candidate ptxas in the same three rounds.
Fifteen samples per arm/round replay five complete invocations each. Candidate
admission requires a separate evaluation of complete-region cost and retained
negative cases; a smaller source/address count alone is insufficient.

```sh
modal run exp/native_model_tensor/run_modal.py \
  --tag my-address-ablation --address-cse
```

The completed paired
[`tensor-pack-cse-v2-20261112T022228Z.json`](runs/tensor-pack-cse-v2-20261112T022228Z.json)
passes all 162 complete original/candidate checks and eight exact conversion
stress checks. Across nine recipes and three paired rounds, OA original/CSE
ratios range from 0.9897 to 1.0183; most are tied or about 0–1.4% faster, near
event resolution. One input-gradient round regresses. No case clears a 2%
improvement in every round, so the candidate remains experimental and is not
installed. The earlier
[`tensor-pack-cse-v1-20261112T022157Z.json`](runs/tensor-pack-cse-v1-20261112T022157Z.json)
also passes 162 checks but times original/candidate in separate groups;
use v2's shuffled four-arm comparison for the ablation decision.
