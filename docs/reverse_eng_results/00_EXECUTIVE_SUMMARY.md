# Executive summary

OpenAssembler was built by observing NVIDIA's public compiler and binary tools
as black boxes, recovering enough of the SM80 instruction and container model
to generate loadable A100 cubins, and validating each change against both
functional references and real hardware. The result is a native compiler for a
bounded PTX/CUDA/Python subset, with explicit fallback and evidence boundaries.

The most reliable reverse-engineered results are:

* The SM80 instruction corpus contains 541 standalone PTX forms, 13,816
  observed instructions and 88 opcode families. A 137-seed bit-flip campaign
  recovered common instruction fields, including control bundles, registers,
  predicates, memory operands and branch encodings.
* The production encoder is template- and field-map-based. It validates ranges
  before packing; unsupported or malformed forms fail closed.
* A ptxas-built ELF/cubin container is retained as the ABI and metadata source.
  OpenAssembler replaces selected kernel text, repairs exit offsets and updates
  register metadata. This is a tested SM80 splice path, not a full replacement
  for CUDA's linker or every fatbin feature.
* Corrected scalar A100 probes found 4-cycle same-pipe dependencies, 5-cycle
  cross-pipe dependencies and 13-cycle predicate-to-guard distances for the
  tested forms. These values are pair-specific and are not an HMMA model.
* Clocked tensor probes separated dependent accumulator latency from independent
  tensor initiation: approximately 24 cycles per HMMA for one dependent chain,
  12 for two independent chains and 8 for three chains after empty-loop
  subtraction.
* The production tensor path retains conservative HMMA8/drain15 and
  async-copy/LDSM/address-WAR protections. The generalized dependency scheduler
  remains experiment-only. Exact measured policy dispatch can select bounded
  fragment/layout policies, including load-next, grouped-N and 64×64 tiles;
  unknown or stale contracts keep the reference schedule. Optional A100 tuning
  validates and measures admitted candidates before writing a reusable record.
* Constant 32-bit integer division/remainder lowering is proven for its stated
  domain. A historical division-by-seven loop improved from 0.629 ms to 0.464
  ms; this is a scalar result, not a Tensor Core claim.
* The native GEMM path is correct under its full-tile FP16/BF16-to-FP32 contract,
  but matched ptxas remains faster on most measured shapes. Public CUTLASS and
  cuBLASLt techniques have been studied and some shape-specific experiments
  pass, but proprietary cuBLAS kernels were not extracted.
* The largest persistent performance signal is excess non-tensor work and weak
  overlap. OA and ptxas often execute equal HMMA counts while OA emits more
  address/control instructions and sees larger wait or scoreboard stalls.
* Native application capture is separate from the native compiler. A configured
  PyTorch application uses TorchInductor/Triton only when explicitly selected;
  the application report records that distinction.

The practical conclusion is precise: OpenAssembler has a validated native
compiler foundation and several conditional optimizations, but it does not yet
replace ptxas or cuBLAS across arbitrary transformer workloads. Every claim in
this folder is tied to a contract, device, toolkit and evidence record.

