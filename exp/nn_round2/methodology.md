# Prospective NN arithmetic experiment contract

This is a test plan inside the experiment workspace, not a results dossier.
[coverage.json](coverage.json) lists 23 arithmetic families, their native
support boundaries, candidate improvements and required checks. A missing native
implementation is a recorded coverage gap; a shared library operation is not
silently credited to OpenAssembler's SASS backend.

This contract belongs to the November 7–9 study. Documentation updated
November 11: the later [exact-shape production replay](../dispatch_tuning/RESULTS.md)
has its own frozen selection/heldout protocol and does not replace this
whole-workload design. Shape-tuned native kernels are still not automatically
substituted into ordinary captured PyTorch models.

The most informative control for a compiler change is **identical PTX compiled
by OpenAssembler and ptxas**, using the same buffers and launch shape. A vendor
operator answers a different question: it may choose a different parallel
algorithm, fused interface or tensor-core precision policy. Both comparisons
matter, but their conclusions must stay separate. The existing application
frontend uses Inductor/Triton; compare it to stock `torch.compile` as well as eager
CUDA before claiming an OpenAssembler-specific application compiler advantage.

## Whole-workload design

Use three residual MLP sizes and at least two attention/block configurations as
native-routed hybrid workloads. Use at least three decoder-transformer shapes
for the existing application route. Count the operations actually replaced by
native OA and list the shared CUDA stages. Each native linear must cover forward,
input-gradient and weight-gradient products. An attention score matrix alone is
not attention: include scaling/masking, softmax, multiplication by V and all
backward derivatives before using that name. The single-GPU transformer study
does not cover convolution or distributed collectives.

Save and compare complete outputs, loss, input gradients, raw parameter gradients,
clipped gradients if enabled, parameters, optimizer moments and step counters
after several identical updates. Use the same initialization, tokens/targets,
precision policy and optimizer settings. The old native MLP uses explicit FP16
operand rounding with FP32 accumulation/output and a custom straight-through
backward rule. It cannot be compared to ordinary autocast as if the derivative
contracts were identical. Its AdamW epsilon `1e-4` is a declared MLP policy, not a
repair for the separate BF16 transformer's failed epsilon-`1e-8` comparison.

Graph kernel timing must poison outputs **after** capture, replay and check every
element, with guards on both sides. A ctypes launch performed during capture can
otherwise appear fast because the intended work was not captured. A cuBLAS
handle must bind the active capture stream and restore its stream afterwards.
Do not repeatedly mutate optimizer inputs during isolated paired timing: use a
read-only input/update-output interface or an identical reset protocol.

Whole workloads include casts, transposed packing, padding, allocation, launch
dispatch, required saved activations, backward, clipping and optimizer work.
Report synchronized wall latency and CUDA event spans separately. Retain compile
and startup costs outside the steady-state boundary. Collect diagnostic stage
events in separate runs: those events perturb short kernels, and sums of stage
medians need not equal an independently measured total.

## Improvements worth testing first

1. Search per-shape GEMM pointer homes and CTA order with fresh holdout timing.
   The previous direct-pointer improvement regressed tall shapes, so automatic
   global selection would discard known evidence.
2. Pack immutable inference weights once, then reuse them, while measuring the
   packing amortization and restoring identical output/state checks. In training,
   every parameter update invalidates those packs.
3. Search two/four independent contiguous elements per thread for pointwise
   functions. This can hide instruction latency but may increase registers,
   reduce active warps or turn a bandwidth-bound operation slower. Include tails.
4. Test backward pointwise expressions and cooperative softmax/RMSNorm derivatives
   rather than limiting new experiments to forward y-only operators.
5. Fuse elementary bias/residual/activation operations only under the same
   mathematical and saved-output contract. A forward-only fused activation does
   not replace the activation data required by a training backward pass.

Any selected candidate needs a fresh timing round after tuning. Save all build,
correctness and performance failures. Keep source/PTX/cubin hashes, resources and
valid disassembly with each admitted kernel. If hardware counters are
unavailable, report that explicitly; shorter instruction listings do not prove
an operand-collector or scheduling explanation.

## Prior-study weaknesses this round must avoid

- Native `rowwise` is bounded finite FP32, width 1–1024, y-only and non-affine.
  LayerNorm saved statistics, affine parameters and backward are absent.
- The scalar `adam_step` example is an Adam-like update, lacking AdamW bias
  correction and decoupled decay. Do not give it an AdamW speed label.
- The scalar cross-entropy and attention-score examples failed old exact gates.
  A new independent numerical contract can assess current code, but does not
  retroactively change that historical failure.
- cuBLAS can pass a GEMM check yet fail a multi-update gradient/state gate.
  Retain its failed whole-workload row rather than widening tolerance.
- Native full-tile GEMM does not automatically cover arbitrary strides, tails,
  padding or batched head dimensions. Those costs and exclusions need a record.
