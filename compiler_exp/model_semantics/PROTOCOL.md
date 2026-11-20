# Normalized model semantics research protocol

Frozen gate before A100 execution: FP32 inputs/state/gradients, exact positive
contiguous shapes, SM80 A100, OA-only arithmetic. Integer indices are int64;
attention masks are bool with True meaning allowed. No implicit casts, dropout,
GQA, padding-index gradients, frequency scaling, sparse embedding gradients,
ignore labels, label smoothing, class weights, AMSGrad or changing epsilon.
Cross-entropy means unweighted rank-two logits and integer labels, mean reduction.
Adam uses coupled decay; AdamW uses decoupled decay, bias-corrected moments,
epsilon outside sqrt, explicit beta/epsilon/decay settings and persistent state.

Predeclared numerical gate for hardware full tensors: rtol=1e-4, atol=1e-5.
CPU formula/autograd gate: rtol<=3e-5, atol<=3e-6. No sampling. Two changed-input
forward/backward iterations in all four view/reuse combinations; repeated token
indices, scalar/rank1..4 broadcasting, full masked rows, widths1/33/513 and
non-square causal attention. Eight changing-label/data CE updates per SGD,
Adam and AdamW; compare every parameter, gradient, first/second moment and step.
Resume after update4 and compare subsequent state. Invalid indices/labels must
reject before launching computation/updating weights. Workspace poison, allocation
canaries and native launch/source/binary provenance remain required.

Cache contract: separate prefill and one-token decode, explicit exact past-length
buckets, immutable FP32 cache input, append outputs, CPU orchestration owns state.
Unknown length and caller mutation reject before transition. Test prefill then
four successive decodes against full-prefix causal attention and complete caches.

## Candidate decisions

- Direct stride mapping for broadcast arithmetic versus materializing expanded
  operands: direct mapping avoids two extra memory passes and allocations;
  deterministic sum-to reductions enumerate only replicated dimensions.
- Masked softmax composed with incumbent OA BMM versus fused tiled attention:
  composed candidate first establishes exact gradients/mask edge cases; fusion
  requires an independently validated online-softmax/backward schedule. Existing
  BMM tensor/async dispatch is retained. Neither a paper nor CPU success admits
  a new memory/synchronization schedule.
- Embedding gradient deterministic per-weight reduction versus FP32 atomics:
  deterministic candidate provides fixed duplicate-index accumulation order;
  atomic and sorted/segmented candidates need separate determinism/crossover
  studies. Do not advertise the serial scan as a large-vocabulary speed solution.
- Three adaptive-update launches versus fused multi-output update: separate
  moments allow guarded checkpoint/state checks using existing runtime ABI.
  Fusion remains a separate latency experiment after this correctness gate.
- Exact KV length buckets with append copies versus paged/in-place cache:
  former can be validated with existing alias lifetime rules; latter needs
  page ownership, capacity/position checks and memory-hazard evidence.

This is a bounded semantic integration gate, not exhaustive implementation of
all conceivable proposals or arbitrary repository compilation. A failed candidate
stays experimental; production promotion requires successful hardware and CLI
checks. All numerical failures and harness failures are retained separately.
