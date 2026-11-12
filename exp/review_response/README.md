# Response to the whole-repository review

The input review and its available raw summaries are preserved under input/.
The baseline is commit e071448. Experiments start here before production edits.
No Inductor, Triton or cuBLAS compute is used by candidate execution.

The first candidate captures a complete existing OA MSE/backward/SGD iteration
into a CUDA graph. Inputs, target, parameters, workspaces and learning rate bind
at capture. Input values may change in place; storage/stream/parameter mutations
reject. Capture performs no optimizer update. The mathematical graph, kernel
bytes, precision, launch order and number of kernels remain unchanged.

Admission requires four changing-input complete-state checks against CPU and
bitwise equality to ordinary OA dispatch, guard/poison/immutable-input tests,
an independently classified exact profiler activity multiset, and at least 2%
speedup in all three paired rounds on two isolated A100 worker runs. Resident
input host dispatch plus synchronization and CUDA event time are reported
separately. Setup/capture costs are separate. No NVIDIA performance claim is
inferred from an OA-versus-OA comparison.

The audit candidate classifies Chrome/Kineto activities by category. A kernel
whose name contains memcpy/memset is still compute and must match the ledger.
Negative tests preserve the baseline loophole and candidate rejection evidence.

The review's unique-kernel samples identify the Tensor Core model's one-thread
MSE (about 1.56 ms) as larger than any GEMM in that sample. The ordered-loss
candidate computes contiguous products in one warp, then broadcasts them to
lane 0 in increasing index order with full-warp XOR shuffles. It preserves
separate FP32 subtract/multiply and the scalar addition sequence. Masked tails
supply zero and both ends of every input/output allocation are guarded. There
is no parallel tree reduction or new tolerance. Shapes 1, 33, 112, 136, 513,
16384 and 65539 cover scalar/tail/real-model boundaries.

The build ablation compares fresh output directories with and without optional
nvdisasm listings. The existing encoded-text inventory and full bundle
validation remain mandatory. All deployed cubin bytes must match in each pair.
