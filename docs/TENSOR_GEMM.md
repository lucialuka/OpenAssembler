# Native GEMM documentation index

The canonical contract, methods and reproduction commands now live in the
[Native SM80 tensor GEMM dossier](research/ideas/native-tensor-gemm.md).
This filename remains a compatibility entry point.

The independently documented components and explicit options are:

- [Paired fragment loads](research/ideas/paired-fragment-loads.md)
- [Address generation](research/ideas/tensor-address-generation.md)
- [Stage-pair specialization](research/ideas/stage-pair-specialization.md)
- [Shared layout/swizzling](research/ideas/shared-layout-swizzling.md)
- [Vectorized FP32 stores](research/ideas/vectorized-fp32-epilogue.md)
- [Operand reuse hints](research/ideas/operand-reuse-hints.md)
- [Morton8 traversal](research/ideas/morton-cta-order.md)
- [CUTLASS-style load-next fragments](research/ideas/cutlass-load-next-compute-current.md)
- [Grouped-N CTA traversal](research/ideas/grouped-n-cta-swizzle.md)
- [Fused bias/residual/ReLU](research/ideas/fused-gemm-epilogues.md)
- [Direct pointer homes](research/ideas/direct-pointer-homes.md)

Each dossier states actual positive and negative results, precision/layout
requirements, applicable commands and remaining limitations. No option is a
universal speedup or an automatic whole-model replacement.

[All CLI flags](COMMANDS.md) · [Architecture and latest results](STATUS.md) ·
[Complete idea catalogue](research/README.md)
