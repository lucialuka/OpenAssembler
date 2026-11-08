# CUBIN, ELF metadata and code splicing

## Why instruction words are not enough

A loadable cubin includes ELF sections, entry-point symbols, kernel parameter
metadata, resource declarations and code offsets. A modified instruction stream
can disassemble correctly and still fail to load or execute if its metadata is
wrong.

## Recovered production path

OpenAssembler keeps a ptxas-built container, replaces selected `.text.<kernel>`
bytes, updates exit-related offsets and repairs the register-count metadata
needed by the CUDA loader. A hardware register-count sweep found that the
declared count must cover the highest written register with the tested SM80
allocation rule. Both relevant `.nv.info` locations matter.

The implementation is intentionally conservative:

* unsupported ELF layouts fail closed;
* only owned kernel sections are replaced;
* unrelated files in an output directory are preserved;
* manifest paths are relative and checked against the output root;
* generated cubins are disassembled before publication when SASS output is
  requested.

## What this does not claim

This is not a from-scratch CUDA linker. Arbitrary device linking, unfamiliar
relocations, every fatbin packaging mode and later CUDA ABIs require separate
work. “CUBIN support” in the current code means the validated SM80 container
splice path.

See [Loadable cubins and metadata-safe code replacement](../research/ideas/cubin-metadata-splicing.md) and the implementation in
[`openassembler/container`](../../openassembler/container).

