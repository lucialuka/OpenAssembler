# Pointer parameter annotations and ABI custody

Status: implemented

G10 previously rejected every PTX kernel parameter with `.ptr` metadata. The
new parser accepts scalar `u64/b64` parameters marked `.ptr`, optionally
`.global` and a positive power-of-two `.align N`. Generic pointers omit the
space. Both spaced and adjacent modifier spellings are parsed. Other pointed
address spaces and aggregate-by-value layouts still reject; this adds no
general device ABI, linker or module storage lowering.

Pointer alignment describes pointed-to storage. Its default is four bytes,
following the [PTX kernel pointer-parameter contract](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#kernel-function-parameters).
It does not enlarge or rearrange the scalar parameter slot: a mixed `u32`,
`u64`, `u64` signature still uses offsets `0x160`, `0x168`, `0x170` on SM80.
Explicit `.align 1` remains a one-byte promise. The parser preserves each
pointer's space and alignment separately in `parameter_attributes`.

The normal build/manifest/native-ownership path carries this metadata. Proof
records take deep value snapshots of parameter lists, launch directives and
new contracts; editing a manifest object cannot also mutate its saved proof
through shared Python references. Standalone explicit launch specs require
buffer bindings for annotated pointers and check offsets against promised
alignment. The default allocator establishes at most 256-byte alignment;
larger promises require a specialized launch harness and are rejected or
skipped by generic planning. Automatic input planning respects pointer metadata
even when a parameter's name resembles an integer size.

The generic emulator checker recognizes an annotated `b64` pointer as a pointer.
Its legacy assumption about unknown element types and finite buffer sizes remains
explicit; annotation acceptance does not prove arbitrary memory access safety.
Checked shaped buffers use their separate metadata/launch contract.

## Research and validation

The [pre-change controls](../../compiler_exp/gap_closure/parameter_abi/candidate-oracle.json)
record CUDA 12.8 ptxas accepting annotated forms that OA rejected. The shared-space
control remains unsupported by this implementation; toolkit acceptance alone
never grants an OA lowering. Candidate implementation preserves original scalar
slot layout and exact instruction words for admitted generic/global pointers.

The [regressions](../../tests/test_parameter_pointer_abi.py) check syntax,
metadata, mixed-slot offsets, invalid layouts/spaces/alignment, actual packaged
ownership, exact annotated/unannotated native equality, deep-copy custody and
prelaunch alignment rejection. An initial negative test exposed shared-reference
aliasing between new metadata and its proof; its failed record remains under
`parameter_abi/` and production now snapshots the values. Independent review
corrected the initial one-byte default to the specified four-byte alignment.

The final [A100 gate](../../compiler_exp/gap_closure/parameter_abi/hardware-20261121T142037Z.json)
passes 32 launches: four generic/global `u64/b64` annotation variants, four OA
optimization levels, and OA versus same-PTX ptxas. All 12,288 output bytes,
inactive tails, both guards and input immutability pass. Every ordinary build
also reports successful emulator verification. The actual device is recorded
in the report with Python 3.12.1 and CUDA toolkit 12.8.1. Source, generated PTX,
manifest/native metadata, cubins, listings and full output bytes are retained in
the adjacent archive. This is correctness/ABI evidence without a speed claim.

```sh
python -m pytest -q tests/test_parameter_pointer_abi.py tests/test_ptx_entries.py tests/test_native_ownership.py
modal run compiler_exp/gap_closure/parameter_abi/run_modal.py
```

G10 remains open beyond this contract: parameter aggregates, new scalar ABI
widths, general function/prototype/linkage grammar, symbols, relocations and
cross-module linking each require their own source/container/runtime evidence.
