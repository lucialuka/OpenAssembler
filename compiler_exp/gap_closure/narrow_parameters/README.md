# Narrow scalar entry parameters

Status: implemented and integration validated for the scalar narrow entry contract.

## Gap and solution

The [preceding layout study](../python_boolean_abi/parameter_layout/README.md)
established actual mixed-byte toolkit slots and original parser/form rejections.
This G10/G11 experiment implements scalar b/u/s8 and b/u/s16 entry parameters,
natural 1/2/4/8-byte alignment, exact subword extraction and signed/zero extension.
It coordinates the parser, shared layout/load validation, production and legacy
selectors, PTX interpreter, verifier, explicit runner and batch-driver arguments.
Frozen baseline and reproducible candidate generator retain all eight modules.

Parameter loads must be naturally aligned and remain inside their declared scalar
slot; they may read narrower slices of a larger scalar. Register-valued parameter
addresses, aggregate/array/vector parameter ABI, explicit slot alignment overrides,
FP16/predicate slots, writes and narrow device-call slots remain unsupported.
Existing 32/64 parameter offsets, selection words, pointer attributes and source/
native custody must remain exact. Logical Python Boolean launch/storage remains
a subsequent frontend/storage contract.

Narrow extraction uses existing aligned constant-bank word reads, shifts and
masks. Signed reads move the selected sign bit to bit31 then arithmetic-shift;
unsigned/bit reads shift and mask. No new ISA template is admitted. Guarded reloads
and loop-carried writes keep existing virtual homes and skipped-write semantics.
The interpreter independently extracts the declared byte offset and widens signed
values into 32-bit registers. Verification packing uses matching scalar formats
and rejects unrepresentable structural dimensions. Explicit run arguments carry
exact u8/u16 byte widths after signed input range validation; the GPU batch driver
constructs actual ctypes scalars of those widths. Wrong-width bindings reject.

## Independent evidence

- [Frozen baseline](baseline.json): sixty original parser rejections.
- [Offline matrix](candidate-offline.json): 34,560 complete independent-buffer
  executions across sixty original entries, four levels, two seeds, six edge
  samples and four lengths through PTX/normal/poisoned engines. All byte slots,
  signed extremes, neighboring fields, guards, inactive tails, alias snapshots,
  predicate reloads, loops and larger-slot subword accesses are checked.
- [Layout properties](layout-properties.json): 4,096 deterministic independently
  packed mixed scalar layouts / 407,972 valid byte/halfword extraction controls,
  seed 78319. This is CPU layout/algebra evidence, without a new GPU claim.
- [Negative controls](negative-controls.json): fourteen genuine original ABI,
  width, offset, alignment, space, store and device-slot rejections.
- All 171 [candidate regressions](candidate-tests.log) pass, including both legacy
  entry points, generic verification, exact runner bindings and native custody.

The [actual A100 gate](candidate-a100-20261125T011357Z.json) passes 14,400 original
PTX launches and 240 actual explicit-run launches. The independent
[production audit](promotion-audit.json) checks **67,461,120 complete bytes** and
240 exact native word/register/frame outputs. All 49 measured policy binaries and
98 sessions remain exact before identity rebinding. The public Modal CLI build
preserves all sixty source signatures and native ownership. The [fresh complete prior replay](application-continuity.json) passes **3,734
direct native comparisons, every word/control/register exact**. Full suites pass
**4,764 supported tests** and **4,279 compiler-only tests with fifteen skips**.
The installed wheel checks 132 identities and 240 exact native regenerations;
twenty source files retain 62 accepted PTX kernels and unchanged rejections. Live
and isolated-export docs pass. [validation.json](validation.json) binds all final
source, proof, experiment and log identities.

## Retained observer failures

The first GPU harness imported an incorrect batch helper name and stopped before
launching. The next retained payload stopped at the public planner's tuple result;
the following observer compared a list of output spans directly to bytes. These
are harness API/shape errors, without evidence of a numerical mismatch. Exact
harnesses, logs and returned reports/archives remain retained. The corrected
observer unpacks the documented planner result and concatenates the actual byte
spans before independent comparison. Source, candidate and references are unchanged.

The original local temporary environment had lost its virtual-environment config
and packages. A new Python 3.12.10 environment restores pytest8.4.2/Torch2.7.1;
its installation log and first missing-test-dependency output are retained. No
production behavior or test expectation changes to resolve these observer issues.

The first layout-property observer incorrectly requested a halfword from a
one-byte slot, which is outside the defined access domain. Its exact source/log
remain retained. The corrected generator enumerates only accesses fitting each
slot; separate negative controls still require every cross-slot access to reject.
The independent byte/shift references and production access validator are unchanged.

The subsequent [logical Boolean scalar study](../python_boolean_abi/scalar/README.md)
retains three original source annotation rejections and separates canonical host
Booleans from raw u8. The [actual CUDA scalar boundary](../narrow_scalars/README.md)
now reaches unsupported setp.s16 operations in the exact original toolkit controls.
Both are research boundaries without a new production or numerical-support claim.
