# Narrow scalar entry ABI

Status: implemented and integration validated for scalar b/u/s8/16 entry parameters.

## Gap and required behavior

G10/G11 reject scalar byte/halfword PTX entry parameters despite admitted narrow
global/shared storage forms. This also blocks a faithful physical Boolean kernel
argument route. The [original toolkit study](gaps-boolean-abi.md) establishes
mixed-width natural scalar slots; the [current experiment](../../compiler_exp/gap_closure/narrow_parameters/README.md)
implements and validates those slots before a Python Boolean ABI can use them.

Scalar b/u/s8 and b/u/s16 parameters require one/two-byte storage and natural
alignment alongside existing four/eight-byte scalars. Loads return zero-extended
bit/unsigned or sign-extended signed values in 32-bit registers. A narrower load
from a larger scalar can access a naturally aligned byte offset only within that
declared slot. Reads must preserve neighboring argument bytes, guard polarity,
old values under skipped writes, alias snapshots and loop-carried definitions.
Existing 32/64 offsets, pointer attributes and source/native identities remain
required. Aggregates, parameter arrays/vectors/writes, FP16/predicate slots,
explicit scalar alignment overrides and narrow device-call slots remain open.

## Coordinated solution

The candidate parser opts into narrow types only for entry declarations; static
device signature/slot parsing retains its existing boundary. One scalar layout/
load validator supplies both selectors. Existing aligned constant operands,
shifts and masks implement extraction and extension without new encoding templates.
The established allocator/scheduler handles explicit narrow result operations and
guarded homes. The legacy selector uses the same layout and corresponding physical
register operations; unsupported legacy predication remains explicit.

The candidate interpreter extracts the requested offset and sign-extends before
writing a 32-bit result. Matching packing and independently generated narrow
values enter the generic verifier; a structural dimension that cannot fit the
declared narrow type fails explicitly. The normal run specification accepts exact
signed/unsigned byte/halfword scalar fields, validates range and actual parameter
width, then the batch driver constructs matching ctypes arguments. Pointer/buffer
bindings cannot occupy scalar narrow slots. Existing manifests already carry exact
parameter declarations; ownership snapshots remain required.

## Current experiment evidence and remaining gates

The [baseline](../../compiler_exp/gap_closure/narrow_parameters/baseline.json)
reproduces all sixty original source parser rejections. The
[offline matrix](../../compiler_exp/gap_closure/narrow_parameters/candidate-offline.json)
passes 34,560 complete-buffer executions, including every byte/natural-halfword
slot position, both signed extremes, neighbor bytes, larger-slot slices, loops,
predication and aliasing through four levels and PTX/normal/poisoned engines.
All [fourteen negative controls](../../compiler_exp/gap_closure/narrow_parameters/negative-controls.json)
reject and 171 candidate regressions pass.

The [original A100 gate](../../compiler_exp/gap_closure/narrow_parameters/candidate-a100-20261125T011357Z.json)
passes **14,400 original PTX launches** and **240 real explicit-run launches**.
The independent [production audit](../../compiler_exp/gap_closure/narrow_parameters/promotion-audit.json)
checks **67,461,120 complete bytes** and **240 exact native word/register/frame
outputs**. Actual ptxas controls retain the complete original PTX and ordinary
local builds. No reference value, tolerance, template or resource limit changes.
All 171 new tests pass; existing tests remain unchanged. The public Modal CLI
preserves every original signature and native ownership. All 49 measured policy
binaries/controls/resources and 98 retained sessions remain exact before compiler
identity rebinding. A [fresh full prior native replay](../../compiler_exp/gap_closure/narrow_parameters/application-continuity.json)
passes **3,734 direct comparisons**, every word/control/register count exact;
no executable result is inherited from a source identity. The full Python
3.12.10/Torch 2.7.1 suite passes **4,764 tests** with zero skips. Python 3.14.5
without Torch passes **4,279 tests with fifteen skips**. The installed wheel checks
132 source/data identities and 240 exact native regenerations; twenty source files
retain 62 accepted PTX kernels and unchanged rejections. Live and isolated-export
documentation checks pass. The [integration record](../../compiler_exp/gap_closure/narrow_parameters/validation.json)
binds the final source, proof, experiment and log identities. Retained observer API/shape/environment failures are
documented in the experiment; no numerical reference or tolerance changes.

This is a prerequisite for [Boolean launch/storage ABI](gaps-boolean-abi.md), whose
logical type metadata, canonical representation and byte arrays/shared/buffers
still need their own original-source gates. The full 51-gap program is incomplete.

The additional [layout property gate](../../compiler_exp/gap_closure/narrow_parameters/layout-properties.json)
checks 4,096 mixed scalar layouts and 407,972 independent valid byte/halfword
extractions with seed 78319. It checks CPU layout/algebra without a new hardware
claim. The first observer requested halfwords from one-byte slots; its retained
correction limits generation to the declared valid domain while original negative
cross-slot rejection controls stay required.

The next [logical Boolean scalar study](../../compiler_exp/gap_closure/python_boolean_abi/scalar/README.md)
retains three original frontend rejections after physical byte slots are admitted.
It defines canonical host values, logical predicates versus bytes and parameter
metadata/native custody before any subsequent candidate or production extension.

Replaying the original independent CUDA Boolean sources now reaches the narrower
[setp.s16 scalar-operation boundary](gaps-narrow-scalars.md), with original toolkit
artifacts and current diagnostics retained. This entry ABI implementation does not
claim those remaining CUDA scalar operations or Python Boolean storage are supported.
