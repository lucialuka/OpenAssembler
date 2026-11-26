# Mixed narrow scalar parameter layout

Status: original parser/form boundaries and actual toolkit layout audited;
no OpenAssembler candidate or production change.

The [original PTX](source.ptx) places adjacent u8/u8/u16/u32/u8/u64/u16
parameters in one entry and stores all six numeric argument values independently.
The [baseline probe](baseline-layout.json) retains the original scalar32/64 entry
parser rejection and all six narrow signed/unsigned/bit load-form rejections.
These are unsupported boundaries before execution, without an admitted numerical
miscompile claim.

Actual ptxas accepts the original source. The retained cubin, disassembly, complete
parameter metadata and log show offsets **0, 1, 2, 4, 8, 16, 24**, with **26 bytes**
in the parameter bank. The [independent layout audit](layout-audit.json) derives
natural scalar alignment/size from the original parameter declarations and matches
each observed ordinal/offset/size. It verifies exact raw artifact and observer
identities. This interpretation applies to this control, without a universal
metadata-format claim.

This toolkit listing uses both byte/halfword LDC and aligned word reads followed
by masks. A candidate can investigate extraction from existing aligned 32-bit
constant operands, avoiding an unvalidated new LDC encoding. It must preserve
every neighbor byte, signed extension, value snapshots, predicated definitions,
offset bounds and mixed 32/64 layout, and make both selector entry paths agree.
The current PTX interpreter's parameter-load branch also needs explicit narrow
signed extension before such forms can be admitted; matching an incomplete
interpreter would not be an independent reference.

Parser/form changes alone are insufficient. Verification packing, normal explicit
launch arguments, generated metadata and native ownership must share the exact
layout. Boolean source parameters then need a defined physical representation,
logical conversion and source contract. Boolean arrays/shared/buffers additionally
need a separate byte-storage contract. Independent complete-byte CPU/PTX/native
and A100 controls with adjacent arguments, original toolkit references, guards,
tails and full integration remain required before promotion.

The initial observer omitted the required instruction guard constructor argument;
its source/log remain retained. The repaired observer does not bypass the parser
or change production/admission/references.

```sh
python compiler_exp/gap_closure/python_boolean_abi/replay_parameter_boundary.py
python compiler_exp/gap_closure/python_boolean_abi/parameter_layout/audit.py
```

The [subsequent narrow-entry cycle](../../narrow_parameters/README.md) now admits
those physical scalar slots after original-PTX/A100 validation. Replay the exact
historical parser/form boundaries with
[`replay_parameter_boundary.py`](../replay_parameter_boundary.py); the original
probe is a dated pre-admission observation. Toolkit artifacts remain unchanged.
