# Extended native SGD and adaptive optimizer policies

Status: implemented

G34 identified optimizer options that were rejected or lacked native state
semantics. The direct OA runtime now implements SGD momentum, dampening,
Nesterov, coupled weight decay and maximize, plus Adam/AdamW AMSGrad and
maximize. Parameter groups, sparse/complex state, differentiable updates,
explicit foreach/fused vendor execution and broader optimizer families remain
outside this contract.

## Exact policy and state

| Family | Admitted options | Persistent per-parameter state |
|---|---|---|
| SGD | Nonnegative finite `momentum`, `dampening`, `weight_decay`; Boolean `nesterov`, `maximize` | `momentum` when momentum is positive; none otherwise |
| Adam | Existing betas, epsilon and coupled decay; Boolean `amsgrad`, `maximize` | `m`, `v`, plus `max_v` for AMSGrad |
| AdamW | Same adaptive options with decoupled decay | `m`, `v`, plus `max_v` for AMSGrad |

SGD initializes its first momentum buffer to the effective gradient directly;
dampening applies only on subsequent updates. Nesterov requires positive
momentum and zero dampening. Maximize reverses the raw gradient before coupled
decay; it does not reverse regularization. Each step, including zero learning
rate or zero gradient, still advances the admitted optimizer state. AMSGrad
tracks the maximum uncorrected second moment and applies the ordinary
step-dependent bias correction to that maximum.

Weights, gradients, momentum, moments and maxima remain FP32. Epsilon remains
outside the square root; no epsilon, precision, decay or tolerance substitution
is made. State is owned by the runtime and retained across calls. The existing
CPU snapshot format now records complete policy-specific `momentum` or
`m/v/max_v` entries under its `moments` key, plus canonical options and step.
Loading checks the entire inventory, shapes, CPU ownership, finiteness and
nonnegative variance state before allocation. An AMSGrad maximum must dominate
its second moment. Snapshot tensors are cloned; caller mutation cannot alter
loaded device state. Parameters still need to be saved separately for this
optimizer-only snapshot API; the separate [complete checkpoint route](gaps-checkpoints.md)
now saves model, optimizer, CPU RNG and explicit application state together.

Default SGD still normalizes to exactly `{"name":"sgd"}`, and old default
Adam/AdamW policies omit false extension flags. The ordinary default SGD kernel
name/PTX identities are unchanged across four retained shapes, including a
scalar and 129-element tail. Default plans still omit an optimizer field.
Compiler source changes intentionally invalidate the whole bundle cache safety
key; this extension does not bypass compiler identity to force old cache hits.

All nondefault optimizers currently require direct execution. The existing
plain-SGD CUDA Graph route remains separate. AMSGrad explicitly requires
`optimizer_fusion=false`; no fused AMSGrad option is silently accepted. Existing
split/fused Adam and AdamW paths both implement maximize. Graph replay of the
new mutable state needs a separate capture/step-coefficient contract.

## Research cycle and evidence

The [independent recurrence candidate](../../compiler_exp/gap_closure/optimizers/candidate.py)
first compared **192 complete updates** with CPU Torch: eight policies, 24
updates each, changing/tiny/zero gradients, a zero-rate update, and resume after
update 12. The [candidate record](../../compiler_exp/gap_closure/optimizers/candidate-results.json)
fixes seed 34052, Python 3.12.10 and Torch 2.7.1. Production promotion followed
that gate. Extended SGD kernels then passed encoding and SASS emulation against
that recurrence, including poisoned registers.

The [combined regression gate](../../compiler_exp/gap_closure/optimizers/cpu-results.json)
passes **538 tests**, covering all earlier model semantics plus policy
normalization, API/config propagation, full CPU reference progression,
pre-allocation invalid-state rejection and graph/fusion rejection. The
[plain-SGD baseline](../../compiler_exp/gap_closure/optimizers/plain-sgd-before.json)
retains the four original kernel names and PTX hashes. The initial regression
exposed three obsolete AMSGrad/maximize rejection assertions; their
[raw failure log](../../compiler_exp/gap_closure/optimizers/initial-regressions.log)
is retained. Those controls now reject invalid Boolean flags or unsupported
fused AMSGrad, paired with positive numerical tests for the new options.

The initial [A100 production record](../../compiler_exp/gap_closure/optimizers/hardware-20261121T131316Z.json)
passes **2,612 full-tensor comparisons / 336,948 elements** on NVIDIA
A100-SXM4-40GB, Python 3.12.1, Torch 2.7.1+cu126, CUDA runtime 12.6 and toolkit
12.8.1. Ten policy/fusion variants each run 24 changing-gradient updates and
resume from update 12. Every parameter, admitted state tensor and step is
checked against independent CPU Torch. Poisoned intermediates, allocation
canaries and caller-input immutability pass. A separate prescribed 25th update
per variant has an independent Kineto ownership audit and matching CPU weight
update; these ten profiles are diagnostic, without a performance claim.
The fixed gate remains rtol=4e-5, atol=5e-6; maximum observed absolute difference
is 8.556526154279709e-8.

Four ordinary CLI compile/saved-run applications each validate six complete
updates, with changed inputs, full state and zero vendor compute:

| Policy | Build checks | Saved-run checks |
|---|---:|---:|
| SGD Nesterov / maximize / decay | 96 | 96 |
| Adam AMSGrad / maximize / decay | 144 | 144 |
| AdamW AMSGrad / decay | 144 | 144 |
| Fused AdamW maximize | 120 | 120 |

The [archive](../../compiler_exp/gap_closure/optimizers/hardware-20261121T131316Z.zip)
retains 14 bundle plans, PTX/cubins, ledgers, full compared tensors, traces and
CLI reports. Its SHA-256 is
`801ba42544937711172058cae2311ce3bf5d07aed978012a85b52c6a7bd41d1c`.
The [audit](../../compiler_exp/gap_closure/optimizers/audit.json) validates those
identities, candidate/harness identities and all ten changed production source
hashes against CPU and the later
[current-host revalidation](../../compiler_exp/gap_closure/checkpoints/hardware-20261121T140736Z.json).
That fresh A100-SXM4-40GB run repeats the same 2,612 finite comparisons and all
CLI gates after the complete checkpoint integration and maximum fix below.
The initial audit is [retained separately](../../compiler_exp/gap_closure/optimizers/initial-audit.json).
The small
[application source](../../compiler_exp/gap_closure/optimizers/repo/model.py)
is retained with its transported source hash.

Independent review identified that PTX `max.f32` ignores one NaN operand, unlike
Torch maximum. AMSGrad now explicitly selects a quiet NaN when either input
is NaN and retains the left value on equality. A
[dedicated hardware gate](../../compiler_exp/gap_closure/checkpoints/maximum-20261121T140047Z.json)
passes 80 full-state comparisons / 1,360 elements for Adam and AdamW AMSGrad
maximize under NaN/Inf gradients, including every weight/moment/maximum
classification against CPU Torch, exact IEEE checkpoint self-roundtrip and
two independent optimizer-region ownership audits. Signed-zero ties have a
deterministic left-operand native contract; Torch CPU scalar and vector maximum
paths can choose different signed-zero bits. No cross-implementation NaN
payload equivalence is claimed. The earlier finite record remains historical;
the fresh gate binds the corrected maximum and current host sources.

## Use and reproduction

In `[model]`, declare the SGD controls directly, for example
`optimizer="sgd"`, `momentum=0.9`, `nesterov=true`, `weight_decay=0.02` and
`maximize=true`. For adaptive state, use `optimizer="adamw"`, `amsgrad=true`
and `optimizer_fusion=false`, with explicit betas/epsilon/decay as needed.
The library `compile_model(..., optimizer={...})` uses the same policy mapping;
SGD options are retained instead of being dropped because its family name is
`sgd`.

```sh
python compiler_exp/gap_closure/optimizers/candidate.py
python compiler_exp/gap_closure/optimizers/cpu_gate.py
modal run compiler_exp/gap_closure/optimizers/run_modal.py
python compiler_exp/gap_closure/optimizers/audit.py
```

The last command audits the fixed retained hardware record. A fresh Modal run
creates a new dated result and does not replace the original. This extends G34
under explicit dense FP32 contracts. The separate complete checkpoint work
addresses the supported owned-state artifact; G36 adaptive graph replay and
arbitrary optimizer coverage remain open.
