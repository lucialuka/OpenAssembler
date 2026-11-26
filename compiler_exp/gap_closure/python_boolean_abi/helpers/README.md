# Logical Boolean helper signatures

Status: implemented and integration validated for pure logical Boolean helper signatures.

This G03/G04 subcycle admits `bool` only in pure, statically inlined `@device`
parameter/return annotations. The logical value stays a predicate; argument copies,
local reassignment and complete return paths use the existing predicate moves.
Every argument evaluates once, left to right, before the pure helper body. Numeric
truth conversion must be explicit with `bool(x)`; integer zero/one is not an
implicit Boolean argument or return. Existing helper purity, source resolution,
initialization, recursion and expansion limits remain required.

Kernel Boolean parameters, Array/Buffer/shared Boolean storage and device-call
stack ABIs remain separate, explicitly rejected boundaries. Annotation resolution
opts into logical predicates only at scalar helper signatures; nested Array/Buffer
element resolution does not inherit that opt-in. No byte parameter parser or
physical layout changes are part of this candidate.

Original source graph helper input/output signatures record `pred`, including a
same-file identity helper with no truth expression. The validator checks every
helper scalar signature before ownership acceptance. Experiments must cover pure
forwarding, early returns, nested imports, caller isolation, predicate loops,
explicit numeric conversions and eager/short-circuit argument effects using
independent complete-byte references, then actual ptxas/A100 and integration gates.

## Validation

The [canonical dossier](../../../../docs/compiler/gaps-boolean-helpers.md) records
this exact logical signature contract. The original [A100 report](candidate-a100-20261124T123953Z.json)
passes 2,400 launches; the independent [production audit](promotion-audit.json)
checks 16,220,160 complete bytes and 240 exact native word/register/frame outputs.
All 155 new regression tests pass before promotion; no existing test changes.

The [offline matrix](candidate-offline.json) passes 4,320 executions, generated
[helper trees](control-properties.json) add 128 cases / 1,536 executions, and all
22 [negative controls](negative-controls.json) reject for their original reasons.
Actual ptxas accepts every original source tree. Full suites pass **4,593 supported
tests** and **4,108 compiler-only tests with fifteen skips**. The installed wheel
checks 132 source/data identities and 240 original source/native regenerations;
the public Modal CLI preserves predicate signatures and native/source custody.
[validation.json](validation.json) binds all final source, evidence and log hashes.

The first test harness fixture-directory error and first auditor aggregate-count
error retain exact logs/source. Every individual numerical/native check passed;
no reference bits, tolerances or production semantics changed to repair observers.
The first continuity observer included changed frontend modules in its unchanged
backend list; the final observer explicitly isolates those three validated source
changes and retains every executable/backend/proof/archive identity check.

The [fresh prior-source replay](application-continuity.json) regenerates all 240
preceding Boolean-value outputs exactly; 3,254 earlier exact comparisons retain
their proof/archive and unchanged executable/backend input binding. Twenty source
files retain 62 accepted PTX kernels and unchanged rejected diagnostics. All fifty
inputs of the existing 49-policy/98-session audit remain exact, without policy or
measurement rewriting. Live and isolated-export documentation checks pass.
