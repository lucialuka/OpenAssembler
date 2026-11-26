# Original typed condition comparison chains

Status: implemented and integration validated for the bounded original-source contract.
The [canonical dossier](../../../../docs/compiler/gaps-comparison-chains.md) retains
source/type/effect/resource ownership and remaining G06 scope.

## Gap and solution

The [initial isolated study](../README.md) reproduced rejected chains and passed
finite condition/atomic/type/special-value gates. Its first chain lowering rejected
useful adjacent contextual literals. Production now shares the ordinary typed
relation emitter, evaluates each nonliteral operand once and snapshots the middle
value before proceeding. A false relation exits before later operands evaluate.
A pure literal retains the existing adjacent relation's exact contextual type
rule. Signed/unsigned/wide/FP32 kinds remain explicit; mixed runtime kinds reject.
Negation and while/and/or composition use ordinary branches and existing FP32 NaN
polarity. First-class Boolean values remain a separate unsupported contract.

Original source paths/functions/lines/columns, ordered operator kinds, specialized
expression AST digests and per-kernel counts flow through versioned source graphs
and native ownership, including standalone kernels without helpers/constants.
Imports, helper/caller scopes, static specialization, fresh source snapshots and
normal local/worker/installed paths retain their original contracts. No user
module/package executes. No backend/ISA/resource limit or vendor fallback changes.

## Independent gates

- [Frozen baseline](baseline.json): sixty original source/entry chain rejections
  under the exact pre-promotion frontend, not current-production false assertions.
- [Offline matrix](candidate-offline.json): **4,320 complete executions / 240 rows**,
  three source trees/twenty kernels/four levels/two seeds/three lengths with PTX,
  normal and poisoned SASS. Independent full-bit references check every output,
  atomic counter, unchanged input, guard and inactive tail.
- [Generated chains](control-properties.json): **128 deterministic 2..5-link
  chains / 1,536 complete executions**, all six relational operators, negation,
  contextual literals and per-thread atomic single-evaluation/short-circuit counts.
  Seed 86713, full original sources, inputs and references remain retained.
- [Negative controls](negative-controls.json): sixteen mixed-kind/literal/value/
  membership/identity/initialization/helper-effect/convergence rejections.
- [Context probe](context-probe.json): five adjacent literal type admissions.
  `ownership-probe.log` independently checks a standalone original chain graph.
- [Toolkit](oracle-probe.json): three original source trees/sixty kernels accepted
  by actual ptxas, with 22 exact raw source/cubin/listing/log identities.
- [Final A100](candidate-a100-20261124T113417Z.json): **2,400 launches**, three
  original source trees/twenty kernels, emitted PTX plus four OA levels, two seeds,
  lengths 0/1/259/384 and three 128-thread CTAs. All original sources/PTX/graphs/
  manifests/cubins/disassembly and complete before/after inputs/outputs/counters
  remain archived. NaN payloads, infinities/signed zeros, wide boundaries,
  literal bridges, nested helpers/returns and memory/atomic snapshots are explicit.
- [Candidate](candidate-audit.json) and [production](promotion-audit.json) audits:
  **15,482,880 complete independent bytes / 240 exact native word/register/frame
  outputs**, zero local frames, original source/expression/helper/ownership,
  inactive tails, immutable inputs and corrupted-output rejection.
- All **159 new tests** pass before promotion. Existing tests are unchanged;
  the earlier **920 frontend tests** pass under the candidate as well.

## Retained failures and limits

`first-standalone-ownership-failed.log` exposes the first candidate's omitted
source graph for a standalone chain without a helper/static value. The final
candidate records original expression ownership and reruns the complete hardware
source gate. The first passed 2,400-launch report, original candidate and harness
remain separate historical evidence. `first-membership-diagnostic-failed.log`
retains a fixture expecting the operator rejection before its unsupported tuple
operand; the fixture now checks that precise existing tuple diagnostic.
`first-context-path-harness-failed.log` retains a report-only string/Path error,
corrected without changing scalar/type/effect semantics.

The full supported Python 3.12.10/Torch 2.7.1 suite passes **4,271 tests**. The
separate Python 3.14.5/no-Torch suite passes **3,786 tests with fifteen skips**.
All **1,079 frontend tests** pass; no existing test input changes. The
[wheel audit](wheel-audit.json) checks 132 installed identities and 240 original
source/native regenerations. The [public Modal build](normal-remote-audit.json)
retains original chain graphs and native ownership. The
[frontend comparison](frontend-continuity.json) checks nineteen source files /
62 identical accepted kernels and unchanged rejection results. The
[application audit](application-continuity.json) newly regenerates 240 preceding
range/else original outputs exactly and separately inherits 2,774 earlier
direct/inherited occurrences under exact proof/archive and unchanged backend
input binding: **3,014 prior occurrences total**, with those counts explicit.
The [integration record](validation.json) binds final source/test/experiment/
proof/log identities and live/exported docs checks. General Boolean values,
dynamic range steps, arbitrary iterables/objects/effects and the complete
51-gap program remain incomplete. No speed claim is made.
