# G17 aligned groups and predicate spills

Canonical methods, results and limits: [spilling dossier](../../../docs/compiler/gaps-spilling.md).

1. **Gap:** `baseline-spill.py` is the exact prior production allocator.
   `baseline.json` retains twenty failures across pair/quad, automatic pressure,
   predicates and combined declared-local cases at four compiler levels.
2. **Solution:** `spill.py` preserves the prior successful scalar plan, then
   retries with whole aligned groups. Predicate pressure uses explicit private
   words, existing SEL/ISETP forms, and unconditional temporary definitions.
   There is no generated encoding/spec change.
3. **Independent validation:** `offline.py` compares complete normal/poisoned
   SASS outputs with algebraic references. `controls.py` checks prior executable
   equality, alignment and impossible resource rejection. The staged regression
   reference derives predicate values from per-iteration truth tables.
4. **Hardware:** `run_modal.py` captures its own hash before dispatch and retains
   complete input/output bytes, guards, manifests, binaries and disassembly.
   The 320-launch A100 report and archive are independently rechecked by
   `audit.py` against candidate and production. All driver resources are kept,
   including ptxas exceeding the two 16-register requests.
5. **Promotion:** only `openassembler/backend/spill.py` changes. Tests and a
   frozen fixture snapshot enter `tests/`; normal build/CLI paths use the new
   allocator. Policy and application continuity compare retained native bytes
   before updating source identities. `validation.json` binds the final suites.

`first-probes.py`, `candidate-offline-first.json` and the corresponding log
retain an earlier passing, weaker predicate domain. The final probes add
self-guarded and negated definitions/uses. `controls-first.log` retains a
harness failure on existing unsupported `fma.rm.f32`; final controls preserve
and compare that rejection. Neither is counted as a separate hardware proof.

This implements bounded scalar/pair/quad/predicate spilling with at most 256
word slots and a 65,536-byte combined frame. Slot reuse, global scheduling and
occupancy/performance policy remain open. The original 51-gap program is not
complete. No timing claim is made for this extension.

These groups belong to the ordinary scalar backend. The specialized tensor
recipe retains its separate allocator and dependency rules; tensor-fragment
spilling requires its own research and hardware gate.
