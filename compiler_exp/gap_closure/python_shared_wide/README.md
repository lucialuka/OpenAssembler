# Typed Python wide shared arrays and CTA convergence

Status: validated production promotion; full original G03/G09/G15 and 51-gap
exit gates remain open. The canonical contracts are in the
[wide-type dossier](../../../docs/compiler/gaps-python-types.md#wide-shared-arrays-and-native-cta-reconvergence)
and [CTA convergence dossier](../../../docs/compiler/gaps-collectives.md#cta-phase-reconvergence).

## Gap and candidate

The scalar shared-width backend existed, but `shared_array(i64/u64, count)`
rejected and the frontend assumed four-byte storage, alignment and index scaling.
`baseline-compile.py` and `baseline.json` retain the original source and twelve
rejections. `candidate.py` stages a standalone frontend with eight-byte elements,
natural alignment and mixed-declaration padding charged to the unchanged 48 KiB
limit. Booleans reject as counts before malformed PTX emission. Direct literal
index checks, producer diagnostics and barrier divergence checks remain active.

`oracle_probe.py` records exact Python source, emitted PTX, tool log, cubin and
disassembly for each original case. Local ptxas V12.8.61 preserves observed
1,024/1,044/49,152-byte layouts and offsets; later existing32 controls use 516
bytes. An initial oracle call used the wrong API; its failure log remains
retained. `initial-*` snapshots preserve the pre-boolean-check candidate.

## Experiment sequence and failures

- `candidate-a100-20261123T152334Z.json` failed before any launch: the harness
  expected manifest language `py` instead of `python`. `first-run-modal.py`
  retains the exact harness, with its failed log and partial archive.
- `candidate-a100-20261123T152514Z.json` executed 768 launches and failed 64:
  mixed layouts at all four OA levels, nontrivial active lengths. Every same-
  PTX control and homogeneous case passed. Guards and immutable inputs passed.
  Functional PTX/SASS models had passed all 864 complete original executions.
  The exact failed frontend, native baselines, source/reference and harness
  remain retained in `failed-*`/`baseline-*`.
- `candidate-a100-20261123T153328Z.json` adds a full scoreboard drain at BAR.
  It still fails the same 64 launches. `barrier-wait-*` snapshots retain this
  unsuccessful candidate. It was not promoted.
- `isolation-base_zero-20261123T154046Z.json` changes declaration order to put
  wide storage at offset zero; it still fails 64 launches. Its independently
  measured reservations reflect the changed layout rather than the original.
- `isolation-predication-20261123T153955Z.json` predicates eligible straight-line
  branches; all 768 launches pass. The narrower writer-only ablation,
  `isolation-writers-20261123T154238Z.json`, also passes all 768 launches.
- `isolation-sync_writers-20261123T154641Z.json` explicitly inserts BSSY B0 and
  BSYNC B0 around the conditional-writer joins; all 768 launches pass.
  `failure-isolation-audit.json` rechecks all **75,497,472 bytes** over these
  six executable reports, exact candidate/harness/reference identities and
  binary/output/input custody. Three failed reports remain failed.
- `first-convergence-*` preserves a bounded compiler pass and its successful
  `candidate-a100-20261123T155732Z.json`: 768 launches. The final pass uses
  iterative reverse graph analysis, avoiding Python recursion on long phases.
- The expanded phase fixture initially omitted a read-before-overwrite barrier.
  `racy-phase-fixtures.py` and `offline-phase-race-failed.log` preserve that PTX
  reference failure. The corrected source synchronizes before storage reuse.

The functional emulator chooses implicit reconvergence points. It cannot prove
native SIMT execution dependencies that the original compiler failed to emit.
A later success never relabels an earlier failed gate.

## Maintained solution and final validation

`convergence_helper.py` is copied as ordinary code into the generated backend
candidate. For a forward conditional phase, every path must reach the same next
CTA BAR without crossing an exit, another barrier or a backedge. One BSSY B0
opens the phase; BSYNC B0 immediately precedes BAR. B0 is reused after that
matching synchronization. Labels remap before encoding, outstanding scoreboards
are waited, and six-cycle convergence issue gaps are retained. Straight-line
and unproven regions keep their existing code. The existing encoder resolves
BSSY targets; no new encoding fields or generated tables are introduced.

`convergence-encoding.json` binds real-corpus B0 BSSY/BSYNC seeds and **56 varied
forward-target/control disassemblies**. `candidate-offline.json` passes **1,440
complete executions**: twenty cases × four levels × two seeds × three active
lengths × PTX/normal/poisoned SASS. References use independent modulo64 Python
arithmetic, signed interpretation and complete byte buffers.

`candidate-a100-20261123T162930Z.json` passes the final **1,280 launches** over
ordinary `.py` builds, all four levels, OA/same-PTX controls, four lengths and
two seeds. Three CTAs of 128 threads cover cross-warp exchange, signed/unsigned
wide edges, literal indices, mixed declarations, exact 48 KiB boundaries,
nested/if-else writers, repeated phases and existing 32-bit shared elements.
Actual device is A100-SXM4-40GB; worker Python 3.12.1/CUDA 12.8.1/ptxas V12.8.93.
All source, generated PTX, manifests, complete guarded inputs/outputs, cubins,
disassembly and actual driver resource attributes remain archived.

`promotion-audit.json` checks **20,971,520 retained bytes** and **80 exact native
word/register comparisons**, actual shared reservations, zero local frames and
manifest ownership. A corrupted output control rejects. `wheel-audit.json`
replays the same eighty native comparisons from an isolated built wheel and
checks every packaged source/data identity. `frontend-continuity.json` compares
old/current isolated compilation of fourteen existing files; **62 kernels**
retain exact PTX. Full suite and prior-native evidence are bound in
`validation.json`. No timing or performance claim is made.

Staged tests and JSON references are copied as independent production inputs;
they import no experiment modules. They include count/resource/index/producer/
convergence rejections, source diagnostics, token lifetime, label remapping,
long nonrecursive phases and ordinary Python builder ownership. The old wide-
shared rejection is replaced by an oversize-resource rejection, retaining the
original test source.

## Reproduction and limits

```sh
python compiler_exp/gap_closure/python_shared_wide/offline.py --candidate
python compiler_exp/gap_closure/python_shared_wide/convergence_encoding.py
modal run compiler_exp/gap_closure/python_shared_wide/run_modal.py
python compiler_exp/gap_closure/python_shared_wide/audit.py compiler_exp/gap_closure/python_shared_wide/candidate-a100-20261123T162930Z.json /tmp/python-shared-audit.json
python -m pytest -q tests/test_pyfront_shared_wide.py tests/test_pyfront_wide.py
uv build --wheel --out-dir compiler_exp/gap_closure/python_shared_wide/wheel
python compiler_exp/gap_closure/python_shared_wide/audit_wheel.py
```

The supported local suite uses Python 3.12.10/Torch 2.7.1; compiler-only Python
3.14.5 has separately recorded skips. Dynamic indices/allocation extents,
initialization ordering and race freedom remain caller contracts. Partial/
counted barriers, cyclic/irreducible convergence, multiple B-token allocation,
FP64 arithmetic and general wider Python types need separate evidence. The
pass's proven forward regions do not make another control-flow path safe.
