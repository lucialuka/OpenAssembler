# G12 scalar shared-memory widths

Status: validated production promotion for scalar byte/halfword/64-bit memory;
G12/G09 and the original 51-gap program remain incomplete.

The canonical contract, scheduling discovery and limits are in the
[memory-width dossier](../../../docs/compiler/gaps-memory.md#scalar-shared-byte-halfword-and-64-bit-memory).
This directory retains the research sequence and exact implementation/evidence.
Production modules and regression tests import no experiment files.

## Gap and solutions

`oracle_probe.py` retains seven toolkit-accepted shared exchanges rejected by
the baseline selector. Existing LDS narrow/64-bit and STS.64 forms are usable;
four narrow-store forms lack templates. `recover_templates.py` uses exact
oracle words with the ordinary field learner, not guessed fields or hand-edited
generated tables. `candidate-templates.json` holds the recovered rows, now
copied exactly into `specs/sm_80/memory_templates.json`. `candidate.py` produces
standalone selector, admission, encoder and compiler candidates from the
retained `baseline-*.py` modules. Normal production reads the same data without
installing experiment overrides.

The first raw-offset fixtures could let donor metadata discard the unused
shared reservation; their `first-*` sources/reports and `first/` artifacts
remain separate. Final fixtures materialize `arena`. A recovery script first
called an incorrect control-decoder API; its failure log remains retained. The
working script uses `word.control` and reproduces exact complete control words.

## Experiments and validation

- `template-recovery.json` checks four exact oracle forms and unique mappings.
  `field-sweep.json` checks 12,096 register/offset/scale/guard disassemblies.
  Deliberately misaligned offsets are encoding controls, not executions.
  Negative-offset and zero-base disassembly spellings are normalized only for
  semantic comparison. Initial failed comparison logs remain retained.
  `template-audit.json` binds the complete oracle source/cubin/listing/log bytes.
- `width-matrix.json` retains 336 global/shared scalar/vector type rows over
  register/parameter-backed operands. All 172 admitted representatives encode;
  164 unsupported rows reject at admission. Numerical/toolkit/hardware stages
  remain explicitly not run for these separate generated fixtures. The
  read-only `width-matrix-audit.json` replays all 172 exact native encodings
  and validates every retained fixture, rejection and stage count.
- `candidate-offline.json` passes 1,368 complete PTX/normal/poisoned SASS
  executions: nineteen fixtures × four levels × two seeds × three active
  lengths × three engines. Independent references compare every output and
  adjacent shared byte, immutable input, guards and inactive tails.
- `candidate-a100-20261123T143055Z.json` is a failed toolkit gate, with zero
  launches: PTX `[addr-10]` rejected. `negative-syntax-oracle.json` retains
  accepted/rejected spellings; final fixtures use `[addr+-10]`.
- `candidate-a100-20261123T143647Z.json` remains failed: 90 of 960 launches,
  all optimized OA scalar 64-bit nonempty exchanges. Safe OA and same-PTX
  controls passed. The first read retained initialization while later full
  shared snapshots were correct. `failed-execution-audit.json` checks all
  27,525,120 retained bytes and preserves the failed outcome. Exact failed
  candidate/harness/probe snapshots remain under `failed-*`.
- The candidate retains at least six cycles at BAR in ordinary LDS/STS.64
  code, preserving the safe/toolkit issue gap instead of the scalar model's
  one. `candidate-a100-20261123T144011Z.json` passes the initial 960 launches.
  Its `barrier-fix-*` snapshots preserve that narrower fixture domain.
- `candidate-a100-20261123T144327Z.json` passes the final 1,216 launches,
  expanding to nineteen fixtures: false-load preservation, aliased wide
  destination, raw FP64 edges and existing vector-two 32-bit control included.
  Four levels, two compilers, four active lengths and two seeds each run on
  three 128-thread CTAs. All threads reach every barrier. Device is actual
  A100-SXM4-40GB, worker Python 3.12.1/CUDA 12.8.1/ptxas V12.8.93; local
  recovery uses ptxas V12.8.61. Source, harness, complete outputs/inputs,
  manifests, native/same-PTX cubins, disassembly and resources are archived.
- `promotion-audit.json` independently checks all 34,865,152 bytes and 76
  exact native words/register comparisons against promoted production. The
  shared reservation is 2,048 bytes, local frames zero, and ownership valid.
  Its corrupted-output control rejects. `wheel-audit.json` verifies exact
  source/data packaging and the same 76 native comparisons from an isolated
  built-wheel installation.

All success/failure JSON and zip archives retain their original identities.
A later success does not relabel a failed gate. `hardware-*.log` retain local
and remote run diagnostics. No timing or performance claim is made.

## Integration and remaining gates

`test_backend_shared_widths.py` and `data/shared_width_fixtures.json` are copied
as independent production regression inputs. The old unsupported shared-u8
assertion is replaced by a genuinely unsupported vector-u8 assertion; its
original source is retained. `validation.json` binds the final supported and
compiler-only suites, docs/export checks, source/data identities and exact
policy/application continuity proofs.

Literal offsets and direct symbol size/alignment have compile-time checks.
Dynamic alignment, initialization, bounds and race freedom remain caller
invariants. Ordered/volatile shared memory and vectors of narrow/64-bit elements
remain unsupported. This extends G12 without claiming general memory safety,
FP64 arithmetic or a universal barrier minimum. Async-copy/LDSM dependencies
are unchanged and keep their own contracts.

```sh
python compiler_exp/gap_closure/shared_widths/audit_templates.py
python compiler_exp/gap_closure/shared_widths/offline.py
modal run compiler_exp/gap_closure/shared_widths/run_modal.py
python compiler_exp/gap_closure/shared_widths/audit.py compiler_exp/gap_closure/shared_widths/candidate-a100-20261123T144327Z.json /tmp/shared-audit.json
python -m pytest -q tests/test_backend_shared_widths.py tests/test_backend_memory_widths.py
uv build --wheel --out-dir compiler_exp/gap_closure/shared_widths/wheel
python compiler_exp/gap_closure/shared_widths/audit_wheel.py
```

Use Python 3.12.10/Torch 2.7.1 for the supported full suite. Native continuity
commands and docs checks are recorded separately; they do not rerun hardware.
