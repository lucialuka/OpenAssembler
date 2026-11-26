# Scalar PTX/CUDA source-library resolution and integration

Status: validated production promotion for explicit scalar source libraries;
the original G16/G10/G42 and full 51-gap exit gates remain partial.
The canonical [device-function dossier](../../../../docs/compiler/gaps-device-functions.md#explicit-ptx-and-cuda-source-libraries)
owns the current contract and limits. The [preceding study](../README.md)
retains actual original two-unit compilation/linking and the first 48-launch
research hypothesis; those observations are not rewritten by this promotion.

## Gap and solution

`baseline-*.py` retains source state before public library input or general
source resolution existed. `link.py` stages deterministic PTX module resolution
before the existing direct inliner. Root entries call selected library helpers;
private symbols remain local/uniquely renamed, exact scalar signatures agree,
strong definitions win over weak definitions, and ambiguous/duplicate/unresolved
or recursive cases reject. Only reachable function definitions enter the closed
PTX module. Debug file/location indices remap together. General module data,
aggregate/register ABI and native object/archive relocation remain unsupported.

`candidate.py` generates standalone inliner/build/CUDA frontend/CLI/transport/
ownership modules. `--device-library SOURCE` is repeatable. PTX and CUDA source
libraries use the normal code path, with per-unit compilation-database flags,
working directory, source spelling, effective relocatable-device flags and
before/after dependency hashes. External remote libraries receive separate
bundle roots and original source identity. Complete link contracts are retained
in each kernel manifest and native ownership snapshot. Production imports no
experiment modules; tests use independent source/reference JSON.

## Experiments and retained failures

- `fixtures.py` covers direct/transitive/private calls, strong/weak resolution,
  identical weak bodies, debug files, mixed PTX versions, unused functions,
  wide slots and local frames. `candidate-offline.json` passes **720 complete
  executions**: ten configurations × four levels × two seeds × three lengths
  × PTX/normal/poisoned SASS. Independent full bytes include input immutability,
  guards and inactive tails.
- `first-candidate-offline.json` and its log retain a failed debug case: a
  `.loc` directive absorbed the following arithmetic in the inliner. The
  maintained repair masks the modeled three-number location prefix without
  deleting the instruction or changing newline count. A direct regression
  preserves the instruction. Earlier incorrectly escaped candidate-generation
  attempts remain in local logs; they supplied no hardware claim.
- The first private-name fixture omitted a local forward prototype needed by
  ptxas. `first-fixtures.py`, `first-oracle-probe.json` and their retained
  toolkit logs preserve the rejection. Corrected sources declare both private
  helpers explicitly; independent original-object linking then passes.
- `first-tests-path-failed.log` records a local compilation-database test whose
  temporary source was outside Colima's mounted tree. The final test uses the
  toolkit scratch directory. `local-worker-path-failed.log` similarly retains
  a Linux-worker replay placed outside the mounted tree; a test-only temporary
  directory override fixes that local replay without changing production.
- `dotted-fixture-failed.log` retains a malformed negative fixture that reused
  a formal name for a local call slot. A lexical block and distinct names fix
  the fixture; distinct dotted/underscored private names now pass.
- `oracle-probe.json` and `oracle-audit.json` preserve all ten original PTX
  object/nvlink controls, merged controls and **142 raw artifact identities**.
  Exact source, object, cubin, listing and tool logs remain under `oracle/`.
  Local toolkit is CUDA 12.8.61. Encoding/toolkit agreement is separate from
  independent hardware values.
- `candidate-a100-20261123T201424Z.json` stays failed after **48 passing direct
  launches**. A separately linked control's callee-stack metadata violated the
  harness's OA-only frame-pair assertion. `first-run-modal.py` and
  `first-hardware-frame-failed.log` preserve the original. The read-only
  `first-failure-audit.json` checks **344,064 partial bytes** and retains the
  failed outcome. The corrected harness records original linked driver
  resources independently and retains exact OA frame checks.
- `candidate-a100-20261123T201545Z.json` passes **576 launches**: twelve
  configurations × original separately linked control, merged-PTX control and
  four OA levels × four active lengths × two seeds. Ten PTX source cases are
  joined by CUDA/database and CUDA/serialized-remote-worker cases. Three CTAs
  of 128 threads compare all output/input/guard/tail bytes with independent
  wrapping32/64 references. Original controls have real separately linked
  helper text; OA emits complete owned entry text after source resolution.
  Actual device: A100-SXM4-40GB; worker Python 3.12.1/CUDA 12.8.1, with actual
  nvcc build metadata retained in the report.
- `promotion-audit.json` checks **4,128,768 complete bytes** and **48 exact
  native word/register/resource comparisons**. Link/symbol/ABI/body/edge,
  original-source/header, per-unit flag, remote origin and ownership contracts
  pass. OA's local fixture reserves 32 bytes; original controls keep their own
  observed stack reservations. A corrupted output control rejects.
- `normal-remote-audit.json` exercises the ordinary CLI client, real Modal
  compiler worker and publication with an external CUDA library/database.
  `remote-db-probe.json` separately retains the serialized local worker replay.
  `wheel-audit.json` verifies **129 installed source/data identities** and all
  forty-eight native comparisons from an isolated wheel subprocess.

## Integration and reproduction

`test_ptx_device_linkage.py` and `data/device_linkage_fixtures.json` are copied
as independent production regression inputs. They cover negative resolution/
ABI/storage/target/resource forms, deterministic signatures, private dotted
names, debug file IDs, changed headers, ordinary CLI ownership mutation and
external library bundling. The full suites, source/executable continuity and
live/exported documentation gates are bound in `validation.json`.

```sh
python compiler_exp/gap_closure/device_linkage/promotion/offline.py
python compiler_exp/gap_closure/device_linkage/promotion/oracle_probe.py
modal run compiler_exp/gap_closure/device_linkage/promotion/run_modal.py
python compiler_exp/gap_closure/device_linkage/promotion/audit.py compiler_exp/gap_closure/device_linkage/promotion/candidate-a100-20261123T201545Z.json /tmp/library-audit.json
python -m pytest -q tests/test_ptx_device_linkage.py
uv build --wheel --out-dir compiler_exp/gap_closure/device_linkage/promotion/wheel
python compiler_exp/gap_closure/device_linkage/promotion/audit_wheel.py
```

The supported local suite uses Python 3.12.10/Torch 2.7.1; separate Python 3.14.5
compiler-only skips remain explicit. Full original host linking/application
execution, binary device libraries, general weak-selection semantics, native
call stacks, recursion/indirect calls, register/aggregate ABI and module data
relocation retain their original exit gates. No performance claim is made.
