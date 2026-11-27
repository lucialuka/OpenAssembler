# Direct scalar PTX device-function specialization

CLI examples below run inside the shell started with `openassembler`; enter
commands such as `compile`, `run` and `cnr` without repeating the executable name.
Homebrew, Python, Modal and SSH setup commands run in your regular terminal.

Status: implemented for direct scalar parameter calls and explicit source libraries;
G16 and G10 remain partial beyond the contract below.

## Gap and solution

The entry-local parser rejected direct calls after nvcc successfully emitted
valid `__noinline__` device functions. The
[original CUDA observation](../../compiler_exp/gap_closure/device_functions/noinline.cu)
and [actual emitted PTX](../../compiler_exp/gap_closure/device_functions/noinline.ptx)
show `.func` signatures, `.param` input/return slots, call-local lexical blocks,
`st.param`, `call.uni` and `ld.param`. This also exposes an independent
`selp.b64` lowering limit, subsequently repaired by the [wide-selection cycle](gaps-wide.md#scalar-64-bit-conditional-selection); parsing functions does not admit every instruction
in them. A separate [supported CUDA fixture](../../compiler_exp/gap_closure/device_functions/nvcc_scalar.cu)
validates the complete noinline route. The
[baseline](../../compiler_exp/gap_closure/device_functions/baseline.json)
retains fifteen rejected cases.

The [maintained inliner](../../openassembler/ptx/inlining.py) resolves direct
calls and specializes their bodies before ordinary instruction selection.
Every invocation copies argument slots, renames all registers/predicates,
labels and static local objects, and converts returns into branches to its
own continuation. Lexical scopes preserve repeated slot names emitted by nvcc.
Caller aliasing and false guards retain exact values. Forward declarations
match by signature types; the defining parameter names bind the body. Recursion
and unresolved reachable functions reject with a call-chain diagnostic.

| Contract | Admitted domain |
|---|---|
| Call form | Direct `call` or `call.uni` to a definition in the same PTX module; optional instruction guard |
| Inputs | Up to 32 scalar `.param` slots of `b32/u32/s32/f32/b64/u64/s64/f64`, matching widths, zero offsets |
| Return | Void or one scalar `.param` slot, definitely initialized on every reachable return |
| Argument semantics | Snapshot copied into fresh formal storage at each call; input formals read-only |
| Caller storage | Lexically scoped scalar slots; reads require definite initialization, including false guarded-call paths |
| Control flow | Branches, loops, early/predicated returns and nested direct calls supported by the ordinary scalar backend |
| Resources | Depth at most 32; at most 100,000 expanded statements; ordinary register/spill limits and total static local frame at most 65,536 bytes |
| Callee storage | Fresh per-invocation static local objects; existing alignment/load/store/resource contract applies |

Scalar slot types describe bit-preserving parameter transfer; they do not add
FP64 arithmetic or unsupported modifiers/operations. Parameter addresses cannot
escape through ordinary register/address use. Aggregates, register-parameter
ABI, multiple `.param` returns, recursion, indirect calls, device shared/module
storage, native call stacks and cross-module/library linking remain unsupported.
The toolkit's actual multiple-return rejection is retained, and that form
rejects in the candidate. Unused external declarations remain inventoried;
a reachable unresolved call fails. This is a static inlining strategy, not a
native CALL/RET ABI or general device linker.

The [parser](../../openassembler/ptx/parse.py) retains definition and per-entry
expanded call inventories and preserves original instruction lines. The
[builder](../../openassembler/build.py) records signatures, body hashes and
call counts in each kernel manifest. [Ownership validation](../../openassembler/native_ownership.py)
binds that inventory to the generated executable. All non-entry executable
bytes must still satisfy the ordinary complete OA ownership rule; no vendor
helper is silently retained. Existing local-frame code reserves OA's own
static inlined frame even when ptxas optimizes its control frame differently.

## Experiments and validation

The [experiment](../../compiler_exp/gap_closure/device_functions/README.md)
retains isolated candidate/baseline sources, failed guard-name renaming,
toolkit ABI rejection, prototype/linkage and source-parameter harness mistakes,
and an independently corrected snapshot reference. A prototype declared
`.extern` conflicted with its local definition in the actual toolkit; matching
bare forward declarations pass. These failures retain their original artifacts.

The final [offline gate](../../compiler_exp/gap_closure/device_functions/candidate-offline.json)
passes **1,080 complete executions** over fifteen sources, four levels, two
seeds, three lengths and PTX/normal/poisoned SASS engines. References use
independent fixed-width arithmetic and exact simple FP32 values. Every output
byte, input byte, outer guard and inactive lane is compared. Functional
interpretation executes expanded instructions, so its result is not independent
proof of the expansion; original-source toolkit and hardware controls remain
separate requirements.

The void-call probe initially overwrote the helper effect in its caller. Its
weaker successful report remains retained separately. The final source removes
that overwrite, and a [negative control](../../compiler_exp/gap_closure/device_functions/void-effect-negative.json)
that deletes the callee store now fails the independent full-output check. The
final hardware gate uses this observable-effect fixture; production code is
unchanged by that validation repair.

The [local oracle audit](../../compiler_exp/gap_closure/device_functions/oracle-audit.json)
binds fifteen actual PTX sources, cubins, listings, tool logs and executable
inventories under ptxas V12.8.61. The
[final A100 gate](../../compiler_exp/gap_closure/device_functions/candidate-a100-20261123T181957Z.json)
passes **960 launches**: fifteen sources × four compiler levels × OA and
same-original-PTX controls × four active lengths × two seeds. Three CTAs each
launch 128 threads. Cases cover nested/repeated calls, high-word arithmetic,
guarded calls, aliased input/return slots, parameter snapshots, lexical name
collisions, forward prototypes, visible helpers, early returns, loops, pointer
side effects, FP32 and local objects. One case starts from ordinary CUDA `.cu`
and runs the worker's nvcc before both native implementations are built.

The [production audit](../../compiler_exp/gap_closure/device_functions/promotion-audit.json)
checks **6,881,280 retained bytes**, **60 complete native word/register
comparisons**, exact original and emitted sources, signatures/body/call
inventories, ownership and driver resources. Repeated local calls reserve 64
bytes in OA; controls retain their independently observed frames. Actual
device: A100-SXM4-40GB; worker Python 3.12.1, CUDA 12.8.1, ptxas V12.8.93. The
[earlier 512-launch report](../../compiler_exp/gap_closure/device_functions/candidate-a100-20261123T173558Z.json)
keeps its narrower fixture set and exact source snapshots.

The [wheel audit](../../compiler_exp/gap_closure/device_functions/wheel-audit.json)
replays all sixty native comparisons from an isolated built wheel. The
[integration record](../../compiler_exp/gap_closure/device_functions/validation.json)
binds full supported/compiler-only suites, docs and separate prior-native
continuity. Production and [regressions](../../tests/test_ptx_device_functions.py)
import no experiment modules. There is no performance claim.

```sh
python compiler_exp/gap_closure/device_functions/offline.py --candidate
modal run compiler_exp/gap_closure/device_functions/run_modal.py
python compiler_exp/gap_closure/device_functions/audit.py compiler_exp/gap_closure/device_functions/candidate-a100-20261123T181957Z.json /tmp/device-functions-audit.json
python -m pytest -q tests/test_ptx_device_functions.py
```

This narrows G16/G10 at a concrete CUDA/PTX boundary. Multiple translation units,
linked device libraries, register/aggregate ABI, native dynamic stack allocation
and full module symbol/relocation semantics retain their original exit gates.

## Explicit PTX and CUDA source libraries

The [separate translation-unit study](../../compiler_exp/gap_closure/device_linkage/README.md)
first compiled real CUDA device objects and linked them with NVIDIA's tools.
Original linked text retained an entry and separate helper function; standalone
OA rejected its unresolved external. An isolated exact-source merge passed a
48-launch A100 research gate. That evidence motivated the maintained resolver
below; the original baseline, failed visibility hypothesis and research gate
remain intact.

```sh
compile entry.cu --device-library helper.cu --out-dir build
compile entry.ptx --device-library math.ptx --device-library util.ptx --out-dir build
compile entry.cu --device-library helper.cu --compile-commands compile_commands.json --build-backend modal --out-dir build
```

`--device-library SOURCE` is repeatable on standalone `compile`/`cnr` routes.
Libraries may be PTX or CUDA/C++ source with no kernel entries. The root supplies
the executable entries; named library files are excluded from independent
kernel discovery. This is source-library resolution followed by the validated
static inliner. It does not consume `.o`/`.a`/cubin libraries, execute host link
commands or implement native object relocation or a runtime call stack.

The [resolver](../../openassembler/ptx/link.py) applies deterministic rules:

| Area | Admitted rule |
|---|---|
| Module headers | Exactly one PTX version, exact `sm_80` target, address size 64; use the largest supplied PTX version |
| Function ABI | Matching ordered scalar parameter/return types, at most 32 inputs and one `.param` return; pointer annotations/aggregate/register ABI reject |
| Private definitions | Module-local; uniquely renamed so equal helper names in distinct libraries remain independent |
| Exported definitions | One strong `.visible` definition; duplicate strong definitions reject |
| Weak definitions | A strong definition wins; otherwise byte-identical weak bodies are selected in explicit source order; nonidentical weak alternatives reject |
| Reachability | Expand only reachable definitions; unresolved reachable symbols and recursive call cycles reject |
| Source bounds | One root plus at most 64 distinct explicit libraries; existing inliner depth/statement/register/local-frame bounds still apply |
| Debug identifiers | Remap module-local `.file` and `.loc` indices together; unresolved or duplicate file indices reject |
| Storage | Per-call static local objects follow the existing frame contract; general module data/relocations and unsupported directives reject |

Root and library prototypes match by exact ordered type signatures. Private
symbols can resolve only within their module; exported library calls bind to
the selected definition. Unused unresolved declarations remain outside the
executable closure. A private-name collision cannot redirect a call to another
library. Source-library order and selected symbol/body identities are retained;
this policy makes no claim about every nvlink weak-symbol selection domain.

CUDA units use `nvcc --ptx` with relocatable device code. A conflicting
`-rdc=false` rejects. Each unit uses its selected compilation-database flags,
working directory and source spelling; explicit NVCC flags/includes also apply.
Library-only CUDA units may produce no entries. Actual tool-environment header
and source dependencies are hashed before and after compilation and must agree.
Changed headers alter the maintained contract. Toolchain identity, effective
flags, source hashes, original/rebased compile contracts and emitted PTX hashes
remain explicit. Compilation databases still do not execute original host build
or link commands.

The normal remote transport bundles external libraries, their selected command
records and includes/quoted dependencies under separate source roots. It
validates archive paths and each submitted library source digest; the worker
resolves with its rebased paths while retaining original source identity.
`device_link` in every kernel manifest records input-unit identities, selected
exports, reachable call edges, library compilation/dependency contracts and
the complete linked PTX digest. Native ownership snapshots that entire contract;
mutating it after compilation rejects. Same-linked-PTX reference cubins remain
separate from original separately linked controls.

### Research, failures and validation

All experiments are under the [promotion study](../../compiler_exp/gap_closure/device_linkage/promotion/README.md).
The first debug fixture exposed an inliner bug: `.loc` without a semicolon
could absorb the following arithmetic instruction. The maintained repair masks
the modeled three-number location prefix while preserving the instruction and
line count; a standalone regression covers it. The private-name fixture initially
called a later local helper without the toolkit-required prototype. Those
original toolkit rejections and corrected sources remain separate.

The first [A100 promotion attempt](../../compiler_exp/gap_closure/device_linkage/promotion/candidate-a100-20261123T201424Z.json)
stopped after 48 passing direct-case launches: its harness incorrectly applied
OA's local-frame record-pair invariant to an original linked control with callee
stack requirements. The [failed-gate audit](../../compiler_exp/gap_closure/device_linkage/promotion/first-failure-audit.json)
checks 344,064 partial bytes and leaves the report failed. The corrected harness
records actual original-link driver resources independently and preserves exact
OA frame checks.

The final [offline gate](../../compiler_exp/gap_closure/device_linkage/promotion/candidate-offline.json)
passes **720 complete executions**: ten source configurations × four levels ×
two seeds × three lengths × PTX/normal/poisoned SASS. Independent full buffers,
input immutability and outer guards cover transitive/private calls, strong/weak
resolution, debug identifiers, mixed versions, unused functions, wide slots and
static local objects. The [oracle audit](../../compiler_exp/gap_closure/device_linkage/promotion/oracle-audit.json)
binds all ten original separate PTX object/nvlink controls and their deterministic
merged controls, with complete source/binary/listing/tool-log custody.

The final [A100 gate](../../compiler_exp/gap_closure/device_linkage/promotion/candidate-a100-20261123T201545Z.json)
passes **576 launches**: twelve configurations × original separately linked
control, merged-PTX control and four OA levels × four active lengths × two seeds.
Three CTAs of 128 threads compare every byte with independent wrapping32/64
references. Two additional configurations start from CUDA source/headers and
per-unit compilation databases; one traverses the actual serialized remote
source-bundle/process worker. All helpers execute independently in the original
linked binaries, which retain separate executable sections. OA emits complete
owned entry text after source resolution and static expansion.

The [production audit](../../compiler_exp/gap_closure/device_linkage/promotion/promotion-audit.json)
checks **4,128,768 complete retained bytes**, **48 exact native word/register
comparisons**, symbol/ABI/body/edge inventories, source/header hashes, per-unit
flags, ownership and driver resources. The local case reserves 32 bytes in OA;
original linked controls retain their independently observed callee-stack
resources. Actual device: A100-SXM4-40GB; worker Python 3.12.1, CUDA 12.8.1.
Local observations use CUDA 12.8.61; the worker records its actual toolkit build.

The [normal Modal CLI audit](../../compiler_exp/gap_closure/device_linkage/promotion/normal-remote-audit.json)
additionally executes the public sender/build-worker/publication path with an
external CUDA library and compilation database. The
[wheel audit](../../compiler_exp/gap_closure/device_linkage/promotion/wheel-audit.json)
checks exact installed source/data and all 48 hardware-native comparisons in an
isolated wheel. The [integration record](../../compiler_exp/gap_closure/device_linkage/promotion/validation.json)
binds supported/compiler-only full suites, docs and prior-native continuity.
No performance claim is made.

This advances G16/G10/G42 through explicit scalar source-library integration.
Native object/archive linking, aggregate/register ABI, distributed library
selection, host application build/link execution, general module storage and
native dynamic stacks remain original open exit gates.
