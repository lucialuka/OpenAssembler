"""Extract the supplied gap exit gates and attach current bounded evidence."""
from datetime import date
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
UPDATES={
    'G01':('counterexample-resolved','Hardware isolates the retained mismatch to emulator FCHK slow-path selection; bounded model repair and deterministic regression. General FCHK truth remains incomplete.','docs/compiler/gaps-verification.md'),
    'G02':('partial','Every scalar admission declaration has representative cross-stage fixture records, operand variants and negative controls. Actual Boolean-loop A100 failures now isolate a native predicate guard readiness gap across CFG edges; 2,400 repaired launches and 16,957,440 independent bytes pass under unchanged references. General independent semantics/hardware conformance is still open.','docs/compiler/gaps-contracts.md'),
    'G03':('partial','Logical Boolean scalar helper arguments/returns now retain exact predicate copies, complete return paths and original source/native signature ownership, with 2,400 A100 launches, 16,220,160 independent complete bytes and 240 exact current native outputs. Logical Boolean kernel parameters now use canonical u8 launch slots with exact source/native type custody and public Boolean run-spec bindings: 9,600 A100 original-source launches, 960 public-run launches and 69,746,688 independently checked bytes. Natural narrow b/u/s8/16 entry slots pass a separate 14,400-launch gate. Boolean Array/Buffer/shared byte storage now passes 5,280 original-source A100 launches, 528 public buffer launches and 22,302,720 independent bytes, including noncanonical loads/canonical stores, rank1..4 strided views and shared permutations. Direct Boolean comparisons and narrow register operations remain open research. Typed i64/u64 low64 multiplication and augmented assignment now pass source/helper/loop/cast/square gates with PTX/CUDA controls: 1,560 A100 launches and 156 native outputs. Typed i64/u64 scalar, global/shared array, helper, conversion/comparison/shift routes pass exact gates. Naturally aligned shared arrays charge padding to 48KiB; a failed hardware phase led to bounded native reconvergence and a final 1,280-launch A100 gate. FP64 arithmetic, narrower types and complete Python semantics remain open.','docs/compiler/gaps-python-types.md'),
    'G04':('partial','Pure bool-annotated helper parameters/returns now pass eager once-only argument/effect, predicate loop/early-return, caller isolation and original signature/source/native custody gates: 2,400 A100 launches and 155 new regressions. Original pure scalar early/branch/loop returns now preserve every-path typing, live-path initialization, caller isolation and same-file/imported source ownership with 2,400 A100 launches/240 native outputs. Static same-file/local-package typed scalar helpers support module aliases, relative imports and static re-exports with defining-module annotations, original diagnostics, dependency hashes and native ownership through normal local/remote builds. A 600-launch A100 gate passes; arbitrary runtime packages, captures/host effects and general Python/library/device ABI remain open.','docs/compiler/gaps-python.md'),
    'G05':('partial','Finite numeric tuple iteration now preserves empty/target/break/continue/nested/else semantics with 2,400 A100 launches and 240 native outputs. Original explicit Final numeric and immutable tuple values now specialize with bounded arithmetic/index/slice/len and exact original source/diagnostic/value/use/ownership. Same-file/imported/re-export/package forms pass 1,440 A100 launches and 144 native outputs; arbitrary objects/runtime containers/metaprogramming/mutable host effects remain open.','docs/compiler/gaps-static-values.md'),
    'G06':('partial','Finite static numeric tuple loops preserve empty/observable targets, runtime break/continue, nested tuple/range loops and exhaustion else with original source/type/convergence/ownership. 2,400 A100 launches and 240 native outputs pass. Pure helper early/branch/loop returns now add typed return homes, live-path initialization and caller/source isolation with a separate 2,400-launch A100 gate. Finite i32 range counters now exhaust before wrap; runtime range/while else preserves nearest-loop exits, live-path initialization, convergence and original source ownership in a separate 2,400-launch A100 gate. Typed original condition chains now preserve single evaluation/short-circuit effects, exact pairwise literal contexts and expression/source/native ownership with a separate 2,400-launch A100 gate. Boolean locals/truth/conversions and operand-returning lazy and/or now preserve exact literal contexts, effects and source/native ownership, with an actual hardware predicate-path repair and 2,400 final A100 launches. Arbitrary iterables, dynamic steps, wide bounds, Boolean ABI/type unions and general Python control/effects remain open.','docs/compiler/gaps-python.md'),
    'G07':('partial','Logical Boolean Buffer elements now use exact one-byte spans/strides and canonical stores, rank1..4 read/broadcast/nonoverlap/checked bounds and exact host Boolean buffer roles, with independent original-source/A100 gates. Rank1..4 checked Buffer views admit explicit nonnegative strides, read-only broadcasts and provably nonoverlapping writes, with validated launch bindings. General views, negative strides, alias effects and device layout algebra remain open.','docs/compiler/gaps-buffers.md'),
    'G08':('partial','Python global i32/u32 atomic_add and u32 exchange/CAS preserve old values, nested evaluation and checked Buffer bounds; A100 contention and convergence gates pass. General collectives, async and tensor primitives remain open.','docs/compiler/gaps-python-primitives.md'),
    'G09':('partial','Boolean shared arrays now use one-byte elements, canonical stores/nonzero loads and exact mixed-width padding under the unchanged static limit, with independent cross-thread A100 permutations/barriers and poisoned-state gates. Checked Buffer accesses implement zero/skip bounds behavior with storage extent, alignment, dtype and write-alias binding validation. Raw pointers remain caller-bounded; shared initialization, arbitrary races and general memory-safety proof remain open.','docs/compiler/gaps-buffers.md'),
    'G10':('partial','Logical Boolean array/shaped/shared storage now preserves byte layout and source/type/native custody with actual public Boolean buffer bindings; 5,280 original-source and 528 public-run A100 launches pass. Logical Boolean kernel parameters now retain exact u8 physical signatures and predicate type metadata through normal builds/native ownership and Boolean-only JSON bindings; 9,600 original-source and 960 actual public-run A100 launches pass. Scalar b/u/s8/16 entry parameters now use exact natural byte slots, signed/zero extension and within-slot subword reads across both selectors, interpreter, verifier and public explicit-run/batch-driver ABI. Original PTX A100 gates pass 14,400 launches plus 240 explicit launches, 67,461,120 complete bytes and 240 exact native outputs. Aggregate/vector/FP16/predicate/device-slot ABI and logical Python Boolean launch/storage remain open. Global/generic pointer metadata preserves ABI slots/alignment and ownership. Module-local scalar device functions now retain signatures/body hashes/call counts through validated static expansion and native manifests. Aggregate/register/local/shared-pointer ABI and general linkage remain open.','docs/compiler/gaps-parameter-abi.md'),
    'G11':('partial','Scalar byte/halfword entry loads now preserve extension/adjacent slots/guarded loop and alias snapshots, with 14,400 original PTX plus 240 explicit-run A100 launches. Actual original CUDA Boolean controls now reach the separately recorded setp.s16/selp.u16 scalar boundary; those narrow register operations remain unsupported. Low64 mul.lo.u64/s64 and typed Python products pass 1,560 original-source A100 launches and 15,974,400 audited bytes, including aliased/guarded/immediate/parameter/helper/loop/cast/square snapshots. Scalar selp.b64/u64/s64/f64 now passes 1,480 original-source A100 launches, 15,155,200 audited bytes and 148 native comparisons; aliases/guards/negation/parameters/immediates preserve bit snapshots. Wide arithmetic/snapshots and literal contexts pass exact gates. Typed integer/floating scalar operand expressions add precedence/casts and binary64 intermediate rounding, with 4,000 independent typed trees, 432 CPU executions and a final 384-launch A100 gate. Failed cross-toolchain controls remain retained with explicit unresolved signedness/NaN/type boundaries; general expressions, arithmetic/modifiers/types/operands and encoder conformance remain open.','docs/compiler/gaps-wide.md'),
    'G12':('partial','Boolean array buffers now require true/false initialization with exact one-byte extents, canonical randomized values and checked logical roles; actual public A100 specs pass. Global v2 loads/v4 stores and scalar narrow/64-bit stores pass byte gates. Scalar shared byte/halfword/64-bit loads/stores add four recovered templates, 12,096 varied native words and a final 1,216-launch A100 gate after retaining a failed barrier schedule. The 336-row width matrix aligns admission/lowering; dynamic shared alignment/initialization/races, vector64 and ordered memory remain open.','docs/compiler/gaps-memory.md'),
    'G13':('partial','Static per-thread local frames, explicit 32-bit local addresses and scalar32 loads/stores pass 704 A100 launches, exact per-function metadata/resource and ownership gates. G17 adds bounded scalar/group/predicate spill frames. Generic pointers, wider/vector local accesses, device calls and module global/constant relocation remain open.','docs/compiler/gaps-storage.md'),
    'G14':('partial','GPU global 32-bit integer add/exchange/CAS admit relaxed/acquire/release/acq_rel; add reductions remain relaxed. CTA shared 32-bit add/exchange and independent A100 publication/contention gates pass. Full memory-model, scope, width and operation coverage remains open.','docs/compiler/gaps-atomics.md'),
    'G15':('partial','Full-mask unpredicated indexed/down/up scalar shuffles admit exact unsegmented clamps, low-five-bit lane operands and reconverged full warps; 1,792 A100 launches pass including aliased operands. Forward scalar CTA phases now receive a hardware-validated BSSY/BSYNC pair where every path provably reaches one barrier. Partial participation, votes/match/redux, cyclic/irreducible convergence and broader barriers remain open.','docs/compiler/gaps-collectives.md'),
    'G16':('partial','Direct scalar calls and explicit PTX/CUDA source libraries resolve through static specialization with private/export/weak/signature rules and per-unit dependency/CLI/remote/ownership contracts. A 576-launch A100 gate includes original separately linked controls, wide/local/transitive cases and ordinary CUDA/remote builds. Native object/archive linking, dynamic stacks, recursion/indirect calls and register/aggregate ABI remain open.','docs/compiler/gaps-device-functions.md'),
    'G17':('partial','Bounded scalar/pair/quad/predicate spills honor explicit and automatic limits. Virtual dependency/guard/order-preserving scheduling admits fewer slots and non-increased block-local transfers/frame. A100 correctness and paired kernel/resource controls pass, including maximum-frame recovery; host code-generation overhead is measured separately. Slot reuse, tensor-fragment spilling and general occupancy policy remain open.','docs/compiler/gaps-spilling.md'),
    'G21':('partial','FP32 Conv2d with contiguous NCHW/OIHW, groups=1 and static stride/padding/dilation now supports input/weight/bias gradients. Independent A100 checks pass 80 full tensors and public CNN compile/replay each pass 45 training-state checks, with 645 model regressions and 142 exact native outputs across eleven bundles. FP32 sigmoid and static rank0..4 sum/mean reductions and backward also pass full-state A100/CLI gates. Grouped/depthwise convolution, pooling, BatchNorm and general framework operator/effect coverage remain open.','docs/compiler/gaps-convolution.md'),
    'G25':('partial','Rank2..4 contiguous batch-broadcast matmul and reduced gradients pass A100; arbitrary rank/layout/operation broadcasting remains open.','docs/compiler/gaps-model.md'),
    'G34':('partial','SGD momentum/dampening/Nesterov/decay/maximize and Adam/AdamW AMSGrad/maximize pass complete-state CPU/A100 and resumed trajectories. Parameter groups, clipping/accumulation/schedulers and broader optimizer families remain open.','docs/compiler/gaps-optimizers.md'),
    'G35':('partial','Owned native programs have versioned complete parameter/buffer/optimizer/CPU-RNG/application-JSON checkpoints with atomic save, staged restore, rollback and exact A100 interrupted/resumed state. Unsupported host, GPU RNG, distributed and AMP state remain separate open contracts.','docs/compiler/gaps-checkpoints.md'),
    'G39':('partial','Explicit opt-in exact-length specialization validates a native factory, maximum context, build-attempt budget, cache/input/math/parameter guards and failure-preserving adoption. CPU and complete A100 token/cache/ownership gates pass; empty prefixes, padding/position and beam/speculative semantics remain open.','docs/compiler/gaps-decode.md'),
    'G42':('partial','Static local Python helper graphs now preserve exact entry/helper/package hashes, original diagnostics, explicit roots and dependency/native ownership through local/remote builds, with 600 A100 launches and public Modal CLI validation. Compilation databases preserve per-source NVCC arguments/cwd/dependency identities and remote rebasing. Explicit scalar source libraries now retain per-unit compilation and deterministic function resolution through CLI/transport, validated against original linked controls. Binary libraries, general build/link orchestration and complete original host application rebuild/execution remain open.','docs/compiler/gaps-build-graph.md'),
    'G47':('partial','Source, cache and worker phases now attributed; cold/hit samples retain identical binaries. Remote startup/transport and separate tensor-pass attribution remain open.','docs/compiler/gaps-measurement.md'),
    'G48':('partial','Optional repeated profiler audits retain per-contract samples/distributions, loaded resources and binary sizes, with profiler-free whole-step arrays separately labeled. Profiler-free per-kernel performance, fusion and complete host/transfer attribution remain open.','docs/compiler/gaps-profiling.md'),
    'G51':('fixed-for-recorded-regressions','Stale bias-broadcast rejection replaced with genuine rejection plus capability control; stable emulator seeds, fixed division case and dtype diagnostics repaired. Tested dependency environments are named; no exhaustive Python/platform matrix claim.','docs/compiler/gaps-model.md'),
}


def run():
    raw=(HERE/'input-gap-register.txt').read_bytes()
    blocks=re.findall(r'^### (G\d+) — (.+?)\n(.*?)(?=^### G\d+ —|^## Practical consequences)',raw.decode(),re.M|re.S)
    rows=[]
    for identity,title,body in blocks:
        priority=re.search(r'\*\*Priority:\*\* (.+)',body).group(1).strip()
        exit_gate=re.search(r'\*\*What closes this gap:\*\* (.+?)(?=\n\n\*\*Evidence:)',body,re.S).group(1).strip()
        status,progress,evidence=UPDATES.get(identity,('open','Original exit gate remains required.',None))
        rows.append({'id':identity,'title':title,'priority':priority,'status':status,
                     'progress':progress,'evidence':evidence,'exit_gate':exit_gate})
    assert len(rows)==51 and len({row['id'] for row in rows})==51
    report={'schema':1,'date':date.today().isoformat(),'input_sha256':hashlib.sha256(raw).hexdigest(),
            'scope':'51 supplied gaps; bounded changes do not close the entire roadmap',
            'all_gaps_closed':False,'gaps':rows}
    (HERE/'gap-register.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    latest=HERE/'integration_20261123/validation.json'
    if (HERE/'decode_specialization/validation.json').is_file():
        latest=HERE/'decode_specialization/validation.json'
    if (HERE/'warp_collectives/validation.json').is_file():
        latest=HERE/'warp_collectives/validation.json'
    if (HERE/'local_storage/validation.json').is_file():
        latest=HERE/'local_storage/validation.json'
    if (HERE/'spilling/validation.json').is_file():
        latest=HERE/'spilling/validation.json'
    if (HERE/'spilling_groups/validation.json').is_file():
        latest=HERE/'spilling_groups/validation.json'
    if (HERE/'pressure_schedule/validation.json').is_file():
        latest=HERE/'pressure_schedule/validation.json'
    if (HERE/'float_literals/validation.json').is_file():
        latest=HERE/'float_literals/validation.json'
    if (HERE/'constant_expressions/validation.json').is_file():
        latest=HERE/'constant_expressions/validation.json'
    if (HERE/'shared_widths/validation.json').is_file():
        latest=HERE/'shared_widths/validation.json'
    if (HERE/'python_shared_wide/validation.json').is_file():
        latest=HERE/'python_shared_wide/validation.json'
    if (HERE/'device_functions/validation.json').is_file():
        latest=HERE/'device_functions/validation.json'
    if (HERE/'device_linkage/promotion/validation.json').is_file():
        latest=HERE/'device_linkage/promotion/validation.json'
    if (HERE/'wide_select/validation.json').is_file():
        latest=HERE/'wide_select/validation.json'
    if (HERE/'wide_multiply/promotion/validation.json').is_file():
        latest=HERE/'wide_multiply/promotion/validation.json'
    if (HERE/'python_module_helpers/promotion/validation.json').is_file():
        latest=HERE/'python_module_helpers/promotion/validation.json'
    if (HERE/'python_static_values/promotion/validation.json').is_file():
        latest=HERE/'python_static_values/promotion/validation.json'
    if (HERE/'python_static_loops/promotion/validation.json').is_file():
        latest=HERE/'python_static_loops/promotion/validation.json'
    if (HERE/'python_helper_returns/promotion/validation.json').is_file():
        latest=HERE/'python_helper_returns/promotion/validation.json'
    if (HERE/'python_range_control/promotion/validation.json').is_file():
        latest=HERE/'python_range_control/promotion/validation.json'
    if (HERE/'python_comparison_chains/promotion/validation.json').is_file():
        latest=HERE/'python_comparison_chains/promotion/validation.json'
    if (HERE/'python_boolean_values/promotion/validation.json').is_file():
        latest=HERE/'python_boolean_values/promotion/validation.json'
    if (HERE/'python_boolean_abi/helpers/validation.json').is_file():
        helper_result=json.loads((HERE/'python_boolean_abi/helpers/validation.json').read_text())
        if helper_result.get('passed'):
            latest=HERE/'python_boolean_abi/helpers/validation.json'
    if (HERE/'narrow_parameters/validation.json').is_file():
        narrow_result=json.loads((HERE/'narrow_parameters/validation.json').read_text())
        if narrow_result.get('passed'):
            latest=HERE/'narrow_parameters/validation.json'
    if (HERE/'python_boolean_abi/scalar/validation.json').is_file():
        scalar_result=json.loads((HERE/'python_boolean_abi/scalar/validation.json').read_text())
        if scalar_result.get('passed'):
            latest=HERE/'python_boolean_abi/scalar/validation.json'
    if (HERE/'python_boolean_abi/storage/validation.json').is_file():
        storage_result=json.loads((HERE/'python_boolean_abi/storage/validation.json').read_text())
        if storage_result.get('passed'):
            latest=HERE/'python_boolean_abi/storage/validation.json'
    if latest.is_file():
        result=json.loads(latest.read_text())
        supported=result['supported_suite']; core=result['compiler_only_suite']
        record=latest.relative_to(ROOT).as_posix()
        validation_date=date.fromisoformat(result['date']).strftime('%B %d').replace(' 0',' ')
        validation_summary=f"{validation_date} integration passes {supported['passed']:,} tests in Python 3.12.10/Torch 2.7.1 with {supported['skipped']} skips, and {core['passed']:,} tests with {core['skipped']} skips in the separate Python 3.14.5 compiler-only environment. A100 evidence and exact source/native-code identities are retained in the linked dossiers and [validation record](../../{record})."
    else:
        validation_summary='The [November 23 baseline-source run](../../compiler_exp/gap_closure/foundations_20261121/supported-suite-20261123-first.log) passes 2,574 supported-environment tests. New integration changes have separate candidate/hardware gates in their dossiers; final stable-source integration is in progress. Earlier dated validation remains historical.'
    lines=['# Compiler gap register','',
           'The supplied 51-gap review remains the scope of this program. Validated additions cover typed helpers/wide integers/buffers/atomics, native memory and parameter contracts, tensor operators, optimizer/checkpoint state, recorded builds and measurement. **The complete 51-gap program is not finished.**','',
           validation_summary,'',
           'The [machine-readable register](../../compiler_exp/gap_closure/gap-register.json) retains every original exit gate. The [supplied review](../../compiler_exp/gap_closure/input-gap-register.txt) is retained unchanged, including its dated evidence and limits. A row marked partial requires the remaining original gate; it is not counted as a completed broad capability.','',
           '| Gap | Finding | Current state |','|---|---|---|']
    for row in rows:
        status=row['status']
        if row['evidence']:status=f'[{status}]({Path(row["evidence"]).name})'
        lines.append(f'| {row["id"]} | {row["title"]} | {status} |')
    lines += ['', '## Research and engineering cycle','',
              'Each bounded dossier defines the gap and numerical/effect/resource contract, compares a candidate before promotion, retains independent validation and failures, and identifies its production path. Experiments live under `compiler_exp/gap_closure/`; production imports none of them. The [policy continuity audit](gaps-continuity.md) separately checks installed policies against frozen code and retained hardware-session artifacts.','',
              'The [repository inventory](../../compiler_exp/gap_closure/repository-map.json) reads and hashes all Git-tracked files and extracts every Python module’s imports and top-level definitions. Deep source review covered Python/PTX/CUDA entry routes; selection, allocation, scheduling and encoding; emulator/reference arithmetic; model capture, autodiff, kernels, storage/runtime/replay/state; and repository/build/transport/CLI/packaging paths. This is not a claim of manual semantic review of every historical experiment or proof of arbitrary workloads.','',
              'Remaining foundation gates include general cross-stage semantic conformance, buffer/layout and shared-memory safety, wider memory/atomic/order support, pressure management and device ABI. Model dtype/effect/RNG/operator coverage, complete original host/build/application execution, dynamic cache/serving/distributed contracts and held-out performance remain independent work. The complete checkpoint covers only the admitted owned native program. New targets and toolchain independence keep their separate milestones. Do not raise resource bounds, weaken tolerances, add vendor fallback or refresh measured policies to change an unsupported row into a success.','',
              'Validation environments and all run outcomes are recorded in the [experiment index](../../compiler_exp/gap_closure/README.md). Historical reports retain their original test counts and dates.']
    (ROOT/'docs/compiler/gap-register.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':run()
