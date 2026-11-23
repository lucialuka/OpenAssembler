"""Extract the supplied gap exit gates and attach current bounded evidence."""
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
UPDATES={
    'G01':('counterexample-resolved','Hardware isolates the retained mismatch to emulator FCHK slow-path selection; bounded model repair and deterministic regression. General FCHK truth remains incomplete.','docs/compiler/gaps-verification.md'),
    'G02':('partial','Every scalar admission declaration has representative cross-stage fixture records, operand variants and negative controls. General independent semantics/hardware conformance is still open.','docs/compiler/gaps-contracts.md'),
    'G04':('partial','Same-file typed scalar @device helpers support structured control flow with terminal returns and definite initialization. Cross-module helpers, general Python/library calls and device ABI remain open.','docs/compiler/gaps-python.md'),
    'G06':('partial','Typed scalar device helpers admit branches/range/while/break/continue with conservative definite initialization; arbitrary iterables, dynamic range steps, early helper returns and general Python control flow remain open.','docs/compiler/gaps-python.md'),
    'G11':('partial','mov.pred and guarded/loop-carried predicate assignments preserve values; packed/unpacked scalar64 snapshots and immediate operands have explicit regression gates. Broader scalar arithmetic and encoder coverage remain open.','docs/compiler/gaps-memory.md'),
    'G12':('partial','Global v2 32-bit loads, v4 32-bit stores and scalar byte/halfword/64-bit stores pass independent byte gates. Shared narrow/64-bit accesses, vector64, ordering and runtime alignment checking remain open.','docs/compiler/gaps-memory.md'),
    'G21':('partial','FP32 sigmoid and static rank0..4 sum/mean reductions and backward pass full-state A100/CLI gates. Convolution/pooling and general framework operator/effect coverage remain open.','docs/compiler/gaps-operators.md'),
    'G25':('partial','Rank2..4 contiguous batch-broadcast matmul and reduced gradients pass A100; arbitrary rank/layout/operation broadcasting remains open.','docs/compiler/gaps-model.md'),
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
    report={'schema':1,'date':'2026-11-21','input_sha256':hashlib.sha256(raw).hexdigest(),
            'scope':'51 supplied gaps; bounded changes do not close the entire roadmap',
            'all_gaps_closed':False,'gaps':rows}
    (HERE/'gap-register.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    latest=HERE/'integration_20261121/validation.json'
    if latest.is_file():
        result=json.loads(latest.read_text())
        supported=result['supported_suite']; core=result['compiler_only_suite']
        validation_summary=f"Final November 21 validation passes {supported['passed']:,} tests in Python 3.12.10/Torch 2.7.1 with {supported['skipped']} skips, and {core['passed']:,} tests with {core['skipped']} skips in the separate Python 3.14.5 compiler-only environment. A100 evidence and exact source/native-code identities are retained in the linked dossiers and [validation record](../../compiler_exp/gap_closure/integration_20261121/validation.json)."
    else:
        validation_summary='The [November 20 validation](../../compiler_exp/gap_closure/validation.json) retains 1,278 supported-environment passes. November 21 additions have focused CPU/A100 gates in their dossiers; stable-source full-suite integration is in progress.'
    lines=['# Compiler gap register','',
           'The supplied 51-gap review remains the scope of this program. This implementation closes the fixed division counterexample and repairs the named regression defects; it adds bounded helper control flow, memory widths, tensor operators, matmul and measurement capabilities. **The complete 51-gap program is not finished.**','',
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
              'The next unresolved foundation gates are general cross-stage semantic conformance, typed buffers and memory safety, wider memory/atomic/order support, pressure management and device ABI. Model dtype/effect/RNG/operator coverage, complete application/build/checkpoint execution, dynamic cache/serving/distributed contracts and held-out performance remain independent work. New targets and toolchain independence keep their separate milestones. Do not raise resource bounds, weaken tolerances, add vendor fallback or refresh measured policies to change an unsupported row into a success.','',
              'Validation environments and all run outcomes are recorded in the [experiment index](../../compiler_exp/gap_closure/README.md). Historical reports retain their original test counts and dates.']
    (ROOT/'docs/compiler/gap-register.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':run()
