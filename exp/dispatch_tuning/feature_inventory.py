"""Regenerate the shape/contract audit of every implemented catalogue feature.

Performance policy choices must be measured by workload contract. Instruction
semantics, proven dependency edges and validation gates are not performance
options to disable when a shape changes. This inventory keeps that distinction
explicit and also records useful unintegrated reduction candidates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def rule(kind, code, contract, dispatch, knobs, action):
    return {"classification": kind, "code_paths": code.split(), "applicability": contract,
            "current_selection": dispatch, "tuning_axes": knobs,
            "next_action": action}


RULES = {
    "arithmetic-instruction-selection": rule("instruction-semantics", "openassembler/backend/fragments.py openassembler/backend/ra.py",
        "Exact PTX opcode/modifiers, operand types/forms and supported SM80 encodings; independent of tensor M/N/K.",
        "Legal instruction fragments are selected from the actual PTX operation, not a shape lookup.",
        ["equivalent instruction forms", "integer/address sequence length", "register pressure"],
        "Search equivalent forms under semantic/encoding/hardware gates; do not make correct arithmetic contingent on a shape win."),
    "constant-division": rule("operand-domain-specialization", "openassembler/opt/divconst.py openassembler/backend/ra.py",
        "32-bit signed/unsigned constant divisor and exact quotient/remainder semantics; runtime dividends retain proved domain.",
        "Divisor- and signedness-specific lowering already chooses shifts/magic-multiply sequences.",
        ["divisor", "signedness", "known range", "equivalent instruction sequence"],
        "Preserve exhaustive/proof checks. Tune alternative legal forms by instruction use, not by unrelated GEMM dimensions."),
    "cooperative-layernorm": rule("shape-sensitive-kernel", "openassembler/kernels/reductions.py openassembler/rowwise.py",
        "Contiguous FP32 rows, width1..1024, positive normal finite FP32 epsilon; y-only population LayerNorm, no affine/saved statistics/backward.",
        "Explicit rowwise command; one warp per row, four warps per CTA fixed. No native graph substitution.",
        ["rows", "width and tail", "warps per CTA", "register pressure", "tree/load overlap"],
        "Run rowwise1/2/4/8-warp campaign on unchanged five-pattern gates; integrate only current exact shape winners."),
    "cooperative-softmax": rule("shape-sensitive-kernel", "openassembler/kernels/reductions.py openassembler/rowwise.py",
        "Contiguous bounded finite FP32 rows, width1..1024; full-warp max/sum reductions, no mask/backward.",
        "Explicit rowwise command with fixed four warps per CTA; tails are semantic masks, not optional tuning.",
        ["rows", "width and tail", "warps per CTA", "register pressure", "load/tree overlap"],
        "Run rowwise1/2/4/8-warp campaign and record shape-specific winners and regressions before promotion."),
    "cubin-metadata-splicing": rule("correctness-infrastructure", "openassembler/container/cubin.py openassembler/container/nvinfo.py openassembler/build.py",
        "Exact kernel entry/ABI, generated text, register declaration, exit offsets and container metadata.",
        "Always required when publishing native cubins.", [],
        "Keep binary/resource identity checks mandatory for every tuned candidate; do not shape-gate metadata repair."),
    "cuda-cpp-frontend": rule("frontend-contract", "openassembler/cppfront.py openassembler/frontend.py",
        "Supported CUDA/C++ device kernel syntax and nvcc PTX output; native OA handles supported PTX backend forms.",
        "Source language/repository configuration, not matrix geometry, chooses the frontend.",
        ["nvcc options affecting exact numerical contract", "generated PTX specialization"],
        "Retain separate native and explicit ptxas fallback labels. No shape-only promise about arbitrary C++ applications."),
    "cutlass-load-next-compute-current": rule("shape-sensitive-kernel", "openassembler/kernels/gemm.py",
        "128tile stagepair m_fast, legacy late pointers, unfused;1024x4096x1024 and4096x1024x1024 FP16/BF16 only.",
        "Explicit fragment_schedule=load_next and an admitted GEMM tuning candidate; no universal auto row.",
        ["M/N/K", "dtype", "fragment load placement", "register/live ranges", "tile"],
        "Current production GEMM campaign compares it against direct/early and geometry choices; promote exact wins, retain large-shape regressions."),
    "dead-code-and-copy-elimination": rule("semantic-local-transformation", "openassembler/backend/ra.py",
        "Actual use/def, aliases, predicates, implicit vector readers and side effects; independent of application shape when proof holds.",
        "Regular scalar/tensor pipeline before allocation.", ["equivalent propagation strategies", "live-range effects"],
        "Preserve mandatory alias/guard/implicit-reader correctness; benchmark broader rewrites by kernel identity rather than disabling valid DCE by shape."),
    "direct-pointer-homes": rule("shape-sensitive-kernel", "openassembler/backend/tensor.py openassembler/kernels/gemm.py",
        "Aligned loop-carried64bit pointer homes; explicit stagepair m_fast unfused contract; no altered address WAR.",
        "Manual pointer_mode=direct; GEMM tuner tests legal shapes. One current exact tile64/direct automatic row at audit start.",
        ["M/N/K", "tile", "dtype", "pointer live ranges", "advance placement"],
        "Revalidate all historical shapes through source-inclusive CLI and retain exact source/cubin/SASS before automatic routing."),
    "early-gemm-pointer-advance": rule("shape-sensitive-kernel", "openassembler/kernels/gemm.py openassembler/backend/tensor.py",
        "128tile stagepair m_fast unfused GEMM; early update preserves LDGSTS address-WAR and copy visibility dependencies.",
        "Explicit late/early and narrow pointer_advance=auto for1024x4096x1024; global GEMM tuner compares legal policies.",
        ["M/N/K", "dtype", "direct/legacy pointer mode", "register lifetime"],
        "Select exact repeated wins against all current controls; keep tall regressions and no shortening of dependency waits."),
    "fused-adam-like-update": rule("operator-semantic-contract", "examples/nn_ops.py",
        "FP32 example returns out/mn/vn; no bias correction, step policy or decoupled weight decay. Not AdamW.",
        "Explicit example kernel only; not substituted into training optimizers.",
        ["element count", "launch/block geometry", "vector width", "fusion boundary"],
        "May tune the exact formula by size; a complete optimizer needs full mutable-state and precision-contract validation first."),
    "fused-gemm-epilogues": rule("semantic-fusion-choice", "openassembler/kernels/gemm.py",
        "128tile stagepair GEMM, FP32 ((matmul+Bias)+Residual), optional finite ReLU; extra ABI buffers and grouping are part of semantics.",
        "Explicit bias_residual/bias_residual_relu plan; unfused GEMM tuner rejects this different contract.",
        ["M/N/K", "dtype", "epilogue role", "saved activations", "fusion versus separate kernels"],
        "Add a separate fused-contract tuner and whole-boundary comparison; never rank against bare GEMM as if equal work."),
    "gpu-execution-and-launch-specs": rule("correctness-infrastructure", "openassembler/runner.py openassembler/gpu/cuda_driver.py openassembler/gpu/batch.py",
        "Exact buffers, layouts, argument ABI, block/grid/shared bytes and selected cubin; faults isolated by worker.",
        "Checked source contract or explicit launch spec; no guessed scalar launch for unannotated tensor kernels.",
        ["legal launch geometry", "graph grouping", "transport overhead separately"],
        "Every tuned shape must retain its correct launch. Execution safety and validation stay mandatory regardless of speed."),
    "grouped-n-cta-swizzle": rule("shape-sensitive-kernel", "openassembler/kernels/gemm.py",
        "128tile stagepair legacy/late unfused; Ntile count divisible8; exact transformed grid(Mtiles*8,Ntiles/8).",
        "Explicit grouped_n8; available in current GEMM candidate set; held-out tall evidence exists, no global default.",
        ["M/N/K", "dtype", "CTA order", "L2 locality", "working set"],
        "Revalidate tall controls as well as forward roles; preserve mfast/Morton alternatives for shapes where groupedN loses."),
    "loop-unrolling": rule("kernel-structure-performance-policy", "openassembler/backend/unroll.py openassembler/build.py",
        "Supported counted-loop form, trip count/remainder/control and register pressure; tensor path bypasses scalar unroller.",
        "Scalar opt applies factor4; safe/sched do not apply the same unroll sequence.",
        ["trip count", "kernel/source identity", "factor", "live registers", "code size"],
        "Create scalar safe/sched/opt/factor search using exact workload/launch specs. A GEMM M/N/K lookup cannot tune arbitrary scalar loops."),
    "low-rank-factorization-library": rule("algorithm-and-precision-contract", "openassembler/fastmm/plan.py openassembler/fastmm/kernels.py openassembler/fastmm/factors.py",
        "Explicit FP32 low-rank/factor research plans; exact factors and temporary transforms; not the dense FP16/BF16 tensor contract.",
        "User-selected research path only.", ["M/N/K", "rank/factors", "transform placement", "workspace", "output precision"],
        "Bound live products and match precision before comparing. Do not silently replace ordered HMMA GEMM with different numerical semantics."),
    "morton-cta-order": rule("shape-sensitive-kernel", "openassembler/kernels/gemm.py",
        "Complete8x8 groups of128-outputtiles; changes CTA traversal, not MMA fragment storage layout.",
        "Explicit morton8 and legal GEMM tuner candidate.", ["M/N/K", "dtype", "CTA grid", "L2 locality"],
        "Current campaign includes prior tall winners and broad role negatives; promote repeated exact wins only."),
    "native-gelu": rule("size-sensitive-operator", "examples/nn_ops.py openassembler/backend/fragments.py",
        "FP32 tanh-approximate forward GELU with declared FP64 tolerance; not exact-erf GELU and not bitwise library equality.",
        "Standalone explicit example; no native graph replacement.", ["element count", "block size", "vectorization", "fusion role"],
        "Tune launch/source variants per size under the same approximation/gate; integrate graph path separately."),
    "native-relu": rule("size-sensitive-operator", "examples/nn_ops.py",
        "Finite FP32 elementwise ReLU; operation semantics and tails must remain fixed.",
        "Standalone explicit example; no automatic library substitution.", ["element count", "block size", "vectorization", "fusion boundary"],
        "Tune small launch-bound and large bandwidth-bound cases separately; retain PyTorch/same-PTX controls and all tails."),
    "native-tensor-gemm": rule("shape-sensitive-kernel", "openassembler/kernels/gemm.py openassembler/kernels/_gemm64.py openassembler/gemm.py",
        "Unbatched aligned contiguousRCR FP16/BF16 inputs, FP32acc/output, alpha1 beta0, full64/128M/Ntiles,K%128; contract excludes general layouts/tails.",
        "Native CLI exact measured dispatch plus optionalGPUtuning; unknown/stale rows baseline.",
        ["M/N/K", "dtype", "tile", "CTA order", "fragment schedule", "pointer mode/advance", "resources"],
        "Current source-inclusive selection/held-out campaign retests all admitted alternatives and preserves exact negative controls."),
    "operand-reuse-hints": rule("operand-liveness-performance-policy", "openassembler/backend/tensor.py openassembler/isa/word.py",
        "Supported neighboring HMMA operand identities; unchanged live fragments and implicit readers; not a programmable register cache.",
        "TensorConfig.reuse=True in production; no arbitrary mask CLI switch.",
        ["fragment order", "operand liveness", "reuse bits", "register groups", "M/N/K through generated schedule"],
        "Future joint fragment/reuse search needs exact words and hardware/counters; never overwrite live inputs to gain a hint."),
    "paired-fragment-loads": rule("layout-constrained-kernel-structure", "openassembler/kernels/gemm.py openassembler/kernels/_gemm64.py openassembler/backend/tensor.py",
        "Validated lane map and aligned LDSM.x4 B half-pairs; both halves remain explicit register readers.",
        "Selected generator uses paired loads; tile family determines counts.",
        ["tile shape", "fragment ownership", "load order", "register live range"],
        "Compare complete source families jointly with registers/layouts; the legal native-half pair rule remains a correctness invariant."),
    "ptx-native-frontend": rule("frontend-contract", "openassembler/ptx/parse.py openassembler/frontend.py",
        "Implemented PTX opcode/type/control subset and exact supportedSM80 architecture; unsupported forms refuse or explicitfallback.",
        "Inputlanguage/IRcontent chooses native scalar or tensor backend.", [],
        "Keep semantics and explicit unsupported boundaries; performance dispatch occurs after contract recognition."),
    "python-kernel-frontend": rule("frontend-contract", "openassembler/pyfront/compiler.py openassembler/pyfront/types.py",
        "Typed numeric @kernel subset, fixedGPUinteger/floatsemantics and safe AST parsing; no arbitrary hostPython compilation.",
        "Source annotations/types and supported constructs, not shape labels.", ["compile-time specialization", "declared launch/workload"],
        "Specialize declared constants safely; native GPU execution is SASS, so host-language rewriting is not a tensor throughput policy."),
    "pytorch-application-capture": rule("graph-and-state-contract", "openassembler/torchfront/worker.py openassembler/torchfront/frontend.py",
        "Configured arguments/observed shapes/branches and exact state/runtime; Inductor/Triton own GPU code in this route.",
        "Explicit configured application frontend; cache replay forbids uncapturedspecializations.",
        ["model/input shapes", "training/inference", "backend graph choices", "packing/launches", "fullstate"],
        "Native kernel tuning does not automatically affect ordinaryPyTorch. Add controllednativegraphhook and first-divergence/fullstate gate before application speed claims."),
    "repository-builds": rule("build-infrastructure", "openassembler/repository.py openassembler/project.py openassembler/build.py",
        "Safe source discovery, naming/dependency context, atomic output,manifest publication andboundedparallelCPUbuilds.",
        "Repository/input context; compileworker count is host-resource choice.", ["source count", "host CPU/toolkit contention", "build jobs"],
        "Measure compile/startup separately; hostbuildconcurrency does not select fastest GPU kernel by shape."),
    "sass-emulator-and-tensor-numerics": rule("validation-infrastructure", "openassembler/emu/machine.py openassembler/emu/tensor.py",
        "Supported functionalSASS/warp/tensor arithmetic subset; no hardwaretimingmodel or allspecialvalueguarantee.",
        "Explicit validation and regression coverage, not runtime throughputdispatch.", [],
        "Retain poisoned-register/complete-output checks; hardwaretiming and canary/graphgates remain separate."),
    "sass-encoding": rule("instruction-semantics", "openassembler/isa/encode.py openassembler/backend/tensor_encoding.py specs/sm_80",
        "SupportedSM80 instruction forms, exactfields and generatedencodingdata.",
        "Opcode/operandform selects encoding, with explicit refusal for unknownforms.", ["equivalentlegalSASSforms under semantic proof"],
        "Do not shape-gate correct encoding. New forms require source-inclusive bits/disassembly/numerical evidence."),
    "scalar-dependency-correctness": rule("mandatory-correctness", "openassembler/backend/ra.py openassembler/backend/compile.py openassembler/backend/sched.py",
        "Aliasbinding, predicates, implicitreaders, scoreboardgeneration retirement, sourceWAR and actualWARPSYNC.",
        "Always preserved by safe/sched/opt/tensor policies.", [],
        "Neverdisable provenedges byshape or timing. Search schedules only within the legal dependency graph."),
    "scalar-register-allocation": rule("resource-and-liveness-policy", "openassembler/backend/ra.py",
        "Actual virtual-use/defliveness, guardeddefinitions, loopcarriedhomes and alignedgroups.",
        "Allocation operates on actualIR/liveness; not tensor M/N/Ktable.",
        ["kernelIR", "schedule/allocation ordering", "pressure", "groupplacement", "spills"],
        "Explore pressureaware schedule/allocation candidates with exactkernel/workloadidentity; keepcorrectgroupconstraints mandatory."),
    "scalar-scheduling": rule("dependency-and-kernel-policy", "openassembler/backend/sched.py openassembler/backend/compile.py openassembler/build.py",
        "Measured pairclasses plus scoreboards and mandatorydependencies; scalarminimumstalltable is not HMMApipeline latency.",
        "Scalar safe/sched/opt policies; tensor uses separate protected sourceorderHMMA8/drain15.",
        ["kernelIR", "safe/sched/opt", "issueoverlap", "livepressure", "runtimeworking set"],
        "Tune legal wholekernels under launch/workloadkey. Never universalS1 or globaldrainshortening from shape alone."),
    "shared-layout-swizzling": rule("layout-constrained-kernel-policy", "openassembler/kernels/gemm.py openassembler/kernels/_gemm64.py",
        "Selectedsymmetric7/7XORsharedmap with provedcopy/LDSMlane/addresscontracts.",
        "Fixedselectedmap within each64/128kernel family; asymmetricsweeps remainresearch.",
        ["tile", "sharedswizzle", "fragment/registerlayout", "loadorder", "banktraffic"],
        "Jointlysearchshared/registerlayout using independentaddresscertificates and fullcurrentGPUchecks; no unmeasuredgeneralpolicy."),
    "small-grid-tensor-cta-tiles": rule("shape-sensitive-kernel", "openassembler/kernels/_gemm64.py openassembler/kernels/gemm.py openassembler/gemm_dispatch.py",
        "64x64x64CTA, four32x32warptiles,32KiBshared; M/N%64,K%128,unfusedstagepairmfast; legacy/late or direct/late.",
        "Exactshapeauto row or manualtile; current tuner compares64against128families.",
        ["M/N/K", "dtype", "tile", "gridparallelism", "pointerpolicy", "resources"],
        "Revalidateallpriorwins/currentbinaries, preserve14large/otherregressions from historical32case study and donotgeneralizesmallwins."),
    "stage-pair-specialization": rule("shape-sensitive-kernel", "openassembler/kernels/gemm.py",
        "Fulltiletwo-stageKloop; alternatestaticsharedstages versus combinedreference, preservingcpasynccommit/wait and barriers.",
        "stagepairbaseline,combinedavailablemanualandGEMMtuner.", ["Ktripcount", "M/N", "tile", "register/codegrowth"],
        "Keepcombinedasacorrectcandidateevenwherepreviouslylost; exactshapeGPUtuner decides,not globalhistory."),
    "state-verification-diagnostics": rule("mandatory-validation", "openassembler/torchfront/worker.py",
        "Fulloutputs,parameters,buffers,gradients,optimizerstate/settingsunderunchangedprecision/tolerances.",
        "Verificationgates diagnoseBF16divergence;notanoptimizerrepairoroptionalperformancepolicy.", [],
        "Keepallstatechecks atintegrationboundaries; neverdispatchfasterinvalidcandidate orchangeepsilon/tolerance for speed."),
    "tensor-address-generation": rule("operand-and-layout-specialization", "openassembler/kernels/gemm.py openassembler/kernels/_gemm64.py openassembler/backend/tensor.py",
        "ProvedU32elementindexbounds,widened64bitglobaladdresses,copydisplacements,alignedLDSMaddresses and addressWAR.",
        "Generatorcontract+nativeinstructionselection;pointerhomepolicyseparate.",
        ["M/N/K", "tile", "addressCSE", "copyinitializerreuse", "directhomes", "LEA/IMADforms"],
        "Currentaddressreusecandidateisnotpromoted; requirecarry/addressidentity,fullbits/canaries andexactshapepairedbenefit."),
    "vectorized-fp32-epilogue": rule("alignment-constrained-kernel-structure", "openassembler/kernels/gemm.py openassembler/kernels/_gemm64.py openassembler/backend/tensor.py",
        "FP32adjacentoutputpairs,8byteCalignment,fulltileN andlegalconsecutiveregisterpair; mathematicalgroupingunchanged.",
        "Selectedgeneratorstorepolicy;legalgroupreusevalidatedbyactualoperandidentities.",
        ["tile", "Nstride/alignment", "epiloguefusion", "registerpairplacement"],
        "Futurestorewidthchoices needalignment/tailcontracts andcurrentmatchedGPUgates; do notoverridelegalpairidentitychecks."),
    "shape-aware-native-gemm-dispatch": rule("shape-and-environment-performance-policy", "openassembler/gemm_dispatch.py openassembler/gemm_tune.py openassembler/gemm_tune_modal.py",
        "ExactM/N/K/dtype/layout/precision/alignment/epilogue/workspace/tailspluscompileridentity/PTXbody; A100only.",
        "Measuredexacttable or explicitcurrentGPUtuning;unsupported/stalerowsfallback; manualknobsretainreproducibility.",
        ["alladmittedGEMMpolicies", "actualA100variant", "toolkit/driver", "pairedtimings", "source/binaryidentity"],
        "Completecurrentselectionandindependentheldoutreplays,retainallartifacts,thenpromoteonlyrepeatedwins without extrapolation."),
}


def inventory(root: Path = ROOT) -> dict:
    catalogue = json.loads((root / "docs/research/catalogue.json").read_text())
    implemented = [idea for idea in catalogue["ideas"] if idea["status"] == "implemented"]
    ids = {idea["id"] for idea in implemented}
    if ids != set(RULES):
        raise ValueError(f"audit rules need review: missing={sorted(ids-set(RULES))}; stale={sorted(set(RULES)-ids)}")
    records = []
    for idea in implemented:
        dossier = root / idea["path"]
        content = dossier.read_text()
        local_links = []
        for raw in re.findall(r"\]\(([^)]+)\)", content):
            if "://" in raw or raw.startswith("#"):
                continue
            candidate = (dossier.parent / raw.split("#", 1)[0]).resolve()
            if candidate.exists() and candidate.is_relative_to(root):
                local_links.append(candidate.relative_to(root).as_posix())
        checked = {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                   for path in RULES[idea["id"]]["code_paths"] if (root / path).is_file()}
        records.append({"id": idea["id"], "title": idea["title"], "status": idea["status"],
                        "dossier": idea["path"], "dossier_sha256": hashlib.sha256(content.encode()).hexdigest(),
                        **RULES[idea["id"]], "code_sha256": checked,
                        "retained_references": sorted(set(local_links)),
                        "catalogue_summary": idea["summary"]})
    return {"schema": 1, "audit_date": "2026-11-10", "scope": "all implemented research catalogue entries at generation",
            "implemented_count": len(records),
            "principle": "Specialize measured performance choices by complete workload contract; keep semantic legality, dependencies and validation unconditional.",
            "not_an_all_shapes_performance_proof": True,
            "pending_nonimplemented_priority": [{"id": "fused-backward-reductions",
                "status": "validated-not-integrated", "code": "exp/nn_round2/operator_candidates.py",
                "reason": "FMAchangesroundingandfullbackwardoperatorcontractisoutsideproductionrowwise; narrowoldwinsnotgenericnormdispatch.",
                "next_action": "Retestrows/width/tree/loadgeometryandexplicitcontractionpolicy,thenaddfull-stateverifiedproductionoperatorifadmitted."}],
            "features": records}


def main():
    result = inventory()
    path = HERE / "feature-inventory.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{path}: {result['implemented_count']} implemented features audited")


if __name__ == "__main__":
    main()
