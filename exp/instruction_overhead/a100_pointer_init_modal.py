"""A100 source-inclusive tensor address-overhead study, with retained binaries.

The experiment invokes the production native encoder directly with an explicit
TensorConfig, builds a same-source ptxas control, and splices through the normal
container writer. It does not bypass native encoding or shorten HMMA controls.
No recorded candidate becomes production policy merely by passing this runner.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import modal

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
IMAGE = (modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .add_local_dir(str(ROOT / "openassembler"), "/root/repo/openassembler", ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir(str(ROOT / "specs"), "/root/repo/specs")
    .add_local_dir(str(ROOT / "corpus"), "/root/repo/corpus")
    .add_local_dir(str(ROOT / "env"), "/root/repo/env")
    .add_local_dir(str(ROOT / "tools"), "/root/repo/tools", ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir(str(ROOT / "exp/shape_coverage"), "/root/repo/exp/shape_coverage", ignore=["**/__pycache__", "**/*.pyc", "artifacts/**"])
    .add_local_dir(str(HERE), "/root/repo/exp/instruction_overhead", ignore=["**/__pycache__", "**/*.pyc", "artifacts/**", "*.json"]))
APP = modal.App("openassembler-tensor-pointer-init-overhead")


def identity() -> dict:
    paths = []
    for directory in ("backend", "ptx", "container", "isa", "emu", "opt"):
        paths += list((ROOT / "openassembler" / directory).glob("*.py"))
    paths += [ROOT / "openassembler" / name for name in ("build.py", "oracle.py", "project.py")]
    paths += [ROOT / "openassembler" / name for name in ("gemm.py",)]
    paths += [ROOT / "openassembler/kernels/gemm.py"]
    paths += list((ROOT / "specs/sm_80").glob("*.json"))
    paths += [ROOT / "corpus/sm_80/pairs.jsonl", ROOT / "corpus/sm_80/machine_model.json",
              ROOT / "exp/shape_coverage/small_tile_gemm.py", HERE / "candidates.py", Path(__file__)]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


@APP.function(image=IMAGE, gpu="A100-40GB", timeout=3600, max_containers=1, retries=0)
def execute(cases: list[dict], rounds: int = 3, samples: int = 12) -> dict:
    import base64
    import ctypes as C
    import glob
    import os
    import random
    import shutil
    import statistics
    import sys
    import tempfile

    sys.path.insert(0,"/root/repo")
    sys.path.insert(0,"/root/repo/tools")
    import torch
    import openassembler.oracle as oracle
    oracle._available.clear()
    oracle.available = lambda tool="ptxas": shutil.which(tool) is not None
    oracle.run_tool = lambda *args, check=True: subprocess.run(list(args), cwd="/root/repo", capture_output=True, text=True, check=check)
    from openassembler.build import ptxas_build, splice_kernels
    from openassembler.backend.tensor import generate, TensorConfig
    from openassembler.gemm import source_for as annotated_gemm_source
    from openassembler.gpu.cuda_driver import Driver
    from openassembler.isa.word import control
    from openassembler.kernels.gemm import GemmPlan
    from openassembler.ptx.parse import parse_module
    from exp.instruction_overhead.candidates import CANDIDATES, source, address_certificate
    from exp.shape_coverage.small_tile_gemm import SmallTilePlan

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.cuda.init()
    _ = torch.empty(1,device="cuda")
    driver = Driver()
    if driver.info().cc != "8.0":
        raise RuntimeError("A100/SM80 required")
    rng=random.Random(20261109)
    paths=glob.glob("/usr/local/lib/python3*/site-packages/nvidia/cublas/lib/libcublas.so*")
    blas=C.CDLL(paths[0] if paths else "libcublas.so.12")
    handle=C.c_void_p()
    create,destroy=blas.cublasCreate_v2,blas.cublasDestroy_v2
    create.argtypes,create.restype=[C.POINTER(C.c_void_p)],C.c_int
    destroy.argtypes,destroy.restype=[C.c_void_p],C.c_int
    setmath,setstream=blas.cublasSetMathMode,blas.cublasSetStream_v2
    setmath.argtypes,setmath.restype=[C.c_void_p,C.c_int],C.c_int
    setstream.argtypes,setstream.restype=[C.c_void_p,C.c_void_p],C.c_int
    gemmex=blas.cublasGemmEx
    gemmex.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_void_p,C.c_int,C.c_int,C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int]
    gemmex.restype=C.c_int
    if create(C.byref(handle)) or setmath(handle,16):raise RuntimeError("cuBLAS policy initialization failed")
    alpha,beta=C.c_float(1),C.c_float(0)
    result_rows=[]
    retained={}
    try:
        with tempfile.TemporaryDirectory(prefix="oa-pointer-init-") as directory:
            root=Path(directory)
            for case in cases:
                plan=SmallTilePlan(case["m"],case["n"],case["k"],case["dtype"])
                key=f"{plan.m}x{plan.n}x{plan.k}_{plan.dtype}"
                native_artifacts,statics,resources,functions,modules,source_map,certs={},{},{},{},{},{},{}
                source_refs={}
                launch_by_method={}
                matched_ptxas={}
                for candidate in CANDIDATES:
                    text,cert=source(plan,candidate)
                    source_hash=hashlib.sha256(text.encode()).hexdigest()
                    if source_hash not in source_refs:
                        ref=ptxas_build(text,"sm_80")
                        source_refs[source_hash]=(f"ptxas_{candidate.label}",ref)
                        source_map[f"{key}/{candidate.label}.ptx"]=text
                    launch_by_method[source_refs[source_hash][0]]=plan.launch
                    launch_by_method[candidate.label]=plan.launch
                    matched_ptxas[candidate.label]=source_refs[source_hash][0]
                    config=TensorConfig(pointer_mode=candidate.pointer_mode,pointer_advance="early" if candidate.early_advance else "late")
                    code,labels,words,exits,regs=generate(parse_module(text).kernels["gemm"],"sm_80","opt",config=config)
                    ref_name,reference=source_refs[source_hash]
                    blob=splice_kernels(reference,{"gemm":(words,regs,exits)})
                    native_artifacts[candidate.label]=blob
                    certs[candidate.label]={**cert,"config":config.__dict__,"matched_ptxas":ref_name,
                        "source_order_instructions":len(code),"encoded_registers":regs,
                        "native_word_sha256":hashlib.sha256(b"".join(w.to_bytes(16,"little") for w in words)).hexdigest()}
                # Revalidate the integrated production tile64 source alongside
                # the experiment-only address candidates.  The experiment
                # emitter is retained above for comparison; these two labels
                # use the checked marker/contract path from openassembler.gemm.
                for label, pointer_mode in (("prod_tile64", "legacy"),
                                            ("prod_tile64_direct", "direct")):
                    plan64 = GemmPlan(plan.m, plan.n, plan.k, dtype=plan.dtype,
                                      tile="64x64", pointer_mode=pointer_mode)
                    text = annotated_gemm_source(plan64)
                    source_hash = hashlib.sha256(text.encode()).hexdigest()
                    if source_hash not in source_refs:
                        ref = ptxas_build(text, "sm_80")
                        source_refs[source_hash] = (f"ptxas_{label}", ref)
                        source_map[f"{key}/{label}.ptx"] = text
                    launch_by_method[source_refs[source_hash][0]] = plan64.launch
                    config = TensorConfig(pointer_mode=pointer_mode, pointer_advance="late")
                    code, labels, words, exits, regs = generate(
                        parse_module(text).kernels["gemm"], "sm_80", "opt", config=config)
                    ref_name, reference = source_refs[source_hash]
                    blob = splice_kernels(reference, {"gemm": (words, regs, exits)})
                    native_artifacts[label] = blob
                    launch_by_method[label] = plan64.launch
                    matched_ptxas[label] = ref_name
                    certs[label] = {"candidate": {"label": label, "tile_mnk": [64, 64, 64],
                                      "pointer_mode": pointer_mode, "pointer_advance": "late"},
                        "manual_launch": plan64.launch, "config": config.__dict__,
                        "matched_ptxas": ref_name, "source_sha256": source_hash,
                        "production_contract": True, "source_order_instructions": len(code),
                        "encoded_registers": regs,
                        "native_word_sha256": hashlib.sha256(
                            b"".join(w.to_bytes(16, "little") for w in words)).hexdigest()}
                # Same input/output contract with the current production-sized
                # 128x128 CTA. These are compiled and timed in the same process
                # as the 64x64 policies so tile routing can be selected honestly.
                for label,pointer_mode,pointer_advance in (
                    ("tile128","legacy","late"),
                    ("tile128_direct","direct","late"),
                    ("tile128_early","legacy","early"),
                    ("tile128_direct_early","direct","early"),
                ):
                    plan128=GemmPlan(plan.m,plan.n,plan.k,dtype=plan.dtype,
                                     pointer_mode=pointer_mode,pointer_advance=pointer_advance)
                    text=annotated_gemm_source(plan128)
                    source_hash=hashlib.sha256(text.encode()).hexdigest()
                    if source_hash not in source_refs:
                        ref=ptxas_build(text,"sm_80")
                        source_refs[source_hash]=(f"ptxas_{label}",ref)
                        source_map[f"{key}/{label}.ptx"]=text
                    launch_by_method[source_refs[source_hash][0]]=plan128.launch
                    config=TensorConfig(pointer_mode=pointer_mode,pointer_advance=pointer_advance)
                    code,labels,words,exits,regs=generate(parse_module(text).kernels["gemm"],"sm_80","opt",config=config)
                    ref_name,reference=source_refs[source_hash]
                    blob=splice_kernels(reference,{"gemm":(words,regs,exits)})
                    native_artifacts[label]=blob
                    launch_by_method[label]=plan128.launch
                    matched_ptxas[label]=ref_name
                    certs[label]={"candidate":{"label":label,"tile_mnk":[128,128,64],
                            "pointer_mode":pointer_mode,"pointer_advance":pointer_advance},
                        "manual_launch":plan128.launch,"config":config.__dict__,"matched_ptxas":ref_name,
                        "source_sha256":source_hash,"tensor_copy_visibility_barrier_stream_equal":True,
                        "source_order_instructions":len(code),"encoded_registers":regs,
                        "native_word_sha256":hashlib.sha256(b"".join(w.to_bytes(16,"little") for w in words)).hexdigest()}
                artifacts={name:blob for name,blob in source_refs.values()} | native_artifacts
                for name,blob in artifacts.items():
                    module=driver.load(blob)
                    modules[name]=module
                    fn=driver.function(module,"gemm")
                    driver.ck(driver.cu.cuFuncSetAttribute(fn,8,launch_by_method[name]["smem"]),"shared opt-in")
                    functions[name]=fn
                    resources[name]=driver.func_attrs(fn)
                    path=root/f"{key}_{name}.cubin"
                    path.write_bytes(blob)
                    sass=subprocess.run(["nvdisasm","-hex","-c",str(path)],capture_output=True,text=True,check=True).stdout
                    listing=oracle.parse_listing(sass)
                    hmma=[control(x.word).stall for x in listing if x.text.startswith("HMMA")]
                    drains=[control(x.word) for i,x in enumerate(listing) if i and x.text.startswith("NOP") and listing[i-1].text.startswith("HMMA")]
                    if name in native_artifacts and (set(hmma)!={8} or not drains or any(c.stall!=15 or not c.yield_ for c in drains)):
                        raise AssertionError({"unapproved_controls":name,"hmma":hmma,"drains":str(drains)})
                    statics[name]={"cubin_sha256":hashlib.sha256(blob).hexdigest(),"sass_sha256":hashlib.sha256(sass.encode()).hexdigest(),
                        "listing_instructions":len(listing),"hmma":len(hmma),"hmma_stalls":sorted(set(hmma)),
                        "ldsm":sum(x.text.startswith("LDSM") for x in listing),"mov":sum(x.text.startswith("MOV") for x in listing),
                        "drain_count":len(drains)}
                    retained[f"{key}/{name}.cubin"]={"type":"base64","data":base64.b64encode(blob).decode()}
                    retained[f"{key}/{name}.sass"]={"type":"text","data":sass}
                for name,text in source_map.items():retained[name]={"type":"text","data":text}
                m,n,k=plan.m,plan.n,plan.k
                dtype=torch.float16 if plan.dtype=="f16" else torch.bfloat16
                methods=list(artifacts)+["cublas"]
                def launch(name,out):
                    stream=C.c_void_p(torch.cuda.current_stream().cuda_stream)
                    if name=="cublas":
                        if setstream(handle,stream):raise RuntimeError("cuBLAS stream failed")
                        operand=2 if plan.dtype=="f16" else 14
                        status=gemmex(handle,1,0,n,m,k,C.byref(alpha),C.c_void_p(b.data_ptr()),operand,k,
                            C.c_void_p(a.data_ptr()),operand,k,C.byref(beta),C.c_void_p(out.data_ptr()),0,n,68,99)
                        if status:raise RuntimeError(f"cuBLAS GemmEx {status}")
                        return
                    args=[C.c_uint64(t.data_ptr()) for t in (a,b,out)]
                    params=(C.c_void_p*3)(*[C.cast(C.pointer(x),C.c_void_p) for x in args])
                    config=launch_by_method[name]
                    driver.ck(driver.cu.cuLaunchKernel(functions[name],*config["grid"],128,1,1,config["smem"],stream,params,None),name)
                checks=[]
                last_refs={}
                for family in ("zero","integer","normal"):
                    torch.manual_seed(9811+m+n+k)
                    if family=="zero":
                        a=torch.zeros((m,k),device="cuda",dtype=dtype)
                        b=torch.randn((n,k),device="cuda",dtype=dtype)
                    elif family=="integer":
                        a=torch.randint(-2,3,(m,k),device="cuda").to(dtype)
                        b=torch.randint(-2,3,(n,k),device="cuda").to(dtype)
                    else:
                        a=torch.randn((m,k),device="cuda",dtype=dtype)
                        b=torch.randn((n,k),device="cuda",dtype=dtype)
                    saved_a,saved_b=a.clone(),b.clone()
                    truth=a.float() @ b.float().T
                    refs={}
                    for name in artifacts:
                        if name.startswith("ptxas_"):
                            out=torch.empty((m,n),device="cuda")
                            launch(name,out)
                            torch.cuda.synchronize()
                            refs[name]=out.clone()
                    for name in methods:
                        store=torch.full((m*n+512,),-9876.5,device="cuda")
                        out=store[256:-256].view(m,n)
                        out.fill_(float("nan"))
                        launch(name,out)
                        torch.cuda.synchronize()
                        relative=float((out-truth).norm()/truth.norm().clamp_min(1e-30))
                        canaries=bool((store[:256]==-9876.5).all() and (store[-256:]==-9876.5).all())
                        input_equal=bool(torch.equal(a,saved_a) and torch.equal(b,saved_b))
                        refname=matched_ptxas[name] if name in matched_ptxas else name
                        bitwise=None if name=="cublas" else bool(torch.equal(out.view(torch.int32),refs[refname].view(torch.int32)))
                        baseline_bitwise=None if name=="cublas" else bool(torch.equal(out.view(torch.int32),refs["ptxas_legacy"].view(torch.int32)))
                        passed=canaries and input_equal and bitwise is not False and baseline_bitwise is not False and bool(torch.isfinite(out).all()) and relative<5e-5
                        check={"method":name,"input":family,"relative_l2":relative,"canaries":canaries,
                            "inputs_unchanged":input_equal,"same_source_ptxas_bitwise":bitwise,"baseline_ptxas_bitwise":baseline_bitwise,"passed":passed}
                        checks.append(check)
                        if not passed:raise AssertionError({"case":case,"check":check})
                    last_refs=refs
                graphs,outputs,graph_checks={},{},[]
                for name in methods:
                    outputs[name]=torch.empty((m,n),device="cuda")
                    for _ in range(3):launch(name,outputs[name])
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(16):launch(name,outputs[name])
                    outputs[name].fill_(float("nan"))
                    graph.replay();torch.cuda.synchronize()
                    refname=matched_ptxas[name] if name in matched_ptxas else name
                    bitwise=None if name=="cublas" else bool(torch.equal(outputs[name].view(torch.int32),last_refs[refname].view(torch.int32)))
                    relative=float((outputs[name]-truth).norm()/truth.norm().clamp_min(1e-30))
                    passed=bool(torch.isfinite(outputs[name]).all()) and bitwise is not False and relative<5e-5
                    graph_checks.append({"method":name,"poisoned_output":True,"same_source_ptxas_bitwise":bitwise,"relative_l2":relative,"passed":passed})
                    if not passed:raise AssertionError({"case":case,"graph":name})
                    graphs[name]=graph
                timings=[]
                for round_id in range(rounds):
                    elapsed={name:[] for name in methods};orders=[]
                    for _ in range(samples):
                        order=list(methods);rng.shuffle(order);orders.append(order)
                        for name in order:
                            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            start.record();graphs[name].replay();end.record();end.synchronize()
                            elapsed[name].append(start.elapsed_time(end)/16)
                    timings.append({"round":round_id,"paired_orders":orders,"samples_ms":elapsed,
                                    "median_ms":{n:statistics.median(v) for n,v in elapsed.items()}})
                row={"case":case,"launch":plan.launch,"certificates":certs,"resources":resources,"static":statics,
                     "checks":checks,"poisoned_graph_checks":graph_checks,"timings":timings}
                result_rows.append(row)
                print(json.dumps({"case":case,"median_ms":[x["median_ms"] for x in timings]}),flush=True)
                del graphs,outputs,a,b,truth,refs,last_refs
                torch.cuda.synchronize()
                for module in modules.values():driver.unload(module)
    finally:
        destroy(handle)
        driver.close()
    return {"schema":1,"device":torch.cuda.get_device_name(),"runtime":{"torch":str(torch.__version__),"cuda":torch.version.cuda},
            "address_certificate":address_certificate(),"cases":result_rows,"retained_artifacts":retained,
            "methodology":{"native_encoder":"production tensor.generate + build.splice_kernels; explicit TensorConfig",
                "hmma_issue_gap":8,"epoch_drain":15,"rounds":rounds,"samples":samples,"graph_launches":16,
                "precision":"FP16/BF16 operands; FP32 accumulation/output; alpha1 beta0; reduced precision and TF32 disabled",
                "cublas":"GemmEx, disallow reduced precision reduction; not tuned cuBLASLt",
                "scope":"isolated GEMM, excludes packing, graph integration and training-state behavior"}}


@APP.local_entrypoint()
def main(pilot: bool = False, rounds: int = 3, samples: int = 12):
    import base64
    shapes=[(128,512,512),(128,1536,512),(1024,1024,1024),(1024,4096,1024),(1024,512,512),(4096,2048,8192)]
    cases=[dict(m=m,n=n,k=k,dtype=dtype) for m,n,k in (shapes[:1] if pilot else shapes) for dtype in (["f16"] if pilot else ["f16","bf16"])]
    before=identity()
    result=execute.remote(cases,rounds,samples)
    after=identity()
    result["source_identity"]={"sha256":before,
        "harness_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope":"native encoder/container/data, checked GEMM plan source, exact tile64 experiment generator and this runner"}
    result["source_unchanged_during_run"]=before==after
    result["timestamp_utc"]=datetime.now(timezone.utc).isoformat()
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    artifact_root=HERE/"artifacts"/stamp
    raw=result.pop("retained_artifacts")
    retained={}
    for relative,entry in raw.items():
        path=artifact_root/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        data=base64.b64decode(entry["data"]) if entry["type"]=="base64" else entry["data"].encode()
        path.write_bytes(data)
        retained[str(path.relative_to(ROOT))]={"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)}
    result["retained_artifacts"]=retained
    out=HERE/f"a100-pointer-init_{stamp}.json"
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(out)
