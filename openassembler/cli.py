"""OpenAssembler 0.2.0: a Python / CUDA C / C++ (and PTX) to NVIDIA SASS compiler with its own backend.

    openassembler SRC [--backend modal|user@host] [--verify]  # compile a source or repo
    openassembler inspect [SRC] [--json]                # static coverage
    openassembler compile [SRC] [-o NAME] [--verify]     # -> output/cubin + optional SASS
    openassembler compile my_repo/                    # -> my_repo/output/{cubin,sass}
    openassembler run my_repo/ [--backend local|modal|user@host] [--dry-run]
    openassembler cnr [SRC] [compile + run options]     # compile, then run what compiled
    openassembler gemm --m M --n N --k K --dtype f16  # specialized A100 tensor GEMM
    openassembler status                              # headline results (docs/STATUS.md)
    openassembler corpus | check | bench | fastmm | gpu run   # research tooling (tools/*.py)
    openassembler                                     # no arguments: interactive shell

Frontends turn each source into PTX (.py: openassembler.pyfront; .cu/.cpp: nvcc --ptx);
our backend generates the SASS and splices it into a ptxas-built ELF container (README.md
"How it works"). Run `openassembler <command> --help` for every flag.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from openassembler import __version__

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"

OPT_LEVELS = ("safe", "mined", "sched", "opt")


def _run_tool(script: str, argv: list[str]) -> int:
    """Exec tools/<script> with the same interpreter, PYTHONPATH=ROOT."""
    import os

    env = dict(os.environ, PYTHONPATH=str(ROOT))
    return subprocess.run([sys.executable, str(TOOLS / script), *argv], cwd=ROOT, env=env).returncode


def _compile_opts(a: argparse.Namespace):
    from openassembler.build import Options

    return Options(arch=a.arch, opt_level=a.opt_level, kernel=a.kernel, verify=a.verify, trials=a.trials,
                   fallback=a.fallback, no_sass=a.no_sass, include_dirs=[Path(d) for d in a.include],
                   nvcc_flags=list(a.nvcc_flag), dump_ptx=a.dump_ptx)


def _frontend_choice(src: Path, out_dir: Path | None, requested: str):
    from openassembler.repository import select_frontend
    return select_frontend(src, out_dir or (Path(src).resolve() / "output"), requested)


def _model_compile_opts(a: argparse.Namespace) -> dict:
    from openassembler.modelfront.frontend import ModelError
    if a.arch != 'sm_80' or a.fallback:
        raise ModelError('native models require sm_80 and OA code generation without vendor fallback')
    if a.kernel or a.include or a.nvcc_flag or a.dump_ptx or a.out or a.opt_level != 'opt' or a.trials != 6:
        raise ModelError('kernel selection, naming, frontend flags and nondefault '
                         '--opt-level/--trials apply to standalone native kernels')
    if a.job_timeout is not None:
        raise ModelError('native model workers use --budget as their hard stage watchdog; omit --job-timeout')
    return {'verify': a.verify, 'dry_run': a.dry_run, 'no_sass': a.no_sass,
            'backend': a.build_backend or a.backend, 'host': a.build_host or a.host,
            'gpu': a.build_gpu or a.gpu,
            'budget': a.build_budget if a.build_budget is not None else a.budget}


def _native_build(a: argparse.Namespace):
    from openassembler.build import compile_paths
    from openassembler.remote_build import compile_native
    from openassembler.transport import backend
    from openassembler.workflow import output_dir
    out = output_dir(Path(a.src), a.out_dir)
    kind, host = backend(a.build_backend or a.backend, a.build_host or a.host)
    kind = 'local' if kind == 'auto' else kind
    if kind == 'local' and (a.build_budget is not None or (a.cmd == 'compile' and a.budget is not None)):
        from openassembler.project import ProjectError
        raise ProjectError('local native compilation has no build watchdog; omit --budget/--build-budget '
                           'or select a remote build backend. For cnr, --budget still bounds GPU execution')
    if a.fallback:
        from openassembler.project import ProjectError
        raise ProjectError("vendor GPU fallback is disabled; unsupported kernels must be implemented in OA")
    options = _compile_opts(a)
    if kind != 'local' or a.dry_run:
        rc, only = compile_native(Path(a.src), out, options, a.out, backend=kind, host=host,
                                  budget=a.build_budget if a.build_budget is not None else (
                                      a.budget if a.budget is not None else 900), dry_run=a.dry_run)
        if rc and not a.dry_run and (out / 'build-errors.json').is_file():
            import json
            a._build_errors = json.loads((out / 'build-errors.json').read_text())
        return rc, out, only
    result = compile_paths(Path(a.src), out, options, a.out)
    a._failed_sources = [str(p) for p in result.failed_sources]
    a._build_errors = result.errors
    return result.rc, result.out_dir, {(str(so.src.resolve()), k.name) for so in result.built for k in so.kept}


def cmd_inspect(a: argparse.Namespace) -> int:
    """Inspect compilation coverage without importing user code or contacting a GPU."""
    import json
    from openassembler.diagnostics import inspect_source
    report = inspect_source(Path(a.src))
    if a.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"{report['source']}: {report['frontend']} frontend")
        for candidate in report['kernel_candidates']:
            print(f"  {candidate['language']}: {candidate['path']}")
        print(f"{len(report['kernel_candidates'])} kernel source(s), {len(report['other_files'])} other file(s)")
        print(report['contract'])
        print('next: ' + report['next_step'])
    return 0 if report['ready_for_capture_or_compilation'] else 1


def cmd_compile(a: argparse.Namespace) -> int:
    """Source file or directory -> output/ (cubin/, sass/, manifest.json, .build/),
    using our backend (openassembler.backend.ra) instead of ptxas's code generator.
    Accepts .ptx (as-is), .py (openassembler.pyfront's numeric subset), or
    .cu/.cuh/.cpp/.cc/.cxx (via nvcc --ptx, openassembler.cppfront) -- all end up as
    PTX handed to the same backend. ptxas builds the ELF container our code is
    spliced into (README.md "How it works"); see openassembler/build.py."""
    from openassembler import project
    from openassembler.transport import TransportError
    from openassembler.workflow import output_dir
    frontend, root, plan = _frontend_choice(Path(a.src), Path(a.out_dir).resolve() if a.out_dir else None,
                                             a.frontend)
    if frontend == "model":
        from openassembler.modelfront.frontend import compile_model_repository
        return compile_model_repository(root, output_dir(root, a.out_dir), **_model_compile_opts(a))
    if plan is not None and not plan.has_native:
        from openassembler.repository import native_error
        a._failure = str(native_error(plan))
        print(f"error: {a._failure}", file=sys.stderr)
        return 1
    try:
        return _native_build(a)[0]
    except (project.ProjectError, TransportError, OSError, ValueError) as exc:
        a._failure = str(exc)
        print(f'error: {exc}', file=sys.stderr)
        return 1


def cmd_gemm(a: argparse.Namespace) -> int:
    """Generate and compile a validated full-tile SM80 tensor GEMM."""
    from openassembler.gemm import compile_gemm
    from openassembler.kernels.gemm import GemmPlan
    from openassembler.project import ProjectError
    try:
        knobs = {name: getattr(a, name) for name in ("schedule", "fragment_schedule", "cta_order",
                 "pointer_mode", "pointer_advance", "tile", "address_init")
                 if getattr(a, name) is not None}
        manual_requested = bool(knobs) or a.epilogue != "none"
        # Explicit implementation knobs remain reproducible manual requests.
        # Automatic policy selection is the normal path when no knob is set.
        tile = "128x128" if a.m % 128 == a.n % 128 == 0 else "64x64"
        plan = GemmPlan(m=a.m, n=a.n, k=a.k, dtype=a.dtype, arch=a.arch,
                        epilogue=a.epilogue, **{"tile": tile, **knobs})
        if (a.autotune or a.tuning_record) and (manual_requested or a.dispatch == "manual"):
            raise ValueError("GPU tuning/record selection cannot be combined with manual policy knobs")
        if a.autotune and a.emit_only:
            raise ValueError("--autotune requires automatic dispatch, no manual policy knobs, and a real build")
        if a.autotune or a.tuning_record:
            from openassembler.gemm_tune import tune_or_load
            record = a.tuning_record or str(Path(a.out_dir) / "gemm-tuning.json")
            plan, decision = tune_or_load(plan, backend=a.autotune, record=record,
                                          rounds=a.tune_rounds, samples=a.tune_samples)
        elif a.dispatch == "auto" and not manual_requested:
            from openassembler.gemm_dispatch import select_plan
            plan, decision = select_plan(plan)
        else:
            decision = {"schema": 1, "mode": "manual", "policy": "explicit",
                        "reason": "explicit implementation options"}
        return compile_gemm(plan, Path(a.out_dir), a.out, no_sass=a.no_sass,
                            emit_only=a.emit_only, dispatch=decision)
    except (ValueError, TypeError, OSError, ProjectError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def cmd_rowwise(a: argparse.Namespace) -> int:
    """Generate and compile a checked cooperative FP32 row reduction."""
    from openassembler.rowwise import compile_rowwise
    from openassembler.kernels.reductions import ReductionPlan
    from openassembler.project import ProjectError
    try:
        plan = ReductionPlan(rows=a.rows, width=a.width, op=a.op, epsilon=a.epsilon, arch=a.arch,
                             warps_per_cta=(a.warps_per_cta if a.warps_per_cta is not None else 4))
        manual = a.warps_per_cta is not None or a.dispatch == "manual"
        if not manual and a.dispatch == "auto":
            from openassembler.rowwise_dispatch import select_plan
            plan, decision = select_plan(plan)
        else:
            decision = {"schema": 1, "mode": "manual", "policy": f"w{plan.warps_per_cta}",
                        "reason": "explicit rowwise warp geometry"}
        return compile_rowwise(plan, Path(a.out_dir), a.out, no_sass=a.no_sass,
                               emit_only=a.emit_only, dispatch=decision)
    except (ValueError, TypeError, OSError, ProjectError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run_opts(a: argparse.Namespace, out_dir: str, only=None):
    from openassembler.runner import RunOptions
    from openassembler.transport import backend as resolve_backend

    kind, host = resolve_backend(a.backend, a.host)
    backend = host if kind == 'ssh' else ('local' if kind == 'auto' else kind)
    return RunOptions(out_dir=Path(out_dir), backend=backend, kernel=getattr(a, "run_kernel", None),
                      source=getattr(a, "source", None), spec=Path(a.spec) if a.spec else None, n=a.n,
                      reps=a.reps, gpu=a.gpu or "A100-40GB", budget=a.budget if a.budget is not None else 300,
                      job_timeout=a.job_timeout if a.job_timeout is not None else 60, dry_run=a.dry_run,
                      save_jobs=Path(a.save_jobs) if a.save_jobs else None,
                      from_results=Path(a.from_results) if getattr(a, "from_results", None) else None,
                      only=only)


def cmd_run(a: argparse.Namespace) -> int:
    """Execute saved OA cubins or a guarded native model launch plan."""
    from openassembler.runner import run_command
    from openassembler import project
    from openassembler.transport import TransportError
    from openassembler.workflow import resolve_run

    from openassembler.modelfront.frontend import resolve_model_output, run_model_repository
    model_output = resolve_model_output(Path(a.outdir) if a.outdir else Path.cwd(), a.out_dir)
    if model_output is not None:
        if a.kernel or a.source or a.spec or a.from_results or a.save_jobs or a.application or a.app_arg or a.n != 1 << 20 or a.reps != 20 or a.job_timeout is not None:
            raise ValueError('standalone kernel/application selectors and --job-timeout do not apply to native model replay')
        return run_model_repository(model_output, backend=a.backend, host=a.host, gpu=a.gpu,
                                    budget=a.budget, verify=a.verify_application, dry_run=a.dry_run)
    try:
        target = resolve_run(a.outdir, a.out_dir, a.application)
    except (project.ProjectError, OSError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1
    a.application = target.application
    kernel_selection = a.kernel or a.source or a.spec or a.from_results or a.save_jobs
    if a.application and kernel_selection:
        print('error: --application cannot be combined with kernel selection, --spec, '
              '--from-results or --save-jobs', file=sys.stderr)
        return 1
    if a.application and (a.n != 1 << 20 or a.reps != 20):
        print('error: -n and --reps apply to standalone kernels; application arguments are saved in its bundle', file=sys.stderr)
        return 1
    if not a.application and (a.app_arg or a.verify_application):
        print('error: --app-arg is disabled; --verify-application requires a saved OA [model] bundle', file=sys.stderr)
        return 1
    if a.application:
        print('error: legacy Inductor/Triton application replay is disabled; recompile with an OA [model] factory', file=sys.stderr)
        return 1
    try:
        a.run_kernel = a.kernel
        return run_command(_run_opts(a, str(target.out_dir), target.only))
    except (TransportError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def cmd_cnr(a: argparse.Namespace) -> int:
    """compile SRC into --out-dir, then run every kernel that compiled (`run`). Kernels
    that failed to compile are reported and not run; nonzero exit if anything failed."""
    from openassembler.runner import RunError, run_command, validate_options
    from openassembler import project
    from openassembler.transport import TransportError
    from openassembler.workflow import output_dir

    frontend, root, plan = _frontend_choice(Path(a.src), Path(a.out_dir).resolve() if a.out_dir else None,
                                             a.frontend)
    if frontend == "model":
        from openassembler.modelfront.frontend import compile_model_repository, run_model_repository
        if a.spec or a.save_jobs or a.n != 1 << 20 or a.reps != 20:
            raise ValueError('standalone launch specs and repetition flags do not apply to native models')
        out = output_dir(root, a.out_dir)
        rc = compile_model_repository(root, out, **_model_compile_opts(a))
        if rc:
            return rc
        if a.dry_run:
            print(f'run: would execute the saved OA native model on {a.backend}; nothing was run')
            return 0
        return run_model_repository(out, backend=a.backend, host=a.host, gpu=a.gpu,
                                    budget=a.budget, verify=a.verify, dry_run=False)
    if plan is not None and not plan.has_native:
        from openassembler.repository import native_error
        a._failure = str(native_error(plan))
        print(f"error: {a._failure}", file=sys.stderr)
        return 1
    try:
        # Validate execution flags before incurring remote build work.
        validate_options(_run_opts(a, str(output_dir(Path(a.src), a.out_dir))), check_spec=True)
        compile_rc, out, only = _native_build(a)
        if a.dry_run:
            print(f'run: would execute the newly compiled kernels on {a.host or a.backend}; nothing was run')
            return compile_rc
        if not only:
            print('error: nothing compiled, so nothing to run', file=sys.stderr)
            return 1
        print(f'\nrunning {len(only)} newly compiled kernel(s) ({a.host or a.backend})')
        a.run_kernel = None
        a.source = None
        rc = run_command(_run_opts(a, str(out), only))
        return 1 if 1 in (rc, compile_rc) else max(rc, compile_rc)
    except (project.ProjectError, RunError, TransportError, OSError, ValueError) as exc:
        a._failure = str(exc)
        print(f'error: {exc}', file=sys.stderr)
        return 1


def cmd_verify(a: argparse.Namespace) -> int:
    """Run an already-built cubin's SASS through the emulator and compare
    against the PTX reference interpreter on random inputs (same check
    compile_check.py and bench_build.py use before trusting a build). Only
    for kernels matching the probe corpus's fixed (out, in, k, kf) ABI --
    for your own kernel's own signature, use `compile --verify`, which picks
    this or the generic checker (openassembler.verify) automatically."""
    import random

    from openassembler import oracle
    from openassembler.emu.probe import run_probe

    try:
        sass = oracle.disasm_text(a.cubin)
    except (oracle.OracleError, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"error: {getattr(e, 'stderr', '') or e}", file=sys.stderr)
        return 1
    ptx = Path(a.ptx).read_text()
    rng = random.Random(0)

    sys.path.insert(0, str(TOOLS))
    import diff_emu  # noqa: E402

    trials = getattr(a, "trials", 6) or 6
    for _ in range(trials):
        inputs = [tuple(diff_emu.rand32(rng) for _ in range(4)) for _ in range(32)]
        k, kf = diff_emu.rand32(rng), diff_emu.rand32(rng)
        got = run_probe(sass, inputs, k, kf)
        if run_probe(sass, inputs, k, kf, poison=0xDEADBEEF) != got:
            print("FAIL: output depends on uninitialized register contents", file=sys.stderr)
            return 1
        exp = diff_emu.run_ptx(ptx, a.kernel, inputs, k, kf)
        for lane, (g, (e, undef, approx)) in enumerate(zip(got, exp)):
            if not undef and not diff_emu.same("verify", g, e, approx):
                print(f"FAIL: lane {lane}: ours {g} vs PTX interpreter {e}", file=sys.stderr)
                return 1
    print(f"OK: {trials} random trials, bit-exact against the PTX reference interpreter")
    return 0


def cmd_check(a: argparse.Namespace) -> int:
    """Compile the full probe corpus, validate with nvdisasm, check semantics
    against the PTX interpreter. Needs the probe corpus (tools/build_corpus.py)
    and ptxas. Wraps tools/compile_check.py."""
    if not any((ROOT / "corpus" / "sm_80" / "ptx").glob("*.ptx")):
        print("error: the probe corpus isn't built yet (corpus/ is build output, not in git).\n"
              "  Build it once:  openassembler corpus   (needs the CUDA toolkit, ~20 s)", file=sys.stderr)
        return 1
    argv = []
    if a.only:
        argv += ["--only", a.only]
    argv += ["--trials", str(a.trials), "--compiler", a.compiler]
    return _run_tool("compile_check.py", argv)


def cmd_corpus(a: argparse.Namespace) -> int:
    """Build the probe corpus (541 PTX probes -> ptxas -> cubin + SASS) under
    corpus/<arch>/. Needed once before `check`, `gpu run`, or the offline
    A100-replay tests; needs the CUDA toolkit. Wraps tools/build_corpus.py."""
    from openassembler import oracle

    if not oracle.available("ptxas"):
        print(f"error: {oracle.NEEDS_TOOLKIT}", file=sys.stderr)
        return 1
    return _run_tool("build_corpus.py", [a.arch])


def _need_toolkit() -> int | None:
    from openassembler import oracle

    if not oracle.available("ptxas"):
        print(f"error: {oracle.NEEDS_TOOLKIT}", file=sys.stderr)
        return 1
    return None


def cmd_bench(a: argparse.Namespace) -> int:
    """Build the microbenchmark cubins at every optimization level
    (safe/mined/sched/opt), each checked in the emulator first. Wraps
    tools/bench_build.py. Pair with `openassembler gpu run` to time them."""
    if (rc := _need_toolkit()) is not None:
        return rc
    return _run_tool("bench_build.py", ["--arch", a.arch])


def cmd_fastmm(a: argparse.Namespace) -> int:
    """Fast matrix multiplication: build Strassen/AlphaTensor-scheme GEMM
    kernels with our compiler (tools/fastmm_build.py), or run a plan on a GPU
    (tools/fastmm_run.py, Modal only -- for SSH/local use `gpu run` on the
    cubins fastmm build produces)."""
    if a.fastmm_cmd == "build":
        if (rc := _need_toolkit()) is not None:
            return rc
        argv = ["--arch", a.arch, "--variant", a.variant]
        if a.only:
            argv += ["--only", a.only]
        return _run_tool("fastmm_build.py", argv)
    if a.fastmm_cmd == "run":
        argv = ["--n", str(a.n), "--schemes", a.schemes, "--gpu", a.gpu,
                "--reps", str(a.reps), "--rounds", str(a.rounds), "--budget", str(a.budget)]
        if a.dry_run:
            argv.append("--dry-run")
        return subprocess.run(["modal", "run", str(TOOLS / "fastmm_run.py"), *argv], cwd=ROOT).returncode
    return 1


def cmd_gpu(a: argparse.Namespace) -> int:
    """Run probe or built cubins on a real GPU and compare against the
    emulator / PTX spec. Default backend is any machine you can SSH into with
    an NVIDIA driver (no CUDA toolkit needed there) -- this is the
    reproduction path anyone without a cloud GPU account can use. `--backend
    modal` is what produced the numbers recorded in corpus/ for this repo."""
    if a.backend == "modal":
        argv = ["--gpu", a.gpu, "--which", a.which, "--trials", str(a.trials), "--budget", str(a.budget)]
        if a.only:
            argv += ["--only", a.only]
        if a.dry_run:
            argv.append("--dry-run")
        cmd = ["modal", "run", str(TOOLS / "gpu_modal.py"), *argv]
        print("$", " ".join(shlex.quote(c) for c in cmd))
        return subprocess.run(cmd, cwd=ROOT).returncode
    # ssh or local: same script, tools/gpu_run.py
    if a.dry_run:
        # no GPU, no network: just say what would be sent (gpu_run.py itself has no --dry-run)
        sys.path[:0] = [str(ROOT), str(TOOLS)]
        import gpu_common as G

        try:
            jobs = G.build_jobs("sm_80", a.which, a.only, a.trials, 1)
        except SystemExit as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"dry run: {len(jobs)} {a.which} probes x {a.trials} trials would run on "
              f"{'this machine' if a.backend == 'local' else a.backend}; nothing was sent")
        return 0
    if a.backend == "local":
        import platform

        if platform.system() != "Linux":
            print("error: --backend local runs the probes on THIS machine's NVIDIA GPU, which needs "
                  f"Linux with an NVIDIA driver (this is {platform.system()}). Use --backend user@host "
                  "to run on a Linux GPU box over SSH, or --backend modal for a cloud GPU.", file=sys.stderr)
            return 1
    argv = ["--which", a.which, "--trials", str(a.trials), "--budget", str(a.budget)]
    if a.only:
        argv += ["--only", a.only]
    if a.backend == "local":
        argv.append("--local")
    else:
        argv += ["--ssh", a.backend]
    return _run_tool("gpu_run.py", argv)


def cmd_status(a: argparse.Namespace) -> int:
    """Print the headline, reproducible numbers from docs/STATUS.md."""
    status = ROOT / "docs" / "STATUS.md"
    text = status.read_text()
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines[:-1])
                  if line.startswith("|") and lines[i + 1].startswith("|---")), None)
    if start is None:
        print("\n".join(lines[:min(40, len(lines))]))
    else:
        end = start
        while end < len(lines) and lines[end].startswith("|"):
            end += 1
        print("\n".join(lines[start:end]))
    print(f"\nFull detail, evidence files, and reproduction commands: {status.relative_to(ROOT)}")
    return 0


COMPILE_HELP = """Compile a source file or a whole source tree to SASS and CUDA binaries.

  openassembler compile examples/div_by_const.ptx -o out --verify
      -> output/cubin/out.cubin  output/sass/out.sass  (+ output/manifest.json, output/.build/)
  openassembler compile my_repo/ --backend modal
      -> my_repo/output/cubin/ and my_repo/output/sass/
  openassembler compile my_repo/ --backend me@gpubox --out-dir build_oa
      -> build_oa/cubin/<path>.cubin and build_oa/sass/<path>.sass for every kernel source

In a directory: .ptx with .entry, .cu/.cuh/.cpp/.cc/.cxx with __global__ (compiled as
CUDA by nvcc --ptx), and .py with an openassembler.pyfront @kernel are compiled; all
other files (host code, shell scripts, docs, ...) are skipped and counted. In a git
repo, .gitignore is respected. All kernels of one source file go into one cubin.

A directory with [model] in openassembler.toml captures its CPU model factory
and emits OA forward/backward/loss/update artifacts for its exact shape contract.
--frontend model (or torch) selects this explicitly; --frontend native selects
static kernels. Triton/Inductor and --fallback ptxas are disabled.
CUDA C/C++ uses nvcc only to emit PTX. All workload executable GPU instructions
are emitted by OA. --backend local|modal|user@host selects the transport.
run only loads a saved native build; it never recompiles.

"""

RUN_HELP = """Execute saved OA-native cubins and model launch plans.

  openassembler run my_repo/ --backend modal
  openassembler run my_repo/ --backend user@a100 --verify-application
  openassembler run output/gemm --backend local
  openassembler run --spec pipeline.json
  openassembler run --backend modal --dry-run

The model route loads plan.json and its exact native ledger, weight bytes and
OA cubins. --verify-application checks independent CPU references, gradients,
state, guards and profiler ownership. No replay GPU compilation is allowed.

Standalone kernels use a convenience elementwise harness (block256) or your
--spec JSON buffers/launches. Buffer types: f32/i32/u32/f16/bf16. Generated
GEMM and rowwise plans carry their actual sizes and launches. Execution smoke
checks are not general numerical proofs; no ptxas reference is launched.
Legacy --application/--app-arg vendor bundles are disabled. --out-dir chooses
a saved custom build location; run never compiles.
"""

CNR_HELP = """Compile SRC (file or directory, same options as `compile`) into --out-dir, then
run every kernel that compiled (same options as `run`). Kernels that failed to compile
are reported and not run. Exit code is nonzero if anything failed.

  openassembler cnr my_repo/ --backend modal --verify
  openassembler cnr my_repo/ --backend me@gpubox --verify
  openassembler cnr examples/vector_add.cu --backend modal --dry-run

--backend selects both compilation and execution. --build-backend can override
only the build location. --dry-run plans both stages without builds or GPU jobs.
"""


def _add_compile_args(pc: argparse.ArgumentParser) -> None:
    pc.add_argument("-o", "--out", default=None, metavar="NAME",
                    help="output name for a single file: output/cubin/NAME.cubin + output/sass/NAME.sass "
                         "(default: the source file's stem)")
    pc.add_argument("--out-dir", default=None, metavar="DIR",
                    help="output folder (default: REPO/output for directories; ./output for files)")
    pc.add_argument("--kernel", default=None, help="compile only this kernel")
    pc.add_argument("--arch", default="sm_80")
    pc.add_argument("--opt-level", choices=OPT_LEVELS, default="opt")
    pc.add_argument("--verify", action="store_true",
                    help="check standalone kernels with the emulator; native models compare CPU references on CUDA")
    pc.add_argument("--trials", type=int, default=6)
    pc.add_argument("--fallback", choices=["ptxas"], default=None,
                    help="retired compatibility flag: vendor GPU fallback is rejected")
    pc.add_argument("-I", "--include", action="append", default=[], metavar="DIR",
                    help="extra #include directory for nvcc (repeatable; the file's dir and the repo root "
                         "are always included)")
    pc.add_argument("--nvcc-flag", action="append", default=[], metavar="FLAG",
                    help="extra nvcc flag, e.g. --nvcc-flag=-DTILE=32 (repeatable)")
    pc.add_argument("--dump-ptx", action="store_true", help="print the PTX each source compiled to")
    pc.add_argument("--no-sass", action="store_true", help="write only the cubins; skip the nvdisasm listing")
    pc.add_argument("--frontend", choices=("auto", "native", "model", "torch", "triton", "inductor"), default="auto",
                    help="auto discovers native kernels or a configured [model]; model/torch compile native OA model graphs; vendor backends reject")
    pc.add_argument("--build-backend", default=None, metavar="auto|local|modal|user@host",
                    help="override only compilation's backend (otherwise --backend or application config)")
    pc.add_argument("--build-host", default=None, metavar="USER@HOST",
                    help="SSH target for --build-backend ssh; --host is shared by build and run")
    pc.add_argument("--build-gpu", default=None, help="override only the native model build GPU")
    pc.add_argument("--build-budget", type=float, default=None, metavar="SECONDS",
                    help="override only the compilation budget")
    pc.add_argument("--dry-run", action="store_true",
                    help="plan compilation and execution without writing outputs or submitting jobs")
    _add_backend_args(pc)


def _add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", default="auto", metavar="auto|local|modal|user@host",
                        help="build/run here, on Modal, or on an existing Linux SSH host; auto uses application config or local")
    parser.add_argument("--host", default=None, metavar="USER@HOST",
                        help="SSH destination for --backend ssh (SSH config aliases are supported)")
    parser.add_argument("--gpu", default=None,
                        help="Modal GPU for native model capture/execution (default: config or A100-40GB)")
    parser.add_argument("--budget", type=float, default=None,
                        help="remote build / GPU-stage limit, seconds (native build 900, kernel run 300; app config otherwise)")
    parser.add_argument("--job-timeout", type=float, default=None,
                        help="worker watchdog in seconds (kernel run 60; application config otherwise)")


def _add_run_args(pr: argparse.ArgumentParser, *, include_dry_run: bool = True,
                  include_backend: bool = True) -> None:
    if include_backend:
        _add_backend_args(pr)
    pr.add_argument("--spec", default=None, metavar="SPEC.json", help="buffers/launches to use (see above)")
    pr.add_argument("-n", "--n", type=int, default=1 << 20, help="auto-harness elements per buffer (default 1048576)")
    pr.add_argument("--reps", type=int, default=20, help="timed repetitions (median reported)")
    if include_dry_run:
        pr.add_argument("--dry-run", action="store_true", help="print what would run (and the cost); touch no GPU")
    pr.add_argument("--save-jobs", default=None, metavar="FILE", help="also write the GPU jobs pickle here")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="openassembler", description=__doc__.split("\n\n")[0],
                                 epilog=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"openassembler {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("compile", help="compile a file or a whole repo to SASS + cubin (into output/)",
                        description=COMPILE_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    pc.add_argument("src", metavar="SRC", nargs="?", default=".", help="source file or repository (default: current directory)")
    _add_compile_args(pc)
    pc.set_defaults(func=cmd_compile)

    pi = sub.add_parser("inspect", help="show source coverage without running code or using a GPU")
    pi.add_argument("src", metavar="SRC", nargs="?", default=".")
    pi.add_argument("--json", action="store_true", help="print the complete machine-readable inventory")
    pi.set_defaults(func=cmd_inspect)

    pm = sub.add_parser("gemm", help="compile a specialized A100 FP16/BF16 tensor GEMM",
                        description="Compile C=A@B.T with FP16/BF16 inputs and FP32 accumulation/output. "
                                    "A[M,K] and B[N,K] must be contiguous in K. M/N must be multiples "
                                    "of 64 (128 for the larger tile), K a multiple of 128. Exact measured "
                                    "shape policies are selected automatically; manual knobs override dispatch.")
    for dim in ("m", "n", "k"):
        pm.add_argument(f"--{dim}", type=int, required=True)
    pm.add_argument("--dtype", choices=("f16", "bf16"), default="f16")
    pm.add_argument("--arch", default="sm_80")
    pm.add_argument("--dispatch", choices=("auto", "manual"), default="auto",
                    help="exact measured native shape selection (default), or baseline/manual policy")
    pm.add_argument("--tile", choices=("128x128", "64x64"), default=None,
                    help="explicit CTA tile; selects manual dispatch")
    pm.add_argument("--copy-address-init", choices=("original", "reuse"), default=None,
                    dest="address_init", help="manual 64x64 copy-address setup: separate per-chunk bases or checked 64-bit derived offsets")
    pm.add_argument("--schedule", choices=("stagepair", "combined"), default=None,
                    help="manual alternating-stage specialization or combined reference")
    pm.add_argument("--fragment-schedule", choices=("source", "load_next"), default=None,
                    help="manual source-order or bounded FP16/BF16 load-next schedule")
    pm.add_argument("--cta-order", choices=("m_fast", "morton8", "grouped_n8"), default=None,
                    help="A100 logical CTA traversal; morton8 requires complete 8x8 groups; grouped_n8 is an opt-in N-grouped launch")
    pm.add_argument("--epilogue", choices=("none", "bias_residual", "bias_residual_relu"), default="none",
                    help="fuse FP32 bias and residual, optionally ReLU, into the output epilogue")
    pm.add_argument("--pointer-mode", choices=("legacy", "direct"), default=None,
                    help="direct reuses aligned pointer homes for staged copies; measured wins on MLP "
                         "shapes, regression on tall GEMM; stagepair/m_fast/unfused only")
    pm.add_argument("--pointer-advance", choices=("late", "early", "auto"), default=None,
                    help="late (default), the validated early-placement candidate, or shape-specific auto")
    pm.add_argument("--autotune", choices=("local", "modal"),
                    help="compile and validate admitted native policies, then time them on an A100 GPU")
    pm.add_argument("--tuning-record", metavar="JSON",
                    help="write a GPU tuning record with --autotune, or reuse an explicitly supplied current record")
    pm.add_argument("--tune-rounds", type=int, default=3, help="paired GPU tuning rounds (minimum 3)")
    pm.add_argument("--tune-samples", type=int, default=12, help="samples per policy per round (minimum 8)")
    pm.add_argument("--out-dir", default="output")
    pm.add_argument("-o", "--out", default="gemm", metavar="NAME")
    pm.add_argument("--no-sass", action="store_true")
    pm.add_argument("--emit-only", action="store_true", help="write generated PTX without invoking the toolkit")
    pm.set_defaults(func=cmd_gemm)

    pw = sub.add_parser("rowwise", help="compile cooperative A100 FP32 softmax or LayerNorm",
                        description="Compile contiguous FP32 rows using one full warp per row, "
                                    "four rows per CTA. Width 1..1024, with row/column tails. "
                                    "LayerNorm uses population variance and has no affine parameters.")
    pw.add_argument("--op", choices=("softmax", "layernorm"), required=True)
    pw.add_argument("--rows", type=int, required=True)
    pw.add_argument("--width", type=int, required=True)
    pw.add_argument("--epsilon", type=float, default=1e-5, help="positive FP32 LayerNorm epsilon")
    pw.add_argument("--arch", default="sm_80")
    pw.add_argument("--dispatch", choices=("auto", "manual"), default="auto",
                    help="exact shape/operation measured A100 geometry (default), or the four-warp reference")
    pw.add_argument("--warps-per-cta", type=int, choices=(1, 2, 4, 8), default=None,
                    help="explicit manual whole-warp geometry; overrides auto dispatch")
    pw.add_argument("--out-dir", default="output")
    pw.add_argument("-o", "--out", default=None, metavar="NAME")
    pw.add_argument("--no-sass", action="store_true")
    pw.add_argument("--emit-only", action="store_true")
    pw.set_defaults(func=cmd_rowwise)

    pr = sub.add_parser("run", help="run a repository's saved build, or an explicit artifact directory",
                        description=RUN_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    pr.add_argument("outdir", metavar="PATH", nargs="?", default=None,
                    help="source repository/file or built output directory (default: current repo/output)")
    pr.add_argument("--out-dir", default=None, metavar="DIR", help="custom build root for the source PATH")
    pr.add_argument("--kernel", default=None, help="run only this kernel")
    pr.add_argument("--source", default=None, metavar="FILE",
                    help="only kernels compiled from FILE (needed when --kernel is in several sources)")
    pr.add_argument("--from-results", default=None, metavar="RES.pkl",
                    help="re-report a results pickle from an earlier run instead of running anything")
    pr.add_argument("--application", default=None, help="retired compatibility flag: legacy vendor application bundles are disabled")
    pr.add_argument("--app-arg", action="append", default=[], help="application arguments; must match the bundle")
    pr.add_argument("--verify-application", action="store_true", help="verify eager and compiled application state")
    _add_run_args(pr)
    pr.set_defaults(func=cmd_run)

    pn = sub.add_parser("cnr", help="compile and run: compile SRC, then run what compiled",
                        description=CNR_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    pn.add_argument("src", metavar="SRC", nargs="?", default=".", help="source file or repository (default: current directory)")
    _add_compile_args(pn)
    _add_run_args(pn, include_dry_run=False, include_backend=False)
    pn.set_defaults(func=cmd_cnr)

    pv = sub.add_parser("verify", help="check a compiled cubin's SASS against the PTX reference interpreter")
    pv.add_argument("cubin")
    pv.add_argument("ptx")
    pv.add_argument("--kernel", required=True)
    pv.add_argument("--trials", type=int, default=6)
    pv.set_defaults(func=cmd_verify)

    pk = sub.add_parser("check", help="compile + validate the full probe corpus (tools/compile_check.py)")
    pk.add_argument("--only", default="")
    pk.add_argument("--trials", type=int, default=6)
    pk.add_argument("--compiler", choices=["ra", "naive"], default="ra")
    pk.set_defaults(func=cmd_check)

    pco = sub.add_parser("corpus", help="build the probe corpus that check / gpu run / some tests need")
    pco.add_argument("--arch", default="sm_80")
    pco.set_defaults(func=cmd_corpus)

    pb = sub.add_parser("bench", help="build the optimization-level benchmark cubins (tools/bench_build.py)")
    pb.add_argument("--arch", default="sm_80")
    pb.set_defaults(func=cmd_bench)

    pf = sub.add_parser("fastmm", help="Strassen/AlphaTensor fast matrix multiplication")
    fsub = pf.add_subparsers(dest="fastmm_cmd", required=True)
    fb = fsub.add_parser("build", help="build the GEMM/lincomb kernels for every scheme")
    fb.add_argument("--arch", default="sm_80")
    fb.add_argument("--variant", choices=OPT_LEVELS, default="opt")
    fb.add_argument("--only", default="")
    fr = fsub.add_parser("run", help="run a fast-matmul plan on a Modal GPU")
    fr.add_argument("--n", type=int, default=4096)
    fr.add_argument("--schemes", default="strassen_2x2x2,at_2x2x2,naive_2x2x2")
    fr.add_argument("--gpu", default="A100-40GB")
    fr.add_argument("--reps", type=int, default=5)
    fr.add_argument("--rounds", type=int, default=2)
    fr.add_argument("--budget", type=float, default=400)
    fr.add_argument("--dry-run", action="store_true")
    pf.set_defaults(func=cmd_fastmm)

    pg = sub.add_parser("gpu", help="run cubins on a real GPU (SSH/local/Modal) and compare")
    gsub = pg.add_subparsers(dest="gpu_cmd", required=True)
    gr = gsub.add_parser("run", help="build jobs, run them on a GPU, compare against the emulator/PTX spec")
    gr.add_argument("--backend", default="local", metavar="local|modal|user@host",
                     help="'local' (this machine's GPU), 'modal' (cloud, needs `modal setup`), "
                          "or an SSH target user@host (any Linux box with an NVIDIA driver)")
    gr.add_argument("--which", default="ptxas", choices=["ptxas", "ours"])
    gr.add_argument("--only", default="")
    gr.add_argument("--trials", type=int, default=4)
    gr.add_argument("--budget", type=float, default=300)
    gr.add_argument("--gpu", default="A100-40GB", help="Modal GPU type (--backend modal only)")
    gr.add_argument("--dry-run", action="store_true", help="print the job count/cost estimate, run nothing")
    gr.set_defaults(func=cmd_gpu, backend_cmd="run")

    ps = sub.add_parser("status", help="print the headline results from docs/STATUS.md")
    ps.set_defaults(func=cmd_status)

    pbench = sub.add_parser("benchmark", help="run or describe the full CLI integration matrix")
    pbench.add_argument("--backend", default="modal", help="GPU backend: modal, local, or user@host")
    pbench.add_argument("--build-backend", choices=("modal", "local", "ssh"), default="modal")
    pbench.add_argument("--build-host", default=None, metavar="USER@HOST")
    pbench.add_argument("--out-dir", default="output/cli-benchmark")
    pbench.add_argument("--budget", type=float, default=300)
    pbench.add_argument("--execute", action="store_true")
    pbench.add_argument("--skip-optional", action="store_true")
    pbench.set_defaults(func=lambda a: _run_tool("cli_benchmark.py", [
        "--backend", a.backend, "--build-backend", a.build_backend,
        "--out-dir", a.out_dir, "--budget", str(a.budget),
        *( ["--build-host", a.build_host] if a.build_host else []),
        *( ["--execute"] if a.execute else []), *( ["--skip-optional"] if a.skip_optional else [])]))

    return p


def dispatch(args: argparse.Namespace) -> int:
    """Run a parsed command. Ctrl-C anywhere: remove half-written outputs (temp files
    of writes in flight; finished files are renamed into place atomically), print one
    line, exit 130 -- the shell convention for SIGINT."""
    from openassembler.project import cleanup_pending, ProjectError

    try:
        rc = args.func(args)
        if rc:
            _save_failure(args, getattr(args, '_failure', 'Compilation or execution failed; see worker output and build manifest.'), rc)
        return rc
    except KeyboardInterrupt:
        cleanup_pending()
        print("\ninterrupted: no half-written files were left behind", file=sys.stderr)
        return 130
    except BrokenPipeError:          # `openassembler ... | head`
        return 0
    except (ValueError, OSError, ProjectError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        _save_failure(args, str(exc))
        return 1
    except Exception as exc:
        import traceback
        print(f'internal compiler error ({type(exc).__name__}): {exc}', file=sys.stderr)
        _save_failure(args, traceback.format_exc())
        return 1


def _save_failure(args, error, rc=1):
    from openassembler.diagnostics import save_failure
    try:
        path = save_failure(args, error, rc)
        if path is not None:
            print(f'build diagnostic: {path}', file=sys.stderr)
    except Exception as exc:
        # Reporting must preserve the original build failure and exit status.
        print(f'could not save build diagnostic: {exc}', file=sys.stderr)


def source_arguments(raw, parser):
    """Use source-path shorthand consistently in the CLI and interactive shell."""
    commands = next(action.choices for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    if raw and raw[0] not in commands and not raw[0].startswith('-') and Path(raw[0]).exists():
        return ['compile', *raw]
    return raw


def main(argv: list[str] | None = None) -> int:
    raw = sys.argv[1:] if argv is None else argv
    if not raw:
        from openassembler.ui.repl import run as run_repl
        return run_repl()

    # First `openassembler <command>` typed directly into a given shell session
    # shows the logo once; later commands in that same session stay quiet
    # (openassembler/ui/session.py). The REPL shows it once at startup instead,
    # on its own, so this only applies to one-shot invocations. Skipped when
    # stdout isn't a terminal (scripts, CI, pipes) so output stays clean, and
    # for -h/--help/--version so the banner doesn't eat into `| head` / pager output.
    if sys.stdout.isatty() and not {"-h", "--help", "--version"} & set(raw):
        from openassembler.ui.banner import render
        from openassembler.ui.session import consume_show_once

        if consume_show_once():
            print(render())
            print()

    parser = build_parser()
    # A literal source path is the common one-shot build operation. Explicit
    # commands retain their meaning even if a same-named file exists.
    args = parser.parse_args(source_arguments(raw, parser))
    if args.cmd == "gpu" and args.gpu_cmd != "run":
        parser.error("gpu: only 'run' is implemented")
    return dispatch(args)


if __name__ == "__main__":
    raise SystemExit(main())
