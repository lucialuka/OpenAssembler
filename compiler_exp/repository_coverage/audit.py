"""Reproduce source-level coverage boundaries; never launch GPU work.

Run from the checkout with .env/bin/python for the optional CPU Torch probes.
These are diagnostic experiments, not fixes or hardware correctness results.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openassembler import project
from openassembler.backend.ra import compile_kernel
from openassembler.cli import main
from openassembler.ptx.parse import parse_module
from openassembler.repository import inspect


def outcome(name, action):
    try:
        details = action()
        rejected = (("dry_run_rc" in details and details["dry_run_rc"] != 0)
                    or details.get("entry_inventory_matches") is False)
        return {"case": name, "status": "rejected" if rejected else "accepted", "details": details}
    except Exception as exc:
        return {"case": name, "status": "rejected", "error_type": type(exc).__name__,
                "error": str(exc)}


def ptx_case(instructions, params=""):
    text = (".version 8.7\n.target sm_80\n.address_size 64\n"
            f".visible .entry probe({params}) {{\n{instructions}\nret;\n}}\n")
    module = parse_module(text)
    code, stats = compile_kernel(module.kernels["probe"])
    return {"instructions": len(code), "registers": stats["regs"],
            "stage": "instruction selection/allocation/scheduling; no encoding or GPU execution"}


def parser_case(declarations):
    from openassembler.cppfront import _kernel_names
    text = ".version 8.7\n.target sm_80\n.address_size 64\n" + declarations
    discovered, parsed = _kernel_names(text), list(parse_module(text).kernels)
    return {"discovered_entries": discovered, "parsed_entries": parsed,
            "entry_inventory_matches": set(discovered) == set(parsed)}


def repository_case(root, files):
    root.mkdir()
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    plan = inspect(root)
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["compile", str(root), "--dry-run", "--backend", "local"])
    return {"native_sources": [c.rel for c in plan.native],
            "build_systems": plan.build_systems, "dependencies": plan.dependencies,
            "model_discovery": plan.model_discovery, "dry_run_rc": rc,
            "stdout": stdout.getvalue(), "stderr": stderr.getvalue()}


def survey():
    entries = []
    for directory in ("openassembler", "tests", "tools", "examples", "exp"):
        for path in sorted((ROOT / directory).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            raw = path.read_bytes()
            record = {"path": path.relative_to(ROOT).as_posix(), "bytes": len(raw),
                      "lines": len(raw.splitlines()), "sha256": hashlib.sha256(raw).hexdigest()}
            try:
                tree = ast.parse(raw, filename=str(path))
                record["description"] = ast.get_docstring(tree)
                record["definitions"] = [n.name for n in tree.body
                                          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
                record["imports"] = sorted({n.module or "" for n in ast.walk(tree)
                                            if isinstance(n, ast.ImportFrom)} |
                                           {a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                                            for a in n.names})
            except SyntaxError as exc:
                record["parse_error"] = str(exc)
            entries.append(record)
    return entries


def run():
    results = [
        outcome("ptx_return_control", lambda: ptx_case("")),
        outcome("ptx_visible_entry_control", lambda: parser_case(".visible .entry normal() { ret; }")),
        outcome("ptx_bare_entry", lambda: parser_case(".entry bare() { ret; }")),
        outcome("ptx_weak_entry", lambda: parser_case(".weak .entry weak() { ret; }")),
        outcome("ptx_launch_bounds", lambda: parser_case(
            ".visible .entry bounded() .maxntid 128, 1, 1 { ret; }")),
        outcome("ptx_mixed_entry_inventory", lambda: parser_case(
            ".visible .entry normal() { ret; }\n.entry bare() { ret; }")),
        outcome("ptx_global_atomic_add", lambda: ptx_case(
            "ld.param.u64 %rd, [p];\natom.global.add.u32 %r, [%rd], 1;", ".param .u64 p")),
        outcome("ptx_fp64_add", lambda: ptx_case(
            "ld.param.f64 %a, [a];\nadd.rn.f64 %b, %a, %a;", ".param .f64 a")),
        outcome("ptx_device_function_call", lambda: ptx_case("call.uni helper, ();")),
        outcome("ptx_register_pressure", lambda: ptx_case(
            "ld.param.u64 %rd, [p];\n"
            + "\n".join(f"ld.global.u32 %r{i}, [%rd+{4*i}];" for i in range(260))
            + "\n" + "\n".join(f"st.global.u32 [%rd+{4*i}], %r{i};" for i in range(260)),
            ".param .u64 p")),
    ]
    with tempfile.TemporaryDirectory(prefix="oa-coverage-", dir=Path(__file__).parent) as temporary:
        base = Path(temporary)
        fixtures = {
            "cuda_literal_control": {"kernel.cu": '__global__ void k(float *p) { p[0] = 1; }\n'},
            "cuda_header_macro": {
                "qualifiers.h": "#define OA_GLOBAL __global__\n",
                "kernel.cu": '#include "qualifiers.h"\nOA_GLOBAL void k(float *p) { p[0] = 1; }\n',
                "CMakeLists.txt": "project(probe LANGUAGES CXX CUDA)\nadd_library(probe SHARED kernel.cu)\n"},
            "sklearn_host_program": {
                "train.py": "from sklearn.ensemble import RandomForestClassifier\n"
                            "model = RandomForestClassifier()\nmodel.fit([[0], [1]], [0, 1])\n",
                "requirements.txt": "scikit-learn\n"},
        }
        for name, files in fixtures.items():
            results.append(outcome(name, lambda name=name, files=files:
                                   repository_case(base / name, files)))
    if importlib.util.find_spec("torch") is not None:
        import torch
        from openassembler.modelfront.graph import capture_model

        class Control(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)

        class Sigmoid(torch.nn.Module):
            def forward(self, x):
                return torch.sigmoid(x)

        class Broadcast(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(8))

            def forward(self, x):
                return x + self.bias

        class Branch(torch.nn.Module):
            def forward(self, x):
                if x.sum() > 0:
                    return x * 2
                return x

        def capture(model):
            plan = capture_model(model, [torch.ones(2, 8)])
            return {"contract": plan["contract"], "kernels": [s["kernel"] for s in plan["steps"]]}

        for name, model in (("torch_relu_control", Control()), ("torch_sigmoid", Sigmoid()),
                            ("torch_bias_broadcast", Broadcast()), ("torch_data_branch", Branch()),
                            ("torch_conv2d", torch.nn.Conv2d(1, 1, 3))):
            results.append(outcome(name, lambda model=model: capture(model)))
        torch_version = torch.__version__
    else:
        torch_version = None
    inventory = survey()
    prefix = sys.argv[1] if len(sys.argv) > 1 else 'replay-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if Path(prefix).name != prefix:
        raise ValueError('report prefix must be a filename stem')
    output = Path(__file__).with_name(prefix + '.json')
    if output.exists():
        raise ValueError('refusing to overwrite retained evidence')
    git = lambda *args: subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    report = {"schema": 1, "scope": "CPU diagnostic baseline; no proposals implemented, no A100 measurements",
              "python": sys.version, "platform": platform.platform(), "torch": torch_version,
              "head": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
              "compiler_sources": [e for e in inventory if e["path"].startswith("openassembler/")],
              "results": results}
    output.write_text(json.dumps(report, indent=2) + "\n")
    Path(__file__).with_name(prefix + '-sources.json').write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps({"report": str(output), "source_files_surveyed": len(inventory),
                      "compiler_files": len(report["compiler_sources"]),
                      "results": [{"case": r["case"], "status": r["status"],
                                   "detail": r.get("error", r.get("details"))} for r in results]}, indent=2))


if __name__ == "__main__":
    run()
