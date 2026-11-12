"""Build source-identical runtime-indexing controls and isolated hazard ablations."""
from __future__ import annotations
import base64
import dataclasses
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build(tag="baseline", ablations=False):
    from openassembler import oracle
    from openassembler.build import Options, compile_paths, generate
    from openassembler.backend import compile as scheduler
    from openassembler.container.cubin import splice_text
    from openassembler.ptx.parse import parse_module

    out = HERE / "builds" / tag
    out.mkdir(parents=True, exist_ok=False)
    (out / "cases.cu").write_bytes((HERE / "cases.cu").read_bytes())
    (out / "run_modal.py").write_bytes((HERE / "run_modal.py").read_bytes())
    products = []
    saved_async = scheduler.MEM_ASYNC
    experiment_async = tuple(op for op in saved_async if op != "F2I") if ablations else saved_async
    variants = {tag: ()}
    if ablations:
        variants = {"baseline": (), "f2i_read_barrier": ("F2I",), "i2f_read_barrier": ("I2F",),
                    "both_read_barriers": ("F2I", "I2F")}
    for level in ("safe", "mined", "sched", "opt"):
        target = out / level
        if compile_paths(HERE / "cases.cu", target, Options(opt_level=level)).rc:
            raise RuntimeError(f"compilation failed: {level}")
        manifest = json.loads((target / "manifest.json").read_text())
        entry = manifest["entries"][0]
        reference = (target / entry["reference"]).read_bytes()
        source = (target / entry["ptx"]).read_text()
        kernels = parse_module(source).kernels
        for variant, extra in variants.items():
            scheduler.MEM_ASYNC = experiment_async + extra
            native = reference
            controls = {}
            for name, kernel in kernels.items():
                code, labels, words, exits, regs = generate(kernel, "sm_80", level)
                native = splice_text(native, name, words, regs, exits, replace_helpers=True)
                controls[name] = [{"text": text, "control": dataclasses.asdict(ctl)}
                                  for text, ctl in code]
            path = target / f"{variant}.cubin"
            path.write_bytes(native)
            sass = oracle.run_tool("nvdisasm", "-hex", "-c", str(path)).stdout
            (target / f"{variant}.sass").write_text(sass)
            (target / f"{variant}.controls.json").write_text(json.dumps(controls, indent=2))
            for method, binary in ((variant, native), ("ptxas", reference)):
                if method == "ptxas" and variant != next(iter(variants)):
                    continue
                products.append({"level": level, "method": method,
                                 "cubin": base64.b64encode(binary).decode(),
                                 "cubin_sha256": sha(binary), "ptx_sha256": sha(source.encode())})
        scheduler.MEM_ASYNC = saved_async
    payload = {"tag": tag, "products": products,
               "kernels": list(kernels),
               "source_sha256": sha((HERE / "cases.cu").read_bytes()),
               "compiler_sha256": {str(p.relative_to(ROOT)): sha(p.read_bytes())
                                   for p in (ROOT / "openassembler").rglob("*.py")}}
    (out / "payload.json").write_text(json.dumps(payload))
    return out / "payload.json"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--ablations", action="store_true")
    args = parser.parse_args()
    print(build(args.tag, args.ablations))
