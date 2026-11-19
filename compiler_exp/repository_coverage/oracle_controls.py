"""Check that missed discovery/parser fixtures are accepted by NVIDIA tools.

No GPU execution or proprietary binary redistribution. Outputs stay in temporary
storage; retain only tool versions, source text and diagnostics in the report.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from openassembler import oracle, project
from openassembler.cppfront import compile_to_ptx_module
from openassembler.ptx.parse import parse_module


def run():
    results = []
    declarations = {
        "visible_entry_control": ".visible .entry normal() { ret; }",
        "bare_entry": ".entry bare() { ret; }",
        "weak_entry": ".weak .entry weak() { ret; }",
        "launch_bounds": ".visible .entry bounded() .maxntid 128, 1, 1 { ret; }",
        "mixed_entry_inventory": ".visible .entry normal() { ret; }\n.entry bare() { ret; }",
    }
    with tempfile.TemporaryDirectory(prefix="oracle-", dir=Path(__file__).parent) as temporary:
        root = Path(temporary)
        for name, declaration in declarations.items():
            text = ".version 8.7\n.target sm_80\n.address_size 64\n" + declaration + "\n"
            source = root / f"{name}.ptx"
            source.write_text(text)
            try:
                done = oracle.run_tool("ptxas", "-arch=sm_80", oracle.tool_path(source),
                                       "-o", oracle.tool_path(source.with_suffix(".cubin")), check=False)
                results.append({"case": name, "source": text, "ptxas_rc": done.returncode,
                                "ptxas_stdout": done.stdout, "ptxas_stderr": done.stderr,
                                "oa_parsed_entries": list(parse_module(text).kernels)})
            except Exception as exc:
                results.append({"case": name, "status": "unavailable", "error": str(exc)})
        (root / "qualifiers.h").write_text("#define OA_GLOBAL __global__\n")
        source = root / "macro.cu"
        source.write_text('#include "qualifiers.h"\nextern "C" OA_GLOBAL void k(float *p) { p[0] = 1; }\n')
        try:
            text, names = compile_to_ptx_module(source)
            classification = project.classify(source, source.name)
            results.append({"case": "cuda_header_macro", "nvcc_status": "accepted",
                            "nvcc_ptx_entries": names, "oa_discovery_language": classification.language,
                            "oa_discovery_skip_reason": classification.skip_reason})
        except Exception as exc:
            results.append({"case": "cuda_header_macro", "status": "unavailable", "error": str(exc)})
    versions = {}
    for tool in ("ptxas", "nvcc"):
        try:
            done = oracle.run_tool(tool, "--version", check=False)
            versions[tool] = {"rc": done.returncode, "text": done.stdout + done.stderr}
        except Exception as exc:
            versions[tool] = {"error": str(exc)}
    report = {"scope": "NVIDIA frontend/assembler acceptance only; no GPU execution", "versions": versions,
              "results": results}
    output = Path(__file__).with_name('oracle-controls-replay-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
    if output.exists():
        raise ValueError('refusing to overwrite retained evidence')
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
