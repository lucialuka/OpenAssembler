"""Re-encode retained GEMM/rowwise candidates before refreshing table identities.

This audits binary continuity, never transfers timing admission to changed code.
Historical raw records/archives remain byte-for-byte untouched.
"""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import zipfile

from openassembler import gemm_dispatch as gd, rowwise_dispatch as rd
from openassembler.backend.tensor import TensorConfig
from openassembler.build import generate as native_generate
from openassembler.container.cubin import regcount, text_words
from openassembler.gemm import source_for
from openassembler.kernels.gemm import GemmPlan, generate as gemm_source
from openassembler.kernels.reductions import ReductionPlan, generate as rowwise_source
from openassembler.ptx.parse import parse_module

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CHECKED_ARTIFACTS = {}


def sha(data):
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=1024)
def encode(source, kernel, pointer_mode="legacy", pointer_advance="late"):
    result = native_generate(parse_module(source).kernels[kernel], "sm_80", "opt",
                             tensor_config=TensorConfig(pointer_mode=pointer_mode,
                                                        pointer_advance=pointer_advance))
    return result[2], result[4]


def checked_file(relative, expected):
    data = (ROOT / relative).read_bytes()
    if sha(data) != expected:
        raise ValueError(f"retained artifact changed: {relative}")
    CHECKED_ARTIFACTS[str(relative)] = {"sha256": expected, "bytes": len(data)}
    return data


def main(update=False):
    CHECKED_ARTIFACTS.clear()
    tables = {"gemm": json.loads(gd.TABLE.read_text()), "rowwise": json.loads(rd.TABLE.read_text())}
    identities = {"gemm": gd.compiler_identity(), "rowwise": rd.compiler_identity()}
    proof = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
             "scope": "Current native instruction words, controls, registers and generated source equal both retained selection and held-out artifacts; no new timing claim",
             "old_compiler_identity": {k: t["compiler_identity"] for k, t in tables.items()},
             "current_compiler_identity": identities,
             "table_before_sha256": {"gemm": sha(gd.TABLE.read_bytes()), "rowwise": sha(rd.TABLE.read_bytes())},
             "compiler_inputs_sha256": {str(p.relative_to(ROOT)): sha(p.read_bytes())
                                         for p in set(gd.compiler_inputs().values()) | set(rd.identity_inputs().values())},
             "gemm": [], "rowwise": [], "fixed_tensor_goldens": []}
    candidate_contracts = set()
    for entry in tables["gemm"]["entries"]:
        for evidence in entry["evidence"]:
            raw = checked_file(evidence["raw_result"], evidence["raw_result_sha256"])
            record = json.loads(raw)
            directory = Path(evidence["raw_result"]).parent
            relative = directory / record["artifact_archive"]["path"]
            checked_file(relative, record["artifact_archive"]["sha256"])
            with zipfile.ZipFile(ROOT / relative) as archive:
                for name, candidate in record["candidates"].items():
                    candidate_contracts.add((json.dumps(entry["contract"], sort_keys=True), name))
                    plan = GemmPlan(**candidate["plan"])
                    source = archive.read(f"{name}/gemm.ptx").decode()
                    assert source_for(plan) == source
                    assert sha(gemm_source(plan).encode()) == candidate["ptx_body_sha256"]
                    binary = archive.read(f"{name}/native.cubin")
                    assert sha(binary) == candidate["artifacts"]["native"]["cubin_sha256"]
                    words, registers = encode(source, "gemm", plan.pointer_mode, plan.pointer_advance)
                    assert words == text_words(binary, "gemm"), (entry["contract"], name)
                    assert registers == regcount(binary, "gemm")
                    proof["gemm"].append({"record": evidence["raw_result"], "policy": name,
                                           "selected": name == entry["policy"],
                                           "ptx_body_sha256": candidate["ptx_body_sha256"],
                                           "native_text_sha256": sha(b"".join(w.to_bytes(16, "little") for w in words)),
                                           "words": len(words), "registers": registers,
                                           "retained_cubin_sha256": sha(binary), "passed": True})
        print("GEMM", entry["contract"]["m"], entry["contract"]["n"], entry["contract"]["k"],
              entry["contract"]["dtype"], len(proof["gemm"]), flush=True)
    for entry in tables["rowwise"]["entries"]:
        contract = entry["contract"]
        for evidence in entry["evidence"]:
            raw = checked_file(evidence["record"], evidence["record_sha256"])
            record = json.loads(raw)
            checked_file(evidence["artifact_archive"], evidence["artifact_archive_sha256"])
            with zipfile.ZipFile(ROOT / evidence["artifact_archive"]) as archive:
                for name, candidate in record["candidates"].items():
                    warps = candidate["certificate"]["warps_per_cta"]
                    plan = ReductionPlan(rows=contract["rows"], width=contract["width"],
                                         op=contract["op"], epsilon=contract["epsilon_f32"],
                                         warps_per_cta=warps)
                    source = archive.read(f"{name}/{plan.op}.ptx").decode()
                    assert rowwise_source(plan) == source
                    binary = archive.read(f"{name}/oa.cubin")
                    words, registers = encode(source, plan.op)
                    assert words == text_words(binary, plan.op), (contract, name)
                    assert registers == regcount(binary, plan.op)
                    proof["rowwise"].append({"record": evidence["record"], "policy": name,
                                              "selected": warps == entry["warps_per_cta"],
                                              "source_sha256": sha(source.encode()),
                                              "native_text_sha256": sha(b"".join(w.to_bytes(16, "little") for w in words)),
                                              "words": len(words), "registers": registers,
                                              "retained_cubin_sha256": sha(binary), "passed": True})
    goldens = {
        (4096, 4096, 4096): {
            "f16": "7db39cc746df3370ccb8158a8394ffa886024bc229a77fee0f53b63b7c475ddd",
            "bf16": "1573a311b93c91342d9c4437ef954624d350a07a07d2d2700ce1c31f2cef41d3"},
        (4096, 16384, 4096): {
            "f16": "a84a7705738b89ae2aa2cfc903db92c0ef34bc7b88726ffc75c95f066927a144",
            "bf16": "40927857547bb35ddae11575264f1f4c300bd4a8772a1c4cfe82c07518257db4"},
        (16384, 4096, 4096): {
            "f16": "7db39cc746df3370ccb8158a8394ffa886024bc229a77fee0f53b63b7c475ddd",
            "bf16": "1573a311b93c91342d9c4437ef954624d350a07a07d2d2700ce1c31f2cef41d3"}}
    for shape, dtypes in goldens.items():
        for dtype, expected in dtypes.items():
            source = source_for(GemmPlan(*shape, dtype, schedule="stagepair"))
            words, registers = encode(source, "gemm")
            digest = sha(b"".join(w.to_bytes(16, "little") for w in words))
            assert digest == expected and len(words) == 1600 and registers == 216
            proof["fixed_tensor_goldens"].append({"shape": list(shape), "dtype": dtype,
                                                   "native_text_sha256": digest, "words": len(words),
                                                   "registers": registers, "passed": True})
    assert gd.compiler_identity() == identities["gemm"]
    assert rd.compiler_identity() == identities["rowwise"]
    proof["counts"] = {"gemm_entries": len(tables["gemm"]["entries"]),
                       "gemm_candidate_contracts": len(candidate_contracts),
                       "gemm_candidate_session_checks": len(proof["gemm"]),
                       "rowwise_entries": len(tables["rowwise"]["entries"]),
                       "rowwise_candidate_session_checks": len(proof["rowwise"]),
                       "fixed_tensor_golden_checks": len(proof["fixed_tensor_goldens"])}
    proof["retained_artifacts"] = dict(sorted(CHECKED_ARTIFACTS.items()))
    proof["passed"] = True
    target = HERE / "dispatch-continuity.json"
    if target.exists():
        target = HERE / ("dispatch-continuity-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json")
    data = (json.dumps(proof, indent=2) + "\n").encode()
    target.write_bytes(data)
    if update:
        for kind, table in tables.items():
            table["measurement_compiler_identity"] = table.get("measurement_compiler_identity", table["compiler_identity"])
            table["compiler_identity"] = identities[kind]
            table["source_continuity"] = {"audit": str(target.relative_to(ROOT)), "sha256": sha(data),
                                          "previous_compiler_identity": proof["old_compiler_identity"][kind],
                                          "current_compiler_identity": identities[kind],
                                          "native_text_and_registers_unchanged": True,
                                          "scope": proof["scope"]}
            for entry in table["entries"]:
                if "compiler_identity" in entry:
                    entry["measurement_compiler_identity"] = entry.get("measurement_compiler_identity", entry["compiler_identity"])
                    entry["compiler_identity"] = identities[kind]
            (gd.TABLE if kind == "gemm" else rd.TABLE).write_text(json.dumps(table, indent=2) + "\n")
    print(target, proof["counts"], flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    main(parser.parse_args().update)
