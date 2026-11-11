"""Synchronous per-case Modal orchestration with durable remote checkpoints."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import modal

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
APP = modal.App("openassembler-dispatch-revalidation")
IMAGE = modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
CHECKPOINTS = modal.Volume.from_name("openassembler-dispatch-evidence-20261110", create_if_missing=True)


@APP.function(image=IMAGE, gpu="A100-40GB", timeout=420, max_containers=1, retries=0,
              volumes={"/checkpoints": CHECKPOINTS})
def execute_case(request: dict, runtime: dict[str, bytes], harness: bytes) -> dict:
    import os
    import subprocess
    import tempfile
    import time
    checkpoint = Path("/checkpoints") / request["campaign"] / request["session"] / request["case"]["id"]
    CHECKPOINTS.reload()
    checkpoint.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    request["session_provenance"]["container_id"] = os.environ.get("MODAL_TASK_ID")
    request["session_provenance"]["function_call_id"] = modal.current_function_call_id()
    try:
        retained_result = checkpoint / "result.json"
        retained_archive = checkpoint / "tuning.artifacts.zip"
        if retained_result.is_file() and retained_archive.is_file():
            cached = json.loads(retained_result.read_text())
            if cached.get("protocol_sha256") == request["protocol_sha256"]:
                return {"status": "ok", "result": cached,
                        "archive": base64.b64encode(retained_archive.read_bytes()).decode(),
                        "stdout": (checkpoint / "stdout.txt").read_text(),
                        "stderr": (checkpoint / "stderr.txt").read_text(),
                        "recovered_remote_checkpoint": True}
        with tempfile.TemporaryDirectory(prefix="oa-shape-worker-") as directory:
            directory = Path(directory)
            for name, data in runtime.items():
                path = directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            (directory / "env/cuda.sh").chmod(0o700)
            path = directory / "campaign.py"
            path.write_bytes(harness)
            inp, out = checkpoint / "request.json", checkpoint / "result.json"
            inp.write_text(json.dumps(request, indent=2))
            completed = subprocess.run([sys.executable, str(path), "--request", str(inp), "--result", str(out)],
                                       cwd=directory, env=dict(os.environ, PYTHONPATH=str(directory)),
                                       capture_output=True, text=True, timeout=380)
            (checkpoint / "stdout.txt").write_text(completed.stdout)
            (checkpoint / "stderr.txt").write_text(completed.stderr)
            if completed.returncode:
                raise RuntimeError(f"worker exit {completed.returncode}: {completed.stderr[-6000:]}")
            result = json.loads(out.read_text())
            result["checkpoint"] = {"volume": "openassembler-dispatch-evidence-20261110",
                                    "path": str(checkpoint.relative_to('/checkpoints')),
                                    "elapsed_seconds": time.monotonic() - started}
            out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            payload = {"status": "ok", "result": result,
                       "archive": base64.b64encode((checkpoint / "tuning.artifacts.zip").read_bytes()).decode(),
                       "stdout": completed.stdout, "stderr": completed.stderr}
    except Exception as exc:
        error = {"status": "failed", "case": request["case"], "campaign": request["campaign"],
                 "session": request["session"], "error": f"{type(exc).__name__}: {exc}",
                 "elapsed_seconds": time.monotonic() - started,
                 "completed_utc": datetime.now(timezone.utc).isoformat()}
        (checkpoint / "failure.json").write_text(json.dumps(error, indent=2) + "\n")
        payload = error
    CHECKPOINTS.commit()
    return payload


@APP.local_entrypoint()
def main(campaign: str = "a100-current-20261110", session: str = "selection",
         case_filter: str = "all", start: int = 0, limit: int = 4, samples: int = 8):
    sys.path.insert(0, str(ROOT))
    from exp.dispatch_tuning.campaign import cases, local_identity, protocol
    from openassembler.transport import runtime_files
    if session not in ("selection", "heldout"):
        raise SystemExit("session must be selection or heldout")
    if samples < 8 or samples > 100:
        raise SystemExit("samples must be between 8 and 100")
    all_rows = cases()
    rows = [row for row in all_rows if case_filter == "all" or case_filter in row["id"]]
    if type(start) is not int or start < 0 or (limit and (type(limit) is not int or limit < 1)):
        raise SystemExit("start must be nonnegative and limit must be a positive integer")
    if not rows[start:start + limit if limit else None]:
        raise SystemExit("no cases selected")
    destination = HERE / "runs" / campaign
    destination.mkdir(parents=True, exist_ok=True)
    frozen = None
    if session == "heldout":
        frozen_path = destination / "frozen-selection.json"
        if not frozen_path.is_file():
            raise SystemExit("freeze selection before starting held-out replay")
        frozen = json.loads(frozen_path.read_text())
    # The persistent session protocol commits all cases before the first
    # sub-batch is measured. Independent Modal apps execute bounded slices of
    # that immutable case inventory to keep auth/input-plane sessions short.
    spec = protocol(campaign, session, all_rows, frozen=frozen)
    spec["samples"] = samples
    protocol_path = destination / f"{session}-protocol.json"
    continuity_path = destination / f"{session}-source-continuity.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text())
        old = existing["source_identity"]
        new = spec["source_identity"]
        if old["compiler_identity"] != new["compiler_identity"]:
            raise SystemExit("GEMM compiler changed since session protocol; choose a new campaign")
        changed_sources = {name: {"protocol_sha256": digest,
                                  "current_sha256": new["source_sha256"].get(name)}
                           for name, digest in old["source_sha256"].items()
                           if new["source_sha256"].get(name) != digest}
        added_sources = sorted(set(new["source_sha256"]) - set(old["source_sha256"]))
        removed_sources = sorted(set(old["source_sha256"]) - set(new["source_sha256"]))
        if (set(changed_sources) - {"openassembler/rowwise.py", "openassembler/rowwise_dispatch.py"}
                or added_sources or removed_sources
                or old["harness_sha256"].get("campaign.py") != new["harness_sha256"].get("campaign.py")):
            raise SystemExit("non-GEMM campaign/runtime inputs changed; choose a new campaign")
        continuity = {"schema": 1, "campaign": campaign, "session": session,
            "original_protocol_path": protocol_path.relative_to(ROOT).as_posix(),
            "original_protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "original_protocol_unchanged": True, "protocol_gemm_compiler_identity": old["compiler_identity"],
            "current_gemm_compiler_identity": new["compiler_identity"], "gemm_identity_unchanged": True,
            "source_drift": changed_sources, "added_sources": added_sources,
            "harness_drift": {name: {"protocol_sha256": digest,
                                      "current_sha256": new["harness_sha256"].get(name)}
                              for name, digest in old["harness_sha256"].items()
                              if new["harness_sha256"].get(name) != digest},
            "removed_sources": removed_sources,
            "scope_review": "Rowwise plan/dispatch code changed after this session freeze; the Modal orchestration wrapper changed to add bounded batching and scoped continuity. Neither is in the GEMM compiler identity or worker campaign.py, and neither changes generated GEMM PTX or candidate/timing methods. All raw records are audited against the current GEMM source, exact re-encoded words, retained SASS and binary artifacts before promotion.",
            "raw_records_rewritten": False}
        if continuity_path.exists():
            previous = json.loads(continuity_path.read_text())
            if previous != continuity:
                raise SystemExit("continuity audit changed; preserve this session and start a new campaign")
        else:
            continuity_path.write_text(json.dumps(continuity, indent=2) + "\n")
        # Retain the originally recorded protocol identity and timestamp.
        spec = existing
    else:
        import uuid
        spec["session_provenance"] = {"id": f"{campaign}-{session}-{uuid.uuid4()}",
                                      "split": session, "modal_app_id": APP.app_id,
                                      "container_id": None}
        protocol_path.write_text(json.dumps(spec, indent=2) + "\n")
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    rows = rows[start:start + limit if limit else None]
    runtime = runtime_files(native=True)
    harness = (HERE / "campaign.py").read_bytes()
    progress_path = destination / f"{session}-progress.json"
    summaries_by_id = {}
    if progress_path.is_file():
        for saved in json.loads(progress_path.read_text()):
            summaries_by_id[saved["id"]] = saved
    for index, case in enumerate(rows):
        if local_identity()["compiler_identity"] != spec["source_identity"]["compiler_identity"]:
            raise SystemExit("compiler changed during campaign; stop instead of mislabelling results")
        local_case = destination / session / case["id"]
        local_case.mkdir(parents=True, exist_ok=True)
        target = local_case / "result.json"
        if target.is_file():
            result = json.loads(target.read_text())
            if result.get("compiler_identity") == spec["source_identity"]["compiler_identity"]:
                print(f"resume {index + 1}/{len(rows)} {case['id']} {result['selected']}", flush=True)
                summaries_by_id[case["id"]] = {"id": case["id"], "status": "ok",
                    "selected": result["selected"],
                    "result_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
                continue
        request = {"campaign": campaign, "session": session, "case": case,
                   "session_provenance": {**spec["session_provenance"], "modal_app_id": APP.app_id},
                   "rounds": 3, "samples": samples, "protocol_sha256": protocol_hash}
        print(f"start {index + 1}/{len(rows)} {case['id']}", flush=True)
        reply = execute_case.remote(request, runtime, harness)
        if reply.get("status") != "ok":
            (local_case / "failure.json").write_text(json.dumps(reply, indent=2) + "\n")
            print(f"failed {case['id']} {reply.get('error')}", flush=True)
            summaries_by_id[case["id"]] = {"id": case["id"], "status": "failed",
                                               "error": reply.get("error")}
            progress_path.write_text(json.dumps(list(summaries_by_id.values()), indent=2) + "\n")
            continue
        result = reply["result"]
        archive = base64.b64decode(reply["archive"], validate=True)
        if hashlib.sha256(archive).hexdigest() != result["artifact_archive"]["sha256"]:
            raise SystemExit("returned archive hash mismatch")
        archive_path = local_case / result["artifact_archive"]["path"]
        archive_path.write_bytes(archive)
        (local_case / "stdout.txt").write_text(reply["stdout"])
        (local_case / "stderr.txt").write_text(reply["stderr"])
        target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        timing = [{key: round(value * 1000, 3) for key, value in row["median_ms"].items()}
                  for row in result["timings"]]
        summary = {"id": case["id"], "status": "ok", "selected": result["selected"],
                   "native_us": timing, "device": result["device"],
                   "result_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        summaries_by_id[case["id"]] = summary
        progress_path.write_text(json.dumps(list(summaries_by_id.values()), indent=2) + "\n")
        print(json.dumps({"completed": index + 1, "total": len(rows), **summary}), flush=True)
    (destination / f"{session}-progress.json").write_text(json.dumps(list(summaries_by_id.values()), indent=2) + "\n")
    print(f"batch complete: {len(summaries_by_id)} cumulative cases, "
          f"{sum(x['status']=='ok' for x in summaries_by_id.values())} successful", flush=True)
