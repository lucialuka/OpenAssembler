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
         case_filter: str = "all", limit: int = 0, samples: int = 8):
    sys.path.insert(0, str(ROOT))
    from exp.dispatch_tuning.campaign import cases, local_identity, protocol
    from openassembler.transport import runtime_files
    if session not in ("selection", "heldout"):
        raise SystemExit("session must be selection or heldout")
    if samples < 8 or samples > 100:
        raise SystemExit("samples must be between 8 and 100")
    rows = [row for row in cases() if case_filter == "all" or case_filter in row["id"]]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise SystemExit("no cases selected")
    destination = HERE / "runs" / campaign
    destination.mkdir(parents=True, exist_ok=True)
    frozen = None
    if session == "heldout":
        frozen_path = destination / "frozen-selection.json"
        if not frozen_path.is_file():
            raise SystemExit("freeze selection before starting held-out replay")
        frozen = json.loads(frozen_path.read_text())
    spec = protocol(campaign, session, rows, frozen=frozen)
    spec["samples"] = samples
    protocol_path = destination / f"{session}-protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text())
        if existing["source_identity"] != spec["source_identity"]:
            raise SystemExit("source changed since session protocol; choose a new campaign")
        # Retain the originally recorded protocol identity and timestamp.
        spec = existing
    else:
        import uuid
        spec["session_provenance"] = {"id": f"{campaign}-{session}-{uuid.uuid4()}",
                                      "split": session, "modal_app_id": APP.app_id,
                                      "container_id": None}
        protocol_path.write_text(json.dumps(spec, indent=2) + "\n")
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    runtime = runtime_files(native=True)
    harness = (HERE / "campaign.py").read_bytes()
    summaries = []
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
                summaries.append({"id": case["id"], "status": "ok", "selected": result["selected"]})
                continue
        request = {"campaign": campaign, "session": session, "case": case,
                   "session_provenance": {**spec["session_provenance"], "modal_app_id": APP.app_id},
                   "rounds": 3, "samples": samples, "protocol_sha256": protocol_hash}
        print(f"start {index + 1}/{len(rows)} {case['id']}", flush=True)
        reply = execute_case.remote(request, runtime, harness)
        if reply.get("status") != "ok":
            (local_case / "failure.json").write_text(json.dumps(reply, indent=2) + "\n")
            print(f"failed {case['id']} {reply.get('error')}", flush=True)
            summaries.append({"id": case["id"], "status": "failed", "error": reply.get("error")})
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
        summaries.append(summary)
        (destination / f"{session}-progress.json").write_text(json.dumps(summaries, indent=2) + "\n")
        print(json.dumps({"completed": index + 1, "total": len(rows), **summary}), flush=True)
    (destination / f"{session}-progress.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"session complete: {len(summaries)} cases, {sum(x['status']=='ok' for x in summaries)} successful", flush=True)
