"""Bounded real-A100 exact indexing validation; emits a new immutable run record."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
app = modal.App("openassembler-native-indexing")
image = (modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
         .add_local_python_source("openassembler"))


@app.function(image=image, gpu="A100-40GB", timeout=600, max_containers=1, retries=0)
def execute(payload):
    import base64
    import ctypes as C
    import dataclasses
    import hashlib
    import struct
    import time
    import random
    from openassembler.gpu.cuda_driver import Driver
    driver = Driver()
    driver.open()
    results = []
    started = time.monotonic()
    try:
        for product in payload["products"]:
            binary = base64.b64decode(product["cubin"])
            assert hashlib.sha256(binary).hexdigest() == product["cubin_sha256"]
            module = driver.load(binary)
            for name in ("divide_indices", "quotient_indices"):
                fn = driver.function(module, name)
                for rows, cols in ((2, 2), (18, 22), (256, 512), (1024, 512),
                                   (513, 513), (1009, 127), (1024, 1009)):
                    count = rows * cols
                    reference = struct.pack(f"<{count}I", *(i % cols if name == "divide_indices"
                                                           else i // cols for i in range(count)))
                    size = 4 * count
                    guard = bytes([0xa5]) * 256
                    ptr = driver.alloc(size + 512)
                    for repeat in range(3):
                        driver.htod(ptr, guard + bytes([[0xff, 0x5a, 0x3c][repeat]]) * size + guard)
                        driver.launch(fn, ((count + 127) // 128, 1, 1), (128, 1, 1),
                                      [C.c_uint64(ptr + 256), C.c_uint32(rows), C.c_uint32(cols)])
                        actual = driver.dtoh(ptr, size + 512)
                        output = actual[256:-256]
                        bad = []
                        mismatches = 0
                        if output != reference:
                            for i, (a, b) in enumerate(zip(struct.iter_unpack("<I", output),
                                                          struct.iter_unpack("<I", reference))):
                                if a != b:
                                    mismatches += 1
                                    if len(bad) < 8:
                                        bad.append([i, a[0], b[0]])
                        canaries = actual[:256] == guard and actual[-256:] == guard
                        record = {k: v for k, v in product.items() if k != "cubin"}
                        record.update(kernel=name, shape=[rows, cols], repeat=repeat,
                                      checked=count, mismatches=mismatches, bad=bad,
                                      canaries=canaries, passed=mismatches == 0 and canaries)
                        results.append(record)
                    driver.free(ptr)
            for name in ("runtime_u32", "runtime_s32"):
                if name not in payload.get("kernels", []):
                    continue
                fn = driver.function(module, name)
                signed = name == "runtime_s32"
                for count in (65539, 524289):
                    rng = random.Random(20261112)
                    divisors = [1, 2, 3, 7, 17, 22, 31, 127, 255, 256, 511, 512, 513,
                                1009, 65535, 65536, 65537, 0x7fffffff]
                    if not signed:
                        divisors += [0x80000000, 0xfffffffe, 0xffffffff]
                    av, bv, qv, rv = [], [], [], []
                    for i in range(count):
                        d = divisors[(i // 11) % len(divisors)]
                        edges = [0, 1, d - 1, d, d + 1, 0x7ffffffe, 0x7fffffff,
                                 0x80000000, 0xfffffffe, 0xffffffff,
                                 rng.getrandbits(32)]
                        x = edges[i % len(edges)] & 0xffffffff
                        if signed:
                            x = x if x < 0x80000000 else x - 0x100000000
                            if (i // (11 * len(divisors))) & 1:
                                d = -d
                            # PTX signed quotient overflow is outside this test contract.
                            if x == -0x80000000 and d == -1:
                                d = -3
                            q = abs(x) // abs(d)
                            if (x < 0) != (d < 0):
                                q = -q
                            r = x - q * d
                        else:
                            q, r = divmod(x, d)
                        av.append(x & 0xffffffff); bv.append(d & 0xffffffff)
                        qv.append(q & 0xffffffff); rv.append(r & 0xffffffff)
                    inputs = [struct.pack(f"<{count}I", *v) for v in (av, bv)]
                    references = [struct.pack(f"<{count}I", *v) for v in (qv, rv)]
                    size = count * 4
                    guard = bytes([0xa5]) * 256
                    pointers = [driver.alloc(size + 512) for _ in range(4)]
                    for p, data in zip(pointers, inputs):
                        driver.htod(p, guard + data + guard)
                    for repeat in range(3):
                        for p in pointers[2:]:
                            driver.htod(p, guard + bytes([[0xff, 0x5a, 0x3c][repeat]]) * size + guard)
                        driver.launch(fn, ((count + 127) // 128, 1, 1), (128, 1, 1),
                                      [*[C.c_uint64(p + 256) for p in pointers], C.c_uint32(count)])
                        checks = []
                        for p, reference in zip(pointers[2:], references):
                            actual = driver.dtoh(p, size + 512)
                            output = actual[256:-256]
                            bad = []
                            mismatches = 0
                            if output != reference:
                                for i, (a, b) in enumerate(zip(struct.iter_unpack("<I", output),
                                                              struct.iter_unpack("<I", reference))):
                                    if a != b:
                                        mismatches += 1
                                        if len(bad) < 8:
                                            bad.append([i, a[0], b[0]])
                            checks.append({"mismatches": mismatches, "bad": bad,
                                           "canaries": actual[:256] == guard and actual[-256:] == guard})
                        immutable = all(driver.dtoh(p, size + 512) == guard + data + guard
                                        for p, data in zip(pointers, inputs))
                        record = {k: v for k, v in product.items() if k != "cubin"}
                        record.update(kernel=name, count=count, repeat=repeat, checked=count * 2,
                                      outputs=checks, inputs_immutable=immutable,
                                      passed=immutable and all(c["mismatches"] == 0 and c["canaries"]
                                                               for c in checks))
                        results.append(record)
                    for p in pointers:
                        driver.free(p)
            driver.unload(module)
            print(product["level"], product["method"], "failures",
                  sum(not r["passed"] for r in results
                      if r["level"] == product["level"] and r["method"] == product["method"]), flush=True)
    finally:
        info = dataclasses.asdict(driver.info())
        driver.close()
    return {"schema": 1, "tag": payload["tag"], "device": info,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "harness_sha256": payload.get("harness_sha256"),
            "compiler_sha256": payload["compiler_sha256"],
            "source_sha256": payload["source_sha256"],
            "elapsed_seconds": time.monotonic() - started, "results": results}


@app.local_entrypoint()
def main(payload: str):
    import hashlib
    value = json.loads(Path(payload).read_text())
    value["harness_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = execute.remote(value)
    path = HERE / "runs" / (value["tag"] + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(path)
