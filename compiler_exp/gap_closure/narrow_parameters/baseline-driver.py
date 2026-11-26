"""Load and run cubins with the CUDA *driver* API via ctypes on libcuda.so.

Stdlib only, so it runs on any Linux host with an NVIDIA driver: Modal containers
(libcuda is mounted, no toolkit needed), an SSH box, or a local workstation.

Robustness:
  * every driver call is checked; error names come from cuGetErrorName
  * a kernel fault poisons the CUDA context, so callers that batch many cubins
    should run each in its own process (see openassembler.gpu.batch)
  * memory beyond the probe slot is guarded: we allocate padding and fill it
    with a canary to detect out-of-bounds writes
"""

from __future__ import annotations

import ctypes as C
import struct
import time
from dataclasses import dataclass


class CudaError(RuntimeError):
    def __init__(self, what: str, code: int, name: str):
        super().__init__(f"{what}: {name} ({code})")
        self.what, self.code, self.name = what, code, name


@dataclass
class DeviceInfo:
    name: str
    cc: str
    driver: int
    sm_count: int
    clock_khz: int
    mem_bytes: int


class Driver:
    def __init__(self, lib: str = "libcuda.so.1"):
        self.cu = cu = C.CDLL(lib)
        u64, vp, sz = C.c_uint64, C.c_void_p, C.c_size_t
        cu.cuMemcpyHtoD_v2.argtypes = [u64, vp, sz]
        cu.cuMemcpyDtoH_v2.argtypes = [vp, u64, sz]
        cu.cuMemsetD8_v2.argtypes = [u64, C.c_ubyte, sz]
        cu.cuMemFree_v2.argtypes = [u64]
        cu.cuMemAlloc_v2.argtypes = [C.POINTER(u64), sz]
        cu.cuLaunchKernel.argtypes = [vp] + [C.c_uint] * 7 + [vp, C.POINTER(vp), vp]
        cu.cuModuleLoadDataEx.argtypes = [C.POINTER(vp), vp, C.c_uint, vp, vp]
        cu.cuModuleGetFunction.argtypes = [C.POINTER(vp), vp, C.c_char_p]
        cu.cuFuncSetAttribute.argtypes = [vp, C.c_int, C.c_int]
        cu.cuEventElapsedTime.argtypes = [C.POINTER(C.c_float), vp, vp]
        cu.cuEventRecord.argtypes = [vp, vp]
        cu.cuEventSynchronize.argtypes = [vp]
        self.ck(cu.cuInit(0), "cuInit")
        self.dev = C.c_int()
        self.ck(cu.cuDeviceGet(C.byref(self.dev), 0), "cuDeviceGet")
        self.ctx = None

    def errname(self, r: int) -> str:
        s = C.c_char_p()
        self.cu.cuGetErrorName(r, C.byref(s))
        return s.value.decode() if s.value else "?"

    def ck(self, r: int, what: str) -> None:
        if r != 0:
            raise CudaError(what, r, self.errname(r))

    def info(self) -> DeviceInfo:
        cu, d = self.cu, self.dev
        name = C.create_string_buffer(256)
        cu.cuDeviceGetName(name, 256, d)
        def attr(a):
            v = C.c_int()
            cu.cuDeviceGetAttribute(C.byref(v), a, d)
            return v.value
        drv, mem = C.c_int(), C.c_size_t()
        cu.cuDriverGetVersion(C.byref(drv))
        cu.cuDeviceTotalMem_v2(C.byref(mem), d)
        # 75/76 = CC major/minor, 16 = SM count, 13 = clock rate (kHz)
        return DeviceInfo(name.value.decode(), f"{attr(75)}.{attr(76)}", drv.value, attr(16), attr(13), mem.value)

    # ---- context / memory -------------------------------------------------------

    def open(self) -> None:
        ctx = C.c_void_p()
        self.ck(self.cu.cuCtxCreate_v2(C.byref(ctx), 0, self.dev), "cuCtxCreate")
        self.ctx = ctx

    def close(self) -> None:
        if self.ctx is not None:
            self.cu.cuCtxDestroy_v2(self.ctx)
            self.ctx = None

    def alloc(self, n: int) -> int:
        p = C.c_uint64()
        self.ck(self.cu.cuMemAlloc_v2(C.byref(p), n), "cuMemAlloc")
        return p.value

    def free(self, p: int) -> None:
        self.cu.cuMemFree_v2(p)

    def htod(self, p: int, data: bytes) -> None:
        self.ck(self.cu.cuMemcpyHtoD_v2(p, data, len(data)), "HtoD")

    def dtoh(self, p: int, n: int) -> bytes:
        buf = C.create_string_buffer(n)
        self.ck(self.cu.cuMemcpyDtoH_v2(buf, p, n), "DtoH")
        return buf.raw

    def memset(self, p: int, byte: int, n: int) -> None:
        self.ck(self.cu.cuMemsetD8_v2(p, byte, n), "memset")

    # ---- modules / launch -----------------------------------------------------------

    def load(self, cubin: bytes) -> C.c_void_p:
        mod = C.c_void_p()
        img = C.create_string_buffer(cubin, len(cubin))
        self.ck(self.cu.cuModuleLoadDataEx(C.byref(mod), img, 0, None, None), "cuModuleLoadData")
        return mod

    def unload(self, mod) -> None:
        self.cu.cuModuleUnload(mod)

    def function(self, mod, name: str):
        fn = C.c_void_p()
        self.ck(self.cu.cuModuleGetFunction(C.byref(fn), mod, name.encode()), f"cuModuleGetFunction({name})")
        return fn

    def func_attrs(self, fn) -> dict:
        """Driver-reported kernel resources (CU_FUNC_ATTRIBUTE_*): checks our .nv.info."""
        out = {}
        for k, a in (("max_threads", 0), ("shared_bytes", 1), ("const_bytes", 2), ("local_bytes", 3),
                     ("num_regs", 4), ("ptx_version", 5), ("binary_version", 6)):
            v = C.c_int()
            r = self.cu.cuFuncGetAttribute(C.byref(v), a, fn)
            out[k] = v.value if r == 0 else None
        return out

    def launch(self, fn, grid, block, args: list, smem: int = 0, time_it: bool = False) -> float | None:
        """args: list of ctypes scalars. Returns kernel time in ms if time_it."""
        params = (C.c_void_p * len(args))(*[C.cast(C.pointer(a), C.c_void_p) for a in args])
        if smem > 49152:
            self.ck(self.cu.cuFuncSetAttribute(fn, 8, smem), "opt-in dynamic shared memory")
        ev0 = ev1 = None
        if time_it:
            ev0, ev1 = C.c_void_p(), C.c_void_p()
            self.ck(self.cu.cuEventCreate(C.byref(ev0), 0), "cuEventCreate")
            self.ck(self.cu.cuEventCreate(C.byref(ev1), 0), "cuEventCreate")
            self.ck(self.cu.cuEventRecord(ev0, None), "cuEventRecord")
        self.ck(self.cu.cuLaunchKernel(fn, *grid, *block, smem, None, params, None), "cuLaunchKernel")
        if time_it:
            self.ck(self.cu.cuEventRecord(ev1, None), "cuEventRecord")
        self.ck(self.cu.cuCtxSynchronize(), "kernel execution (cuCtxSynchronize)")
        if time_it:
            ms = C.c_float()
            self.ck(self.cu.cuEventElapsedTime(C.byref(ms), ev0, ev1), "cuEventElapsedTime")
            self.cu.cuEventDestroy_v2(ev0)
            self.cu.cuEventDestroy_v2(ev1)
            return ms.value
        return None


# ---- probe ABI (tools/ptx_probes.py) --------------------------------------------------

THREADS = 32
SLOT = 16
PAD = 4096          # probes may touch [in+16]/[in+64]; canary-filled to catch OOB writes
CANARY = 0xA5
SCRATCH = (0x800, 0x800 + THREADS * SLOT)  # per-thread store-probe scratch (out + 0x800 + 16*tid)


def run_probe_job(drv: Driver, job: dict) -> dict:
    """job: {kname, cubin, trials: [(inputs[32][4], k, kf)]} -> {out, attrs, oob}.
    Caller owns the context."""
    size = THREADS * SLOT + PAD
    din, dout = drv.alloc(size), drv.alloc(size)
    mod = None
    try:
        mod = drv.load(job["cubin"])
        fn = drv.function(mod, job["kname"])
        attrs = drv.func_attrs(fn)
        outs, oob = [], []
        for inputs, k, kf in job["trials"]:
            drv.memset(din, 0, size)
            # slots + scratch start at 0 (the emulator's memory default); the rest is canary
            drv.memset(dout, CANARY, size)
            drv.memset(dout, 0, THREADS * SLOT)
            drv.memset(dout + SCRATCH[0], 0, SCRATCH[1] - SCRATCH[0])
            data = b"".join(struct.pack("<4I", *(w & 0xFFFFFFFF for w in row)) for row in inputs)
            drv.htod(din, data)
            args = [C.c_uint64(dout), C.c_uint64(din), C.c_uint32(k & 0xFFFFFFFF), C.c_uint32(kf & 0xFFFFFFFF)]
            drv.launch(fn, (1, 1, 1), (THREADS, 1, 1), args)
            raw = drv.dtoh(dout, size)
            outs.append([list(struct.unpack_from("<2I", raw, SLOT * t)) for t in range(THREADS)])
            # bytes 8..15 of each slot are probe scratch; beyond the last slot must be untouched
            oob.append(sum(b != CANARY for k, b in enumerate(raw[THREADS * SLOT:], THREADS * SLOT)
                           if not SCRATCH[0] <= k < SCRATCH[1]))
        return {"out": outs, "attrs": attrs, "oob_bytes": oob}
    finally:
        if mod is not None:
            drv.unload(mod)
        drv.free(din)
        drv.free(dout)


def now() -> float:
    return time.monotonic()


def time_probe_job(drv: Driver, job: dict) -> dict:
    """Timing mode: launch the probe kernel with `grid` blocks x 32 threads, `reps`
    times after `warmup` launches; report per-launch GPU time (cuEvent) statistics.
    Inputs come from trial 0. Correctness is checked separately (run_probe_job), since
    every block writes the same per-thread output slots under the probe ABI."""
    grid = job.get("grid", 4096)
    reps = job.get("reps", 50)
    warmup = job.get("warmup", 5)
    size = THREADS * SLOT + PAD
    din, dout = drv.alloc(size), drv.alloc(size)
    mod = None
    try:
        mod = drv.load(job["cubin"])
        fn = drv.function(mod, job["kname"])
        inputs, k, kf = job["trials"][0]
        data = b"".join(struct.pack("<4I", *(w & 0xFFFFFFFF for w in row)) for row in inputs)
        drv.memset(din, 0, size)
        drv.htod(din, data)
        args = [C.c_uint64(dout), C.c_uint64(din), C.c_uint32(k & 0xFFFFFFFF), C.c_uint32(kf & 0xFFFFFFFF)]
        for _ in range(warmup):
            drv.launch(fn, (grid, 1, 1), (THREADS, 1, 1), args)
        ts = sorted(drv.launch(fn, (grid, 1, 1), (THREADS, 1, 1), args, time_it=True) for _ in range(reps))
        return {"time_ms": {"median": ts[len(ts) // 2], "min": ts[0], "p90": ts[int(len(ts) * 0.9)]},
                "grid": grid, "reps": reps, "attrs": drv.func_attrs(fn)}
    finally:
        if mod is not None:
            drv.unload(mod)
        drv.free(din)
        drv.free(dout)


# ---- plans: general multi-kernel programs (fast matmul, GEMM) --------------------------

_RAND = {}


def rand_f32(seed: int, count: int) -> bytes:
    """Deterministic uniform(-1, 1) f32 data, identical on the host and in the container
    (random.Random is specified to be reproducible across platforms). Cached."""
    import random

    import array

    key = (seed, count)
    if key not in _RAND:
        r = random.Random(seed)
        a = array.array("f", (r.uniform(-1, 1) for _ in range(count)))   # 4 bytes/value, no tuple
        _RAND.clear()                    # keep at most the matrices of one size
        _RAND[key] = a.tobytes()
    return _RAND[key]


def rand_tensor16(seed: int, count: int, dtype: str) -> bytes:
    """Exactly representable small integers for bitwise tensor-kernel comparisons.

    These are smoke inputs, not a replacement for floating-point GEMM validation.
    Generate them at the worker to avoid uploading entire matrices in job files.
    """
    import array
    import random
    import struct
    import sys
    if dtype not in ("f16", "bf16") or type(count) is not int or count < 0:
        raise ValueError("tensor initialization requires f16/bf16 and a nonnegative count")
    values = range(-3, 4)
    table = ([struct.unpack("<H", struct.pack("<e", v))[0] for v in values] if dtype == "f16"
             else [struct.unpack("<I", struct.pack("<f", v))[0] >> 16 for v in values])
    rng = random.Random(seed)
    data = array.array("H", (table[rng.randrange(7)] for _ in range(count)))
    if sys.byteorder != "little":
        data.byteswap()
    return data.tobytes()


def run_plan_job(drv: Driver, job: dict) -> dict:
    """job = {
        "modules": {name: cubin bytes},
        "buffers": {name: size_bytes},
        "init":    {buffer: bytes}               host data copied in once
        "init_byte": {buffer: 0..255}            optional device byte fill, before host data
        "init_rand": {buffer: (seed, count)}     or generated with rand_f32 (no upload)
        "init_tensor16": {buffer: (seed, count, dtype)} small f16/bf16 integers
        "steps":   [(module, kernel, grid, block, smem, [arg, ...]), ...]
                   arg: ("buf", name, byte_offset) | ("u32", v) | ("f32bits", v) | ("u64", v)
        "outputs": {buffer: [(offset, nbytes)]}  read back after one run
        "reps", "warmup": timing of the whole step list (cuEvents around it)
    }
    Stdlib only. Returns {"out": {buf: bytes}, "time_ms": {...}, "attrs": {...}}."""
    mods, fns, bufs = {}, {}, {}
    try:
        for name, cubin in job["modules"].items():
            mods[name] = drv.load(cubin)
        for name, size in job["buffers"].items():
            bufs[name] = drv.alloc(max(size, 16))
            drv.memset(bufs[name], 0, max(size, 16))
        for name, value in job.get("init_byte", {}).items():
            if name not in bufs or type(value) is not int or not 0 <= value <= 255:
                raise ValueError("byte initialization requires a known buffer and a byte in [0, 255]")
            drv.memset(bufs[name], value, max(job["buffers"][name], 16))
        for name, data in job.get("init", {}).items():
            drv.htod(bufs[name], data)
        for name, (seed, count) in job.get("init_rand", {}).items():
            drv.htod(bufs[name], rand_f32(seed, count))
        for name, (seed, count, dtype) in job.get("init_tensor16", {}).items():
            if 2 * count != job["buffers"][name]:
                raise ValueError("tensor initialization size does not match its buffer")
            drv.htod(bufs[name], rand_tensor16(seed, count, dtype))
        launches = []
        attrs = {}
        shared_limits = {}
        blas = None
        for mname, kname, grid, block, smem, args in job["steps"]:
            if mname == "@memset":
                launches.append(("@memset", bufs[args[0][1]] + args[0][2], args[1][1]))
                continue
            if mname == "@cublas":
                # args: A, B, C bufs, then M N K lda ldb ldc
                if blas is None:
                    blas = Cublas()
                ptrs = [bufs[a[1]] + a[2] for a in args[:3]]
                ints = [a[1] for a in args[3:]]
                launches.append(("@cublas", blas, ptrs, ints))
                continue
            key = (mname, kname)
            if key not in fns:
                fns[key] = drv.function(mods[mname], kname)
                attrs[f"{mname}:{kname}"] = drv.func_attrs(fns[key])
            if smem > max(49152, shared_limits.get(key, 0)):
                drv.ck(drv.cu.cuFuncSetAttribute(fns[key], 8, smem), "opt-in dynamic shared memory")
                shared_limits[key] = smem
            cargs = []
            for a in args:
                if a[0] == "buf":
                    cargs.append(C.c_uint64(bufs[a[1]] + a[2]))
                elif a[0] == "u32":
                    cargs.append(C.c_uint32(a[1] & 0xFFFFFFFF))
                elif a[0] == "f32bits":
                    cargs.append(C.c_uint32(a[1] & 0xFFFFFFFF))
                elif a[0] == "u64":
                    cargs.append(C.c_uint64(a[1]))
                else:
                    raise ValueError(a)
            params = (C.c_void_p * len(cargs))(*[C.cast(C.pointer(x), C.c_void_p) for x in cargs])
            launches.append((fns[key], tuple(grid), tuple(block), smem, params, cargs))

        def run_all():
            for L in launches:
                if L[0] == "@memset":
                    drv.ck(drv.cu.cuMemsetD8_v2(L[1], 0, L[2]), "memset")
                    continue
                if L[0] == "@cublas":
                    L[1].sgemm_rowmajor(*L[2], *L[3])
                    continue
                fn, g, b, smem, params, _keep = L
                drv.ck(drv.cu.cuLaunchKernel(fn, *g, *b, smem, None, params, None), "cuLaunchKernel")

        run_all()
        drv.ck(drv.cu.cuCtxSynchronize(), "kernel execution (cuCtxSynchronize)")
        # outputs: {buffer: [(byte offset, nbytes), ...]} -> {buffer: [bytes, ...]}
        out = {b: [drv.dtoh(bufs[b] + off, n) for off, n in spans] for b, spans in job.get("outputs", {}).items()}
        times = []
        reps = job.get("reps", 0)
        if reps:
            for _ in range(job.get("warmup", 3)):
                run_all()
            drv.ck(drv.cu.cuCtxSynchronize(), "kernel execution (cuCtxSynchronize)")
            ev0, ev1 = C.c_void_p(), C.c_void_p()
            drv.ck(drv.cu.cuEventCreate(C.byref(ev0), 0), "cuEventCreate")
            drv.ck(drv.cu.cuEventCreate(C.byref(ev1), 0), "cuEventCreate")
            for _ in range(reps):
                drv.ck(drv.cu.cuEventRecord(ev0, None), "cuEventRecord")
                run_all()
                drv.ck(drv.cu.cuEventRecord(ev1, None), "cuEventRecord")
                drv.ck(drv.cu.cuEventSynchronize(ev1), "kernel execution (cuEventSynchronize)")
                ms = C.c_float()
                drv.ck(drv.cu.cuEventElapsedTime(C.byref(ms), ev0, ev1), "cuEventElapsedTime")
                times.append(ms.value)
            drv.cu.cuEventDestroy_v2(ev0)
            drv.cu.cuEventDestroy_v2(ev1)
        res = {"out": out, "attrs": attrs}
        if times:
            ts = sorted(times)
            res["time_ms"] = {"median": ts[len(ts) // 2], "min": ts[0], "p90": ts[int(len(ts) * 0.9)],
                              "n": len(ts)}
        return res
    finally:
        for m in mods.values():
            drv.unload(m)
        for p in bufs.values():
            drv.free(p)


class Cublas:
    """Minimal cuBLAS (libcublas.so.*, e.g. from pip nvidia-cublas-cu12) for a reference
    SGEMM in plans. Row-major C = A B is computed as column-major C^T = B^T A^T."""

    def __init__(self):
        import glob
        import os

        cands = []
        for p in [os.environ.get("CUBLAS_LIB", "")] + glob.glob("/usr/local/lib/python3*/site-packages/nvidia/cublas/lib/libcublas.so*"):
            if p:
                cands.append(p)
        cands += ["libcublas.so.12", "libcublas.so"]
        last = None
        for p in cands:
            try:
                self.lib = C.CDLL(p)
                break
            except OSError as e:
                last = e
        else:
            raise OSError(f"libcublas not found: {last}")
        self.h = C.c_void_p()
        r = self.lib.cublasCreate_v2(C.byref(self.h))
        if r:
            raise RuntimeError(f"cublasCreate: {r}")
        self.lib.cublasSgemm_v2.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
                                            C.POINTER(C.c_float), C.c_uint64, C.c_int, C.c_uint64, C.c_int,
                                            C.POINTER(C.c_float), C.c_uint64, C.c_int]
        self.one, self.zero = C.c_float(1.0), C.c_float(0.0)

    def sgemm_rowmajor(self, a, b, c, M, N, K, lda, ldb, ldc):
        r = self.lib.cublasSgemm_v2(self.h, 0, 0, N, M, K, C.byref(self.one), b, ldb, a, lda,
                                    C.byref(self.zero), c, ldc)
        if r:
            raise RuntimeError(f"cublasSgemm: {r}")
