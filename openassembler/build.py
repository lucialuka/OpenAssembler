"""`openassembler compile` (and the compile half of `cnr`): source file or source
tree -> output/ (layout and manifest schema: openassembler.project).

Per source file:
  1. frontend: .ptx as-is, .py via openassembler.pyfront, .cu/.cpp via nvcc --ptx
     (openassembler.frontend.to_ptx_module) -> ONE PTX module with all its kernels
  2. our backend (openassembler.backend.ra) generates every kernel's SASS; --verify
     checks each on the SASS emulator against the PTX reference interpreter
  3. ptxas -O3 builds the same PTX (restricted to the kernels that will be in the
     cubin) once: that cubin is both the ELF container we splice our code into and
     the reference `openassembler run` compares against (.build/reference/)
  4. each kernel's .text is spliced into that one container; a kernel our backend
     can't compile is an error, or with --fallback ptxas keeps ptxas's code -- said
     so in the manifest, the .sass header and the summary, never silently
  5. nvdisasm lists the final cubin -> sass/<name>.sass (and re-reads every word we
     encoded as a sanity check)

Multi-kernel cubins: container.cubin.splice_text, applied once per kernel, rewrites
EVERY EIATTR_REGCOUNT record in .nv.info with that kernel's count (it was written
for one-kernel probes). The last splice would win for all kernels, so a kernel
needing more registers than the last one would be launched with too few and fault
(CUDA_ERROR_ILLEGAL_INSTRUCTION, see required_regcount's evidence). splice_kernels
below re-fixes each REGCOUNT record by its function's symbol index after splicing.
"""

from __future__ import annotations

import re
import struct
import subprocess
import sys
import concurrent.futures
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from openassembler import __version__, oracle
from openassembler import project as P

ROOT = oracle.ROOT
TOOLS = ROOT / "tools"
OPT_LEVELS = ("safe", "mined", "sched", "opt")
_PROBE_PARAMS = (("u64", "out"), ("u64", "in"), ("u32", "k"), ("f32", "kf"))


@dataclass
class Options:
    arch: str = "sm_80"
    opt_level: str = "opt"
    kernel: str | None = None
    verify: bool = False
    trials: int = 6
    fallback: str | None = None          # None | "ptxas"
    no_sass: bool = False
    include_dirs: list[Path] = field(default_factory=list)
    nvcc_flags: list[str] = field(default_factory=list)
    dump_ptx: bool = False


def _err(msg: str) -> None:
    sys.stdout.flush()               # keep stdout/stderr lines in order on a terminal
    print(f"error: {msg}", file=sys.stderr)


def _show(p: Path) -> str:
    """A path as the user would type it: relative to the cwd when under it."""
    try:
        return str(Path(p).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(p)


# ---- backend ------------------------------------------------------------------

_ENCODERS: dict = {}


def encoder(arch: str):
    """One Encoder per arch per process (loading its templates is the slow part).
    Raises EncodeError("no SASS encoder data for ...") for an arch we have no data for."""
    from openassembler.isa.encode import Encoder

    if arch not in _ENCODERS:
        _ENCODERS[arch] = Encoder(arch)
    return _ENCODERS[arch]

def generate(kernel, arch: str, opt_level: str, *, tensor_config=None):
    """One PTX kernel -> (code, labels, words, exits, nregs) with our backend. Raises
    ra.NotSupported / EncodeError for what it can't do (never guesses)."""
    from openassembler.backend import ra
    from openassembler.backend.compile import Out, schedule
    from openassembler.backend.sched import list_schedule
    from openassembler.backend.unroll import unroll_loops
    from openassembler.container.cubin import required_regcount
    from openassembler.backend import tensor

    if tensor.needs_tensor(kernel):
        return tensor.generate(kernel, arch, opt_level, config=tensor_config)

    if opt_level == "safe":
        code, st = ra.compile_kernel(kernel)
        labels = st["labels"]
    else:
        # Mirrors tools/bench_build.py:_reschedule_k, the reference pipeline for
        # the mined/sched/opt variants (list_schedule runs on the raw lowered
        # body, before it's split into labels + Out tuples for schedule()).
        k = unroll_loops(kernel) if opt_level == "opt" else kernel
        s = ra.Selector(k)
        ops = ra.coalesce(ra.dce(s.run()))
        phys = ra.allocate(ops, s.r1)
        lowered = ra.lower(ops, phys)
        if opt_level in ("sched", "opt"):
            lowered = list_schedule(lowered)
        labels, body = {}, []
        for t, vl, r, w in lowered:
            if t.startswith("<label "):
                labels[t[7:-1]] = len(body)
            else:
                body.append((t, vl, r, w))
        code = schedule([Out(t, vl, r, w) for t, vl, r, w in body], labels=labels, model="mined")

    enc = encoder(arch)
    words = ra.encode_code(enc, code, labels)
    nop = enc.encode("NOP")
    words.append(enc.encode_branch("BRA `(.L_x_0)", 16 * len(words), 16 * len(words)))
    while len(words) % 8:
        words.append(nop)
    exits = [16 * i for i, (t, _ctl) in enumerate(code) if t.endswith("EXIT")]
    nregs = required_regcount([t for t, _ctl in code])
    return code, labels, words, exits, nregs


def count_insns(code) -> int:
    return sum(1 for t, _ in code if not t.startswith("NOP") and not t.startswith("BRA `(.L_x_0)"))


# ---- verify -------------------------------------------------------------------

def verify_kernel(module, kname: str, code, labels, ptx_text: str, trials: int) -> tuple[str, str]:
    """("ok" | "failed" | "not-verified", message). Picks the original fixed
    probe-ABI harness for probe-shaped kernels (out/in/k/kf -- the corpus this
    repo's own tests use), or the generic one (openassembler.verify) for anything
    else, which covers pyfront/cppfront kernels and hand-written PTX with any
    signature."""
    kernel = module.kernels[kname]
    if tuple(kernel.params) == _PROBE_PARAMS:
        import random

        from openassembler.emu.probe import run_probe

        sys.path.insert(0, str(TOOLS))
        import diff_emu  # noqa: E402

        from bench_build import listing  # noqa: E402

        sass_text = listing(code, labels)
        rng = random.Random(0)
        for _ in range(trials):
            inputs = [tuple(diff_emu.rand32(rng) for _ in range(4)) for _ in range(32)]
            k, kf = diff_emu.rand32(rng), diff_emu.rand32(rng)
            got = run_probe(sass_text, inputs, k, kf)
            if run_probe(sass_text, inputs, k, kf, poison=0xDEADBEEF) != got:
                return "failed", "output depends on uninitialized register contents"
            exp = diff_emu.run_ptx(ptx_text, kname, inputs, k, kf)
            for lane, (g, (e, undef, approx)) in enumerate(zip(got, exp)):
                if not undef and not diff_emu.same("verify", g, e, approx):
                    return "failed", f"lane {lane}: ours {g} vs PTX interpreter {e}"
        return "ok", f"{trials} random trials (probe ABI), bit-exact against the PTX reference interpreter"

    from openassembler.ptx.interp import NotModeled
    from openassembler.verify import run_and_diff

    for trial in range(trials):
        try:
            mismatches = run_and_diff(module, kname, code, labels, n_elems=64, seed=trial)
        except NotModeled as e:
            return "not-verified", (f"the PTX reference interpreter can't model this kernel ({e}). "
                                    "The cubin was kept, but --verify cannot check it; write your own "
                                    "test (see docs/FRONTENDS.md, \"What --verify proves\")")
        if mismatches:
            m = mismatches[0]
            return "failed", (f"{m.param}[{m.index}]: ours {m.ours:#x} vs PTX interpreter {m.expected:#x} "
                              f"({len(mismatches)} mismatching elements in trial {trial})")
    return "ok", (f"{trials} random trials (generic, {kname}'s own parameters), "
                  "bit-exact against the PTX reference interpreter")


# ---- container ------------------------------------------------------------------

_ENTRY_DECL = re.compile(r"(?:\.(?:visible|weak)\s+)?\.entry\s+([\w$]+)\s*\(")


def _skip_comment_or_string(s: str, i: int) -> int:
    if s.startswith("//", i):
        j = s.find("\n", i)
        return len(s) if j < 0 else j
    if s.startswith("/*", i):
        j = s.find("*/", i + 2)
        return len(s) if j < 0 else j + 2
    if s[i] == '"':
        j = i + 1
        while j < len(s) and s[j] != '"':
            j += 2 if s[j] == "\\" else 1
        return j + 1
    return i


def restrict_ptx(ptx: str, keep: set[str]) -> str:
    """The same PTX module with every .entry not in `keep` removed (its whole
    declaration and body), so ptxas builds a container with exactly the kernels that
    go into the cubin. Device functions, globals and the header stay untouched."""
    out, pos = [], 0
    for m in _ENTRY_DECL.finditer(ptx):
        if m.start() < pos or m[1] in keep:
            continue
        i, depth = m.end(), 0
        while i < len(ptx):
            j = _skip_comment_or_string(ptx, i)
            if j != i:
                i = j
                continue
            c = ptx[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            elif c == ";" and depth == 0:      # a bare declaration (prototype) without a body
                i += 1
                break
            i += 1
        out.append(ptx[pos:m.start()])
        out.append(f"// (entry {m[1]} left out of this cubin by openassembler)\n")
        pos = i
    out.append(ptx[pos:])
    return "".join(out)


def splice_kernels(base: bytes, ours: dict[str, tuple[list[int], int, list[int]]]) -> bytes:
    """Splice several kernels' code (name -> (words, nregs, exits)) into one ptxas
    container. Kernels not in `ours` keep ptxas's code and its register count."""
    from openassembler.container import nvinfo
    from openassembler.container.cubin import splice_text
    from openassembler.container.elf import Elf

    orig = Elf.parse(base)
    sym_names = {i: s.name for i, s in enumerate(orig.symbols)}
    orig_regs = {}
    for r in nvinfo.parse(orig.section(".nv.info").data):
        if r.name == "REGCOUNT":
            fn, n = struct.unpack_from("<II", r.data, 0)
            orig_regs[fn] = n
    cub = base
    for name, (words, nregs, exits) in ours.items():
        cub = splice_text(cub, name, words, nregs, exits, replace_helpers=True)
    # splice_text set every REGCOUNT record to the last kernel's count; put each back.
    e = Elf.parse(cub)
    recs = nvinfo.parse(e.section(".nv.info").data)
    for r in recs:
        if r.name == "REGCOUNT":
            fn = struct.unpack_from("<I", r.data, 0)[0]
            name = sym_names.get(fn)
            n = ours[name][1] if name in ours else orig_regs.get(fn, struct.unpack_from("<II", r.data)[1])
            r.data = struct.pack("<II", fn, n)
    e.replace_data(".nv.info", nvinfo.serialize(recs))
    return e.serialize()


def regcounts(cubin: bytes) -> dict[str, int]:
    """kernel name -> registers as the driver will see them (.nv.info REGCOUNT)."""
    from openassembler.container import nvinfo
    from openassembler.container.elf import Elf

    e = Elf.parse(cubin)
    out = {}
    for r in nvinfo.parse(e.section(".nv.info").data):
        if r.name == "REGCOUNT":
            fn, n = struct.unpack_from("<II", r.data, 0)
            if fn < len(e.symbols):
                out[e.symbols[fn].name] = n
    return out


def _scratch_tool(data: bytes, suffix: str, fn):
    """Run fn(path) on a scratch copy of `data` under the repo (on macOS the toolkit
    runs in a VM that only sees $HOME, wherever the output folder is)."""
    import os
    import uuid

    oracle.SCRATCH.mkdir(parents=True, exist_ok=True)
    tmp = oracle.SCRATCH / f"build_{os.getpid()}_{uuid.uuid4().hex[:8]}{suffix}"
    tmp.write_bytes(data)
    try:
        return fn(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def ptxas_build(ptx: str, arch: str) -> bytes:
    """ptxas -O3 (its default) on a PTX module -> cubin bytes."""
    def run(src: Path) -> bytes:
        out = src.with_suffix(".cubin")
        try:
            oracle.ptxas(src, out, arch=arch)
            return out.read_bytes()
        finally:
            out.unlink(missing_ok=True)
    return _scratch_tool(ptx.encode(), ".ptx", run)


def disasm(cubin: bytes) -> str:
    return _scratch_tool(cubin, ".cubin", oracle.disasm_text)


def split_listing(text: str) -> dict[str, str]:
    """nvdisasm output of a multi-kernel cubin -> {kernel: that .text section's lines}."""
    out: dict[str, list[str]] = {}
    cur = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(".section ") or s.startswith(".section\t"):
            sec = s.split()[1].split(",")[0]
            cur = sec[len(".text."):] if sec.startswith(".text.") else None
        if cur is not None:
            out.setdefault(cur, []).append(line)
    return {k: "\n".join(v) for k, v in out.items()}


def sanity_check_listing(section_text: str, code) -> str | None:
    """Soft check, never fatal: every non-branch instruction nvdisasm printed should
    be the one we encoded (branch targets are renamed by nvdisasm, so those lines
    legitimately differ, and nvdisasm prints an f32 immediate we wrote as raw bits,
    FADD R3, R3, 0x3f800000, as its value, 1). Returns a warning or None."""
    norm = lambda s: re.sub(r"\s+", "", s)  # noqa: E731

    def same(w: str, g: str) -> bool:
        if norm(w) == norm(g):
            return True
        wt, gt = norm(w).split(","), norm(g).split(",")
        if len(wt) != len(gt):
            return False
        for a, b in zip(wt, gt):
            if a == b:
                continue
            m = re.fullmatch(r"(-?)0x([0-9a-f]{1,8})", a)
            if not m:
                return False
            if b == f"{m[1]}{int(m[2], 16)}":          # integer immediate printed in decimal
                continue
            try:
                v = struct.unpack("<f", struct.pack("<I", int(m[2], 16)))[0]
                if m[1]:
                    v = -v
                if float(b.replace("+INF", "inf").replace("-INF", "-inf")) != v and str(v) != b:
                    return False
            except (ValueError, struct.error):
                return False
        return True

    got = [i.text for i in oracle.parse_listing(section_text)]
    want = [t for t, _ in code]
    bad = [(w, g) for w, g in zip(want, got) if "BRA" not in w and "BSSY" not in w and not same(w, g)]
    if len(got) < len(want):
        return f"nvdisasm listed {len(got)} instructions, we encoded {len(want)}"
    if bad:
        return (f"{len(bad)} instruction(s) disassemble differently than we encoded them, "
                f"e.g. {bad[0][0]!r} -> {bad[0][1]!r}")
    return None


def listing_insn_count(section_text: str) -> int:
    """Instructions in a ptxas-built section, counted like count_insns: no NOP padding,
    no trailing self-branch."""
    ins = oracle.parse_listing(section_text)
    texts = [i.text for i in ins]
    while texts and texts[-1].startswith("NOP"):
        texts.pop()
    if texts and texts[-1].startswith("BRA"):
        texts.pop()
    return sum(1 for t in texts if not t.startswith("NOP"))


# ---- one source file ---------------------------------------------------------------

_C_SIG = re.compile(r"__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?void\s+"
                    r"(?:__launch_bounds__\s*\([^)]*\)\s*)?(\w+)\s*\(([^)]*)\)")


def c_param_names(text: str) -> dict[str, list[str]]:
    """kernel -> its parameter names as written in CUDA C. nvcc's PTX names them
    <kernel>_param_<i>, which hides that the last one is `n`; `openassembler run`'s
    auto-harness uses these to give n/count/size-named ints the element count.
    Best effort (extern "C" kernels; templates and macros are simply not matched)."""
    from openassembler.project import _C_COMMENT

    out = {}
    for m in _C_SIG.finditer(_C_COMMENT.sub(" ", text)):
        names = []
        for part in m[2].split(","):
            ids = re.findall(r"[A-Za-z_]\w*", part.split("=")[0])
            if ids and ids[-1] != "void":
                names.append(ids[-1])
        out[m[1]] = names
    return out

class ToolkitMissing(Exception):
    pass


def _need(tool: str) -> None:
    if not oracle.available(tool):
        raise ToolkitMissing(f"{tool} not available: {oracle.NEEDS_TOOLKIT}")


@dataclass
class KernelOut:
    name: str
    params: list
    compiled_by: str = "openassembler"     # | "ptxas-fallback"
    instructions: int | None = None
    registers: int | None = None
    verify: str = "not-run"                # ok | failed | not-run | not-verified
    note: str = ""
    error: str = ""                        # set: not in the cubin
    verify_msg: str = ""
    code: object = None
    labels: object = None
    words: object = None

    source_params: list[str] | None = None
    tensor: dict | None = None
    gemm: dict | None = None
    rowwise: dict | None = None

    def record(self) -> dict:
        r = {"name": self.name, "params": [list(p) for p in self.params], "compiled_by": self.compiled_by,
             "backend": self.compiled_by,
             "instructions": self.instructions, "registers": self.registers, "verify": self.verify}
        if self.source_params:
            r["source_params"] = self.source_params
        if self.note:
            r["note"] = self.note
        if self.tensor is not None:
            r["tensor"] = self.tensor
        if self.gemm is not None:
            r["gemm"] = self.gemm
        if self.rowwise is not None:
            r["rowwise"] = self.rowwise
        return r


@dataclass
class SourceOut:
    src: Path
    language: str
    name: str                                # output base name (cubin/<name>.cubin)
    sha: str = ""
    kernels: list[KernelOut] = field(default_factory=list)
    error: str = ""                          # whole-file failure (frontend, ptxas, ...)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    all_kernels: list[str] = field(default_factory=list)   # what the file has (for --kernel errors)
    files: dict[str, bytes | str] = field(default_factory=dict)
    rc: int = 0
    native_ownership: dict | None = None

    @property
    def kept(self) -> list[KernelOut]:
        return [k for k in self.kernels if not k.error]

    @property
    def failed(self) -> list[KernelOut]:
        return [k for k in self.kernels if k.error]

    def paths(self, no_sass: bool) -> dict[str, str | None]:
        return {"cubin": f"cubin/{self.name}.cubin", "sass": None if no_sass else f"sass/{self.name}.sass",
                "ptx": f".build/ptx/{self.name}.ptx", "reference": f".build/reference/{self.name}.cubin"}

    def entry(self, no_sass: bool) -> dict:
        e = {"source": str(self.src.resolve()), "source_sha256": self.sha, "language": self.language,
             **self.paths(no_sass), "kernels": [k.record() for k in self.kept]}
        if self.native_ownership is not None:
            e['native_ownership'] = self.native_ownership
        if self.failed:
            e["failed"] = [{"name": k.name, "error": k.error.splitlines()[0]} for k in self.failed]
        return e


def _first_line(s: str) -> str:
    lines = [l for l in str(s).strip().splitlines() if l.strip()]
    return lines[0] if lines else str(s)


def build_source(src: Path, language: str, name: str, opt: Options, *, single: bool) -> SourceOut:
    """Compile one source file into in-memory outputs (SourceOut.files); nothing is
    written here. `single`: a --kernel that isn't in this file is an error (in a repo
    it just means this file has nothing to build)."""
    from openassembler.backend import ra
    from openassembler.frontend import FrontendError, to_ptx_module
    from openassembler.isa.encode import EncodeError
    from openassembler.ptx.parse import parse_module

    so = SourceOut(src, language, name)
    try:
        so.sha = P.sha256_file(src)
    except OSError as e:
        so.error, so.rc = f"can't read {src}: {e.strerror}", 1
        return so

    # 1. frontend
    try:
        if language == "cuda":
            _need("nvcc")
        ptx, names, notes = to_ptx_module(src, arch=opt.arch, include_dirs=opt.include_dirs,
                                          nvcc_flags=opt.nvcc_flags)
        so.notes += notes
    except KeyboardInterrupt:
        raise
    except ToolkitMissing as e:
        so.error, so.rc = str(e), 1
        return so
    except FrontendError as e:
        so.error, so.rc = str(e), 1
        return so
    except Exception as e:  # pyfront.NotSupported, cppfront.CppFrontendError, an import error in a .py, ...
        so.error, so.rc = f"{type(e).__name__}: {e}", 1
        return so
    if opt.dump_ptx:
        print(ptx)
    try:
        module = parse_module(ptx)
    except Exception as e:  # noqa: BLE001 -- PTX we can't parse (yet): report, don't crash
        so.error, so.rc = f"can't parse the PTX: {type(e).__name__}: {e}", 1
        return so
    so.all_kernels = list(module.kernels) or names
    cnames = c_param_names(src.read_text(errors="replace")) if language == "cuda" else {}
    if not module.kernels:
        so.error, so.rc = f"{src} has no kernels (.entry)", 1
        return so
    sel = list(module.kernels)
    if opt.kernel:
        if opt.kernel not in module.kernels:
            if single:
                so.error, so.rc = f"kernel {opt.kernel!r} not found in {src}; have {list(module.kernels)}", 1
            return so
        sel = [opt.kernel]

    # 2. our backend (+ --verify) per kernel
    from openassembler.gemm import contract_from_source
    gemm_contract = contract_from_source(ptx)
    from openassembler.rowwise import contract_from_source as rowwise_contract_from_source
    rowwise_contract = rowwise_contract_from_source(ptx)
    for kname in sel:
        k = KernelOut(kname, list(module.kernels[kname].params))
        tensor_config = None
        if gemm_contract is not None and kname == "gemm":
            from openassembler.backend.tensor import TensorConfig
            tensor_config = TensorConfig(pointer_mode=gemm_contract["pointer_mode"],
                                         pointer_advance=gemm_contract.get("pointer_advance", "late"))
        if len(cnames.get(kname, ())) == len(k.params) and any(n != p[1] for n, p in zip(cnames.get(kname, ()), k.params)):
            k.source_params = cnames[kname]
        so.kernels.append(k)
        why, can_fallback = "", False
        try:
            code, labels, words, exits, nregs = generate(module.kernels[kname], opt.arch, opt.opt_level,
                                                      tensor_config=tensor_config)
        except KeyboardInterrupt:
            raise
        except ra.NotSupported as e:
            why = f"our backend can't compile this kernel yet: {e}"
            can_fallback = True
        except EncodeError as e:
            why = f"our encoder can't encode this kernel: {e}"
            can_fallback = True
        except Exception as e:  # noqa: BLE001
            why = f"internal error in our backend ({type(e).__name__}: {e}); please report it"
        else:
            k.code, k.labels, k.words = code, labels, (words, nregs, exits)
            k.instructions, k.registers = count_insns(code), nregs
            from openassembler.backend.tensor import needs_tensor, policy
            if needs_tensor(module.kernels[kname]):
                k.tensor = {**policy(tensor_config), "architecture": "sm_80",
                            "async_scoreboard": 0, "ordinary_scoreboards": [1, 2, 3, 4, 5],
                            "requested_opt_level": opt.opt_level}
            if gemm_contract is not None and kname == "gemm":
                k.gemm = gemm_contract
            if rowwise_contract is not None and kname == rowwise_contract["kernel"]:
                k.rowwise = rowwise_contract
            if opt.verify:
                try:
                    k.verify, k.verify_msg = verify_kernel(module, kname, code, labels, ptx, opt.trials)
                except KeyboardInterrupt:
                    raise
                except Exception as e:  # noqa: BLE001 -- the emulator/interpreter gave up: say so
                    k.verify, k.verify_msg = "not-verified", f"the checker couldn't run ({type(e).__name__}: {e})"
                if k.verify == "failed":
                    why = f"--verify FAILED: {k.verify_msg}"
                    so.rc = 1
                elif k.verify == "not-verified":
                    k.note = "not verified: " + _first_line(k.verify_msg)
                    so.rc = max(so.rc, 2) if so.rc != 1 else 1
        if why:
            k.code = k.labels = k.words = None
            k.instructions = k.registers = None
            if opt.fallback == "ptxas" and can_fallback:
                k.compiled_by, k.note = "ptxas-fallback", why.splitlines()[0]
                k.verify = "not-run"
            else:
                k.error = why
                so.rc = 1
    if not so.kept:
        return so

    # 3. one ptxas build of exactly the kept kernels: container + reference
    keep = {k.name for k in so.kept}
    if keep != set(module.kernels):
        ptx = restrict_ptx(ptx, keep)
    try:
        _need("ptxas")
        ref = ptxas_build(ptx, opt.arch)
    except KeyboardInterrupt:
        raise
    except ToolkitMissing as e:
        so.error, so.rc = str(e), 1
        return so
    except subprocess.CalledProcessError as e:
        why = (e.stderr or "").strip() or f"exit code {e.returncode}"
        so.error, so.rc = ("couldn't build the ELF container with ptxas (our backend splices its code "
                           f"into one; see README \"How it works\"):\n  {why}"), 1
        return so
    except (FileNotFoundError, oracle.OracleError) as e:
        so.error, so.rc = str(e), 1
        return so

    # 4. splice every kernel we compiled into that one container
    ours = {k.name: k.words for k in so.kept if k.compiled_by == "openassembler"}
    cubin = splice_kernels(ref, ours)
    regs = regcounts(cubin)
    for k in so.kept:
        k.registers = regs.get(k.name, k.registers)
    # Bind deployed bytes to every exact OA instruction, even with --no-sass.
    # Mixed fallback modules intentionally receive no native ownership ledger.
    if all(k.compiled_by == 'openassembler' for k in so.kept):
        from openassembler.container.cubin import text_words
        from openassembler.native_ownership import record as ownership_record
        try:
            for k in so.kept:
                if text_words(cubin,k.name) != k.words[0]:
                    raise ValueError(f'{k.name}: cubin text differs from OA encoder output')
            so.native_ownership = ownership_record(cubin,[k.record() for k in so.kept],
                                                   ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest())
        except (ValueError,StopIteration,KeyError,IndexError,struct.error) as exc:
            so.error,so.rc = f'could not establish native executable ownership: {exc}',1
            return so

    # 5. the .sass listing of the final cubin
    sass = None
    if not opt.no_sass:
        try:
            _need("nvdisasm")
            listing = disasm(cubin)
        except KeyboardInterrupt:
            raise
        except (ToolkitMissing, subprocess.CalledProcessError, FileNotFoundError, OSError, oracle.OracleError) as e:
            detail = _first_line(getattr(e, "stderr", "") or str(e))
            so.error, so.rc = (f"couldn't disassemble the cubin into a .sass file: {detail}\n"
                               "  nvdisasm comes with the CUDA toolkit: run env/fetch_cuda.sh (on macOS also "
                               "`colima start`), or pass --no-sass to skip this step."), 1
            return so
        sections = split_listing(listing)
        for k in so.kept:
            sec = sections.get(k.name)
            if sec is None:
                so.warnings.append(f"nvdisasm listed no .text.{k.name} section")
            elif k.compiled_by == "openassembler":
                if k.tensor is not None:
                    # Alias spellings (e.g. IMAD.SHL) are expected. Compare
                    # complete encoded words and controls, and independently
                    # check the tensor/memory operands before publication.
                    records = oracle.parse_listing(sec)
                    if [r.word for r in records[:len(k.words[0])]] != k.words[0]:
                        so.error, so.rc = f"{k.name}: tensor cubin words differ from our encoder output", 1
                        return so
                    norm = lambda text: re.sub(r"\s+", "", text.replace(".reuse", "").replace("+-", "-"))
                    for (want, _), got in zip(k.code, records):
                        if want.startswith(("HMMA", "LDSM", "LDGSTS", "LDGDEPBAR", "DEPBAR", "BAR", "STG")):
                            if norm(want) != norm(got.text):
                                so.error, so.rc = f"{k.name}: tensor instruction failed disassembly: {want} -> {got.text}", 1
                                return so
                    continue
                if (w := sanity_check_listing(sec, k.code)):
                    so.warnings.append(f"{k.name}: {w}")
            else:
                k.instructions = listing_insn_count(sec)
        head = [f"// openassembler {__version__}: {src.name} -> cubin/{name}.cubin "
                f"({opt.arch}, opt-level={opt.opt_level}); listing by nvdisasm"]
        for k in so.kept:
            if k.compiled_by == "ptxas-fallback":
                head.append(f"// kernel {k.name}: ptxas-fallback -- this kernel's code is ptxas's, NOT "
                            f"openassembler's ({k.note})")
            else:
                head.append(f"// kernel {k.name}: compiled by openassembler")
        sass = "\n".join(head) + "\n" + listing
    p = so.paths(opt.no_sass)
    so.files = {p["cubin"]: cubin, p["ptx"]: ptx, p["reference"]: ref}
    if sass is not None:
        so.files[p["sass"]] = sass
    return so


# ---- driver: a file or a tree -> output folder -------------------------------------------

@dataclass
class CompileResult:
    rc: int
    out_dir: Path | None = None
    built: list[SourceOut] = field(default_factory=list)   # sources whose cubin was written this run
    failed_sources: list[Path] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)


def _brief(msg: str) -> str:
    """The informative line of a (possibly multi-line, nvcc) error, for one-line summaries."""
    lines = [l.strip() for l in str(msg).splitlines() if l.strip()]
    for l in lines[1:]:
        if "error" in l.lower():
            return (l.split(": error: ", 1)[-1] if ": error: " in l else l)[:160]
    return (lines[0] if lines else str(msg))[:160]


def _print_source(so: SourceOut, out_dir: Path, opt: Options, single: bool) -> None:
    rel = lambda r: _show(out_dir / r)  # noqa: E731
    p = so.paths(opt.no_sass)
    if not single and (so.error or so.failed):
        # one line per failing file on stdout (in order with the ok lines); full detail on stderr
        what = so.error or "; ".join(f"{k.name}: {k.error}" for k in so.failed)
        print(f"  {'FAILED' if not so.kept else 'partial':<8} {_show(so.src)}: {_brief(what)}")
        sys.stdout.flush()
    if so.error:
        _err(f"{_show(so.src)}: {so.error}" if not single else so.error)
        return
    for k in so.failed:
        _err(f"{_show(so.src)}: {k.name}: {k.error}" if not single else f"{k.name}: {k.error}")
        if "can't compile this kernel yet" in k.error:
            print("  That PTX construct isn't implemented in openassembler/backend/ra.py. Rewrite the kernel "
                  "without it, pass --fallback ptxas to keep ptxas's code for this kernel, or see "
                  "docs/FRONTENDS.md for what is supported.", file=sys.stderr)
        elif "encode" in k.error:
            print("  See docs/FRONTENDS.md (\"Python\" / \"CUDA C/C++\" sections) for what is supported.",
                  file=sys.stderr)
    if not so.kept:
        return
    kept = so.kept
    if single:
        if len(kept) == 1 and kept[0].compiled_by == "openassembler":
            k = kept[0]
            print(f"{rel(p['cubin'])}  ({k.name}: {k.instructions} instructions, opt-level={opt.opt_level}, "
                  f"regs={k.registers})")
        else:
            print(f"{rel(p['cubin'])}  ({len(kept)} kernel{'s' * (len(kept) != 1)}, opt-level={opt.opt_level})")
            for k in kept:
                tag = "" if k.compiled_by == "openassembler" else f"  [ptxas-fallback: {k.note}]"
                print(f"  {k.name}: {k.instructions} instructions, regs={k.registers}{tag}")
        if p["sass"]:
            print(f"{rel(p['sass'])}  (SASS listing from nvdisasm)")
    else:
        status = "ok" if all(k.compiled_by == "openassembler" for k in kept) else "fallback"
        names = ", ".join(k.name + ("*" if k.compiled_by != "openassembler" else "") for k in kept)
        print(f"  {status:<8} {so.src.name if single else _show(so.src)} -> {p['cubin']}  [{names}]")
        for k in kept:
            if k.compiled_by != "openassembler":
                print(f"           * {k.name}: ptxas-fallback ({k.note})")
    for n in so.notes:
        print(f"note: {n}")
    for w in so.warnings:
        print(f"warning: {w}", file=sys.stderr)
    if opt.verify:
        for k in kept:
            if k.compiled_by != "openassembler":
                continue
            if k.verify == "ok":
                print(f"OK: {k.name}: {k.verify_msg}")
            elif k.verify == "not-verified":
                print(f"not verified: {k.name}: {k.verify_msg}", file=sys.stderr)
    for k in so.kernels:
        if k.verify == "failed" and k.error:
            print(f"(not written: {k.name} did not pass --verify)", file=sys.stderr)


def compile_paths(src: Path, out_dir: Path, opt: Options, o_name: str | None = None) -> CompileResult:
    """Compile into a tree held exclusively by this writer."""
    if not Path(src).exists() or opt.arch != 'sm_80':
        return _compile_paths(src, out_dir, opt, o_name)
    if type(opt.trials) is not int or opt.trials < 1:
        _err('--trials must be a positive integer')
        return CompileResult(1)
    try:
        with P.output_lock(out_dir):
            return _compile_paths(src, out_dir, opt, o_name)
    except (P.ProjectError, OSError) as exc:
        _err(str(exc))
        return CompileResult(1)


def _compile_paths(src: Path, out_dir: Path, opt: Options, o_name: str | None = None) -> CompileResult:
    """`openassembler compile SRC`: a file or a directory. Returns the exit code and what
    was written (cnr runs exactly those kernels)."""
    from openassembler.isa.encode import EncodeError

    src = Path(src)
    if not src.exists():
        _err(f"{src} does not exist")
        return CompileResult(1)
    try:
        encoder(opt.arch)            # an arch we have no encoder data for: say so before any work
    except EncodeError as e:
        _err(str(e))
        return CompileResult(1)
    single = src.is_file()
    out_dir_abs = Path(out_dir).resolve()
    src_abs = src.resolve()
    if not single and P._inside(src_abs, out_dir_abs):
        _err(f"{src} is inside the output folder {out_dir}; pass a different --out-dir")
        return CompileResult(1)

    try:
        if single:
            lang = P.language_of(src)
            if lang is None:
                _err(f"don't know how to compile {src.suffix or 'extension-less'!r} files "
                     f"(supported: {' '.join(P.SOURCE_EXTS)}); a directory compiles every kernel in it")
                return CompileResult(1)
            name, note = P.single_output_name(src, o_name)
            if note:
                print(note)
            candidates = [(src_abs, lang, name)]
            skipped: dict[str, int] = {}
            root = src_abs.parent
        else:
            if o_name is not None:
                _err("-o names the output of a single file; for a directory the outputs mirror the tree "
                     "(use --out-dir to choose where)")
                return CompileResult(1)
            root = src_abs
            files, _used_git = P.walk(root, out_dir_abs)
            cls = [P.classify(f, f.relative_to(root).as_posix()) for f in files]
            skipped = {}
            for c in cls:
                if c.language is None:
                    skipped[c.skip_reason] = skipped.get(c.skip_reason, 0) + 1
            comp = [c for c in cls if c.language]
            names = P.output_names([c.rel for c in comp])
            candidates = [(c.path.resolve(), c.language, names[c.rel]) for c in comp]
            if not candidates:
                what = ", ".join(f"{n} {k}" for k, n in sorted(skipped.items())) or "no files"
                _err(f"nothing to compile in {src}: no .ptx with .entry, .cu/.cpp with __global__, "
                     f"or .py with @kernel (found: {what}). For a PyTorch application, configure "
                     "[application] frontend=\"torch\" and entrypoint in openassembler.toml; "
                     "see docs/REPOSITORY_WORKFLOW.md")
                return CompileResult(1)
        out_dir_abs = P.prepare_out_dir(out_dir)
    except P.ProjectError as e:
        _err(str(e))
        return CompileResult(1)

    if not single:
        need = ["ptxas"] + (["nvdisasm"] if not opt.no_sass else []) + \
            (["nvcc"] if any(l == "cuda" for _, l, _ in candidates) else [])
        missing = [t for t in need if not oracle.available(t)]
        if missing:
            _err(f"{', '.join(missing)} not available: {oracle.NEEDS_TOOLKIT}"
                 + (" (or pass --no-sass if only nvdisasm is missing)" if missing == ["nvdisasm"] else ""))
            return CompileResult(1)
        print(f"compiling {len(candidates)} source file{'s' * (len(candidates) != 1)} under {_show(src)} "
              f"-> {_show(out_dir_abs)}/")

    # ---- previous manifest: what we may replace / clean up
    try:
        old = P.load_manifest(out_dir_abs)
    except P.ManifestError as e:
        _err(str(e))
        return CompileResult(1)
    applications = old.get('applications', []) if old else []
    application_files = {path.casefold() for app in applications for path in P.application_outputs(app)}
    for _source, _lang, name in candidates:
        if any(path.casefold() in application_files for path in (
                f'cubin/{name}.cubin', f'sass/{name}.sass', f'.build/ptx/{name}.ptx')):
            _err(f'output name {name!r} overlaps a compiled application; choose another -o or --out-dir')
            return CompileResult(1)
    entries: dict[str, dict] = {}
    if old:
        if single and old.get("root") and P._inside(src_abs, Path(old["root"]).resolve()):
            root = Path(old["root"]).resolve()           # keep repo-relative source paths
        for e in old["entries"]:
            e = dict(e)
            e["source"] = str(P.entry_source_abs(old, e).resolve())
            entries[e["cubin"]] = e
    cand_by_src = {str(s): f"cubin/{n}.cubin" for s, _l, n in candidates}
    for cub, e in list(entries.items()):
        s = Path(e["source"])
        gone = not s.exists()
        # A full tree compile owns the tree: a source under it that no longer has kernels,
        # or now maps to another output name (a same-stem sibling appeared), loses its old
        # outputs too. A single file or a --kernel run only cleans up deleted sources.
        no_longer = (not single and not opt.kernel and P._inside(s, root)
                     and cand_by_src.get(str(s)) != cub)
        if gone or no_longer:
            P.remove_outputs(out_dir_abs, P.entry_outputs(e))
            print(f"removed outputs of {_show(s)} ({'source deleted' if gone else 'no longer compiled here'})")
            del entries[cub]
    P.save_manifest(out_dir_abs, root, opt.arch, opt.opt_level, list(entries.values()), applications=applications)

    # ---- build
    rc = 0
    built: list[SourceOut] = []

    # Native source files are independent until their in-memory cubin/PTX
    # results are ready to publish. Bounded parallelism removes the largest
    # repository-build bottleneck (serial lowering plus ptxas/nvdisasm
    # subprocesses) while preserving deterministic result order. Set
    # OPENASSEMBLER_BUILD_JOBS=1 to reproduce the historical serial path.
    build_jobs = 1
    if not single and len(candidates) > 1 and not opt.dump_ptx:
        try:
            requested_jobs = int(os.environ.get("OPENASSEMBLER_BUILD_JOBS", "4"))
        except ValueError:
            requested_jobs = 4
        build_jobs = max(1, min(requested_jobs, len(candidates), 8))

    def build_one(item):
        s, lang, name = item
        o = opt if single else Options(**{**opt.__dict__, "include_dirs": [*opt.include_dirs, root]})
        return build_source(s, lang, name, o, single=single)

    if build_jobs == 1:
        results = [build_one(item) for item in candidates]
    else:
        print(f"building {len(candidates)} sources with {build_jobs} workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=build_jobs,
                                                    thread_name_prefix="openassembler-build") as pool:
            results = list(pool.map(build_one, candidates))

    for so in results:
        s, name = so.src, so.name
        if so.files:
            new = so.entry(opt.no_sass)
            prev = entries.get(new["cubin"])
            with P.defer_sigint():
                P.write_group(out_dir_abs, so.files)
                if prev:
                    stale = set(P.entry_outputs(prev)) - set(P.entry_outputs(new))
                    P.remove_outputs(out_dir_abs, sorted(stale))
                entries[new["cubin"]] = new
                P.save_manifest(out_dir_abs, root, opt.arch, opt.opt_level, list(entries.values()), applications=applications)
            built.append(so)
        elif (prev := entries.get(f"cubin/{name}.cubin")) and Path(prev["source"]).resolve() == s.resolve() and (so.error or so.failed):
            prev["build_failed"] = True
            P.save_manifest(out_dir_abs, root, opt.arch, opt.opt_level, list(entries.values()), applications=applications)
            so.warnings.append(f"kept the previous build of {_show(s)} in {prev['cubin']}; it is out of date")
        _print_source(so, out_dir_abs, opt, single)
        rc = 1 if 1 in (rc, so.rc) else max(rc, so.rc)

    if not single:
        if opt.kernel and not any(k.name == opt.kernel for so in results for k in so.kernels):
            have = sorted({k for so in results for k in so.all_kernels})
            _err(f"kernel {opt.kernel!r} not found in any source under {src}; have {have}")
            rc = 1
        _print_summary(results, skipped, out_dir_abs)
    return CompileResult(rc, out_dir_abs, built,
                         [so.src for so in results if not so.files and (so.error or so.failed)],
                         [{'source': str(so.src), 'error': so.error,
                           'kernels': [{'name': k.name, 'error': k.error} for k in so.failed]}
                          for so in results if so.error or so.failed])


def _print_summary(results: list[SourceOut], skipped: dict[str, int], out_dir: Path) -> None:
    ok = [so for so in results if so.files]
    failed_files = [so for so in results if so.error or (not so.files and so.failed)]
    ours = sum(1 for so in ok for k in so.kept if k.compiled_by == "openassembler")
    fb = sum(1 for so in ok for k in so.kept if k.compiled_by == "ptxas-fallback")
    kfail = sum(len(so.failed) for so in results)
    rows = [("files compiled", len(ok)), ("kernels by openassembler", ours), ("kernels by ptxas (fallback)", fb),
            ("kernels failed", kfail), ("files failed", len(failed_files)),
            ("files skipped", sum(skipped.values()))]
    print("\nsummary")
    for label, n in rows:
        print(f"  {label:<28} {n}")
    for reason, n in sorted(skipped.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"    {n:>4}  {reason}")
    if ok:
        print(f"outputs: {_show(out_dir)}/cubin, {_show(out_dir)}/sass, manifest {_show(out_dir / P.MANIFEST)}")
