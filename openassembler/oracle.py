"""Black-box oracle: NVIDIA's ptxas/nvdisasm run through env/cuda.sh.

The oracle turns raw 128-bit words into nvdisasm's text (or None when nvdisasm
rejects the encoding), and compiles PTX into cubins. Everything else in openassembler
is validated against it.

Word convention: a Volta+ instruction is one Python int `w = lo | (hi << 64)`,
stored little-endian as two qwords (lo first), exactly as in a cubin .text.
"""

from __future__ import annotations

import os
import re
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CUDA_SH = ROOT / "env" / "cuda.sh"
# Scratch space must live under $HOME: that's what the colima VM mounts.
SCRATCH = ROOT / "corpus" / ".tmp"

# A known-valid sm_70+ word (MOV R1, c[0x0][0x28]) used to fill slots that
# nvdisasm rejected, so the remaining words keep their addresses.
FILLER = 0x00000A0000017A02 | (0x000FE40000000F00 << 64)

_INSN = re.compile(
    r"/\*(?P<addr>[0-9a-f]{4,})\*/\s+(?P<text>.*?)\s*;\s*/\*\s*0x(?P<lo>[0-9a-f]{16})\s*\*/"
)
_HI = re.compile(r"^\s*/\*\s*0x(?P<hi>[0-9a-f]{16})\s*\*/\s*$")
_BAD_ADDR = re.compile(r"at address 0x(?P<addr>[0-9a-f]+)")


@dataclass(frozen=True)
class Insn:
    addr: int
    text: str
    word: int


class OracleError(RuntimeError):
    """The NVIDIA toolkit can't be reached or can't see the given path."""


def run_tool(*args: str, check: bool = True, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run an NVIDIA tool from the project root: directly on Linux, inside the
    colima VM on macOS (both handled by env/cuda.sh)."""
    return subprocess.run(
        [str(CUDA_SH), *args], cwd=cwd or ROOT, capture_output=True, text=True, check=check
    )


_available: dict[str, bool] = {}


def available(tool: str = "ptxas") -> bool:
    """Can `tool` (ptxas, nvdisasm, nvcc, ...) actually be run right now? False if
    the toolkit hasn't been fetched (env/fetch_cuda.sh), or on macOS if the
    colima VM isn't running. Cached per process; used by tests to skip rather than
    fail, and by the CLI to give one clear message."""
    if tool not in _available:
        try:
            r = run_tool(tool, "--version", check=False)
            _available[tool] = r.returncode == 0
        except (OSError, subprocess.SubprocessError):
            _available[tool] = False
    return _available[tool]


NEEDS_TOOLKIT = ("needs NVIDIA's CUDA toolkit: run env/fetch_cuda.sh "
                 "(on macOS also `colima start`) -- see REPRODUCING.md")


def platform_is_macos() -> bool:
    import platform

    return platform.system() == "Darwin"


def tool_path(p: Path | str) -> str:
    """How to name file `p` on a tool's command line. Inside the repo it's the
    repo-relative path; anywhere else it's absolute. On macOS the tool runs in a
    VM that only mounts $HOME, so a path outside $HOME can't be seen and raises
    OracleError up front instead of failing inside the VM with a confusing
    'no such file'. (On Linux the tool runs natively and can read any path.)"""
    resolved = Path(p).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        pass
    if platform_is_macos():
        try:
            resolved.relative_to(Path.home().resolve())
        except ValueError:
            raise OracleError(
                f"{resolved} is outside your home directory, which is all the colima VM "
                "mounts on macOS. Put the file somewhere under $HOME.") from None
    return str(resolved)


def disasm_text(path: Path | str) -> str:
    """`nvdisasm -hex -c` output for a cubin: the .sass text, two lines per
    instruction (text + 64-bit halves), exactly the format tools/ and the
    emulator read."""
    return run_tool("nvdisasm", "-hex", "-c", tool_path(path)).stdout


def words_to_bytes(words: list[int]) -> bytes:
    return b"".join(struct.pack("<QQ", w & (2**64 - 1), w >> 64) for w in words)


def parse_listing(text: str) -> list[Insn]:
    """Parse `nvdisasm -hex` output into (addr, text, word) records."""
    out: list[Insn] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = _INSN.search(line)
        if not m or i + 1 >= len(lines):
            continue
        h = _HI.match(lines[i + 1])
        if not h:
            continue
        word = int(m["lo"], 16) | (int(h["hi"], 16) << 64)
        out.append(Insn(int(m["addr"], 16), m["text"].strip(), word))
    return out


def _scratch_file(data: bytes, suffix: str) -> Path:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=suffix, dir=SCRATCH)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return Path(name)


def disasm_words(words: list[int], arch: str = "SM80", max_rounds: int = 64) -> list[str | None]:
    """Disassemble raw words; returns text per word, None where nvdisasm rejects it.

    nvdisasm aborts the whole batch on an illegal word but reports its address,
    so we replace reported words with FILLER and retry until the batch passes.
    """
    work = list(words)
    rejected: set[int] = set()
    for _ in range(max_rounds):
        path = _scratch_file(words_to_bytes(work), ".bin")
        try:
            r = run_tool("nvdisasm", "-b", arch, "-hex", str(path.relative_to(ROOT)), check=False)
        finally:
            path.unlink(missing_ok=True)
        if r.returncode == 0:
            by_addr = {i.addr: i.text for i in parse_listing(r.stdout)}
            return [None if k in rejected else by_addr.get(16 * k) for k in range(len(words))]
        bad = {int(a, 16) // 16 for a in _BAD_ADDR.findall(r.stderr)} - rejected
        if not bad:
            raise RuntimeError(f"nvdisasm failed without a usable address:\n{r.stderr}")
        rejected |= bad
        for k in bad:
            work[k] = FILLER
    raise RuntimeError("nvdisasm kept rejecting words; raise max_rounds")


def disasm_cubin(path: Path) -> list[Insn]:
    """Disassemble every code section of a cubin."""
    return parse_listing(disasm_text(path))


def ptxas(ptx: Path, out: Path, arch: str = "sm_80", *flags: str) -> None:
    """Compile PTX to a cubin. Raises CalledProcessError with ptxas's message."""
    out.parent.mkdir(parents=True, exist_ok=True)
    run_tool("ptxas", f"-arch={arch}", *flags, "-o", tool_path(out), tool_path(ptx))
