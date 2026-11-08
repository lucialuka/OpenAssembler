"""The splash banner: the exact OPENASSEMBLER wordmark art (openassembler.ui.logo)
above a bordered status/tips box -- the same two pieces as Claude Code's own
welcome screen, just stacked instead of side by side since our wordmark is
wider than Claude's small glyph.
"""

from __future__ import annotations

import re
import shutil

from openassembler import __version__
from openassembler.ui.colors import supports_color
from openassembler.ui.logo import LINES as LOGO_LINES

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _strip(s: str) -> str:
    return _ANSI.sub("", s)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        lines.append(cur)
    return lines or [""]


TIPS = [
    "Compiles Python (@kernel), CUDA C / C++ (__global__) and PTX to NVIDIA SASS"
    " (sm_80) with our own backend, not ptxas's code generator.",
    "`compile kernel.cu` writes output/cubin/kernel.cubin + output/sass/kernel.sass"
    " (+ manifest.json). `-o NAME` renames it, `--out-dir DIR` moves output/,"
    " `--verify` checks it against the PTX reference interpreter.",
    "`compile REPO`, `run REPO`, `cnr REPO`: build, reuse, or build and run."
    " Directory outputs live in REPO/output/{cubin,sass}; --out-dir overrides it.",
    "Repositories discover native GPU kernels by default."
    " Use --frontend torch/triton/inductor to explicitly capture a configured"
    " PyTorch workload through TorchInductor/Triton.",
    "`gemm --m 128 --n 128 --k 128` builds a specialized A100 FP16/BF16 tensor GEMM.",
    "`run` executes output/ on a GPU: our cubin vs ptxas's, bit-exact + timing."
    " `--backend local|modal|user@host`, `--spec`, `--dry-run`.",
    "All three repository commands accept --backend local|modal|user@host."
    " In an SSH terminal on the GPU host, use local. --dry-run only plans work.",
    "`cnr SRC` = compile and run in one step.",
    "`status` prints the headline results; `help` lists every command;"
    " `<command> --help` shows its flags.",
]


def logo_lines(color: bool | None = None) -> list[str]:
    """The literal wordmark art, or its plain (uncolored) text if color is
    disabled -- same visible shape either way."""
    if color is None:
        color = supports_color()
    return list(LOGO_LINES) if color else [_strip(l) for l in LOGO_LINES]


def render(width: int | None = None, include_logo: bool = True) -> str:
    cols = width or shutil.get_terminal_size(fallback=(80, 24)).columns
    cols = max(60, min(cols, 140))
    color = supports_color()

    out = []
    if include_logo:
        out.append("")
        out += logo_lines(color)
        out.append("")

    inner = cols - 2
    header = f"openassembler v{__version__} · Python / CUDA C / C++ → SASS"
    out.append(f"╭{'─' * inner}╮")
    out.append(f"│{header.center(inner)}│")
    out.append(f"│{' ' * inner}│")
    out.append(f"│{'Tips for getting started'.ljust(inner)}│")
    for tip in TIPS:
        for line in _wrap(tip, inner - 2):
            out.append(f"│  {line.ljust(inner - 2)}│")
    out.append(f"╰{'─' * inner}╯")
    return "\n".join(out)
