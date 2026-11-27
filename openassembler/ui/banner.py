"""A compact, responsive welcome for each interactive OpenAssembler session."""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import textwrap

from openassembler import __version__
from openassembler.ui.colors import supports_color
from openassembler.ui.logo import LINES as LOGO_LINES

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# COMPACT_LOGO = ("  ▄██▄  ▄██▄", "  █  █  █  █", "  ▀██▀  █▀▀█")
COMPACT_LOGO = (
    "\033[0;37m \033[0;97m▄▄▄\033[0;96m▄\033[0;97m▄\033[0;96m▄\033[0;37m \033[0;97m▄▄▄▄▄\033[0;37m  \033[0;97m▄▄▄\033[0;96m▄\033[0;97m▄\033[0;96m▄\033[0;37m \033[0;97m▄\033[0;37m    \033[0;97m▄\033[0;37m   \033[0;97m▄▄\033[0;96m▄\033[0;97m▄\033[0;96m▄\033[0;37m   \033[0;97m▄▄▄\033[0;96m▄▄\033[0;36m▄\033[0;37m  \033[0;97m▄▄▄\033[0;96m▄▄\033[0;36m▄\033[0;37m \033[0;97m▄▄▄\033[0;96m▄\033[0;97m▄\033[0;96m▄\033[0;37m \033[0;97m▄\033[0;37m       \033[0;97m▄\033[0;37m \033[0;97m▄▄▄▄\033[0;37m   \033[0;97m▄\033[0;37m      \033[0;97m▄▄▄\033[0;96m▄\033[0;97m▄\033[0;96m▄\033[0;37m \033[0;97m▄▄▄\033[0;96m▄\033[0;97m▄\033[0;37m \033[0m",
    "\033[0;97m█\033[0;37m     \033[0;96m▄\033[0;37m \033[0;97m█\033[0;37m    \033[0;96m█\033[0;37m \033[0;97m█\033[0;37m      \033[0;97m█▄\033[0;37m    \033[0;96m█\033[0;37m \033[0;97m█\033[0;37m     \033[0;96m█\033[0;37m \033[0;97m█\033[0;37m       \033[0;97m█\033[0;37m       \033[0;97m█\033[0;37m      \033[0;97m█▀▄\033[0;37m   \033[0;96m▄\033[0;97m▀\033[0;96m█\033[0;37m \033[0;97m█\033[0;37m   \033[0;97m▀▄\033[0;37m \033[0;97m█\033[0;37m      \033[0;97m█\033[0;37m      \033[0;97m█\033[0;37m    \033[0;96m█\033[0m",
    "\033[0;97m█\033[0;37m     \033[0;96;46m▄\033[0;37m \033[0;96m█▄▄\033[0;36m▄▄▀\033[0;37m \033[0;96m█▄\033[0;36m▄▄\033[0;37m   \033[0;97m█\033[0;37m \033[0;97m▀\033[0;96m▄\033[0;37m  \033[0;96m█\033[0;37m \033[0;97m█▄\033[0;96m▄▄\033[0;97m▄\033[0;96m▄\033[0;96;46m▀\033[0;37m \033[0;96m█▄▄\033[0;36m▄\033[0;96m▄\033[0;36m▄▄\033[0;37m \033[0;96m█▄▄\033[0;36m▄\033[0;96m▄\033[0;36m▄▄\033[0;37m \033[0;96m█▄\033[0;36m▄▄\033[0;37m   \033[0;96m█\033[0;37m  \033[0;96m▀\033[0;36m▄▀\033[0;37m  \033[0;96;46m▄\033[0;37m \033[0;96m█\033[0;97m▄\033[0;96m▄▄\033[0;36m▄▀\033[0;37m \033[0;97m█\033[0;37m      \033[0;96m█▄\033[0;36m▄▄\033[0;37m   \033[0;96m█▄\033[0;97m▄\033[0;96m▄\033[0;36m▄▀\033[0m",
    "\033[0;96m█\033[0;37m     \033[0;36m█\033[0;37m \033[0;36m█\033[0;37m      \033[0;96;46m▄\033[0;37m      \033[0;97m█\033[0;37m   \033[0;96m▀\033[0;36m▄█\033[0;37m \033[0;97m█\033[0;37m     \033[0;96;46m▀\033[0;37m       \033[0;90;46m▀\033[0;37m       \033[0;90;46m▀\033[0;37m \033[0;96;46m▄\033[0;37m      \033[0;96;46m▄\033[0;37m   \033[0;90m▀\033[0;37m   \033[0;36m█\033[0;37m \033[0;96;46m▀\033[0;37m    \033[0;36m█\033[0;37m \033[0;96m█\033[0;37m      \033[0;96;46m▄\033[0;37m      \033[0;96;46m▄\033[0;37m  \033[0;36m▀▄\033[0;37m \033[0m",
    "\033[0;97m▀\033[0;96m▄\033[0;36m▄\033[0;96m▄\033[0;36m▄\033[0;90m▄▀\033[0;37m \033[0;90;46m▄\033[0;37m      \033[0;36m█▄▄\033[0;90m▄\033[0;36m▄\033[0;90m▄\033[0;37m \033[0;96m▀▄\033[0;37m    \033[0;90m█\033[0;37m \033[0;96m█\033[0;37m     \033[0;90m█\033[0;37m \033[0;97m▄\033[0;96m▄▄\033[0;36m▄▄\033[0;90m▄▀\033[0;37m \033[0;97m▄\033[0;96m▄▄\033[0;36m▄▄\033[0;90m▄▀\033[0;37m \033[0;36m█▄▄\033[0;90m▄\033[0;36m▄\033[0;90m▄\033[0;37m \033[0;36m█\033[0;37m       \033[0;90m█\033[0;37m \033[0;96;46m▀\033[0;36m▄\033[0;96m▄\033[0;36m▄\033[0;90m▀\033[0;37m  \033[0;96m█\033[0;97m▄\033[0;96m▄\033[0;36m▄▄\033[0;90m▄\033[0;37m \033[0;36m█▄▄\033[0;90m▄\033[0;36m▄\033[0;90m▄\033[0;37m \033[0;36m█\033[0;37m    \033[0;90m█\033[0m",
)



def _strip(text: str) -> str:
    return _ANSI.sub("", text)


# def logo_lines(color: bool | None = None, width: int = 140) -> list[str]:
#     if color is None:
#         color = supports_color()
#     if width >= max(len(_strip(line)) for line in LOGO_LINES):
#         return list(LOGO_LINES) if color else [_strip(line) for line in LOGO_LINES]
#     return [f"\033[36m{line}\033[0m" if color else line for line in COMPACT_LOGO]

def logo_lines(color: bool | None = None, width: int = 140) -> list[str]:
    if color is None:
        color = supports_color()
    if width < max(len(_strip(line)) for line in COMPACT_LOGO):
        # Keep the updated wordmark intact on terminals that can display it.
        # A small OA glyph avoids wrapping the artwork on narrow terminals.
        short = ("  ▄██▄  ▄██▄", "  █  █  █  █", "  ▀██▀  █▀▀█")
        return [f"\033[36m{line}\033[0m" if color else line for line in short]
    if width >= max(len(_strip(line)) for line in LOGO_LINES):
        return list(LOGO_LINES) if color else [_strip(line) for line in LOGO_LINES]
    return list(COMPACT_LOGO) if color else [_strip(line) for line in COMPACT_LOGO]





def render(width: int | None = None, include_logo: bool = True) -> str:
    cols = max(20, min(width if width is not None else shutil.get_terminal_size((80, 24)).columns, 140))
    inner = cols - 4
    out = [""]
    if include_logo:
        out.extend(logo_lines(width=cols))
        out.append("")
    out.append("╭" + "─" * (cols - 2) + "╮")
    for text in (
        f"OpenAssembler v{__version__}",
        "Python · CUDA C/C++ · PTX → NVIDIA SASS",
        "",
        f"Workspace: {Path.cwd()}",
        "",
        "compile PATH --backend modal --verify",
        "run PATH --backend modal",
        "cnr PATH --backend modal --verify",
        "",
        "Backends: modal, local, or user@host",
        "help · Tab complete · ↑ history · exit",
    ):
        for line in textwrap.wrap(text, inner) or [""]:
            out.append(f"│ {line.ljust(inner)} │")
    out.append("╰" + "─" * (cols - 2) + "╯")
    return "\n".join(out)
