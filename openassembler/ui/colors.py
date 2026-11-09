"""Detect whether terminal output should use ANSI color.

Honors NO_COLOR, OPENASSEMBLER_FORCE_COLOR, non-tty output and TERM=dumb.
Uses only the standard library.
"""

from __future__ import annotations

import os
import sys


def supports_color(stream=sys.stdout) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("OPENASSEMBLER_FORCE_COLOR"):
        return True
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return True
