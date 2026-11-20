"""Detect whether terminal output should use ANSI color.

Honors NO_COLOR, OPENASSEMBLER_FORCE_COLOR, non-tty output and TERM=dumb.
Uses only the standard library.
"""

from __future__ import annotations

import os
import sys


def supports_color(stream=None) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("OPENASSEMBLER_FORCE_COLOR"):
        return True
    stream = sys.stdout if stream is None else stream
    try:
        tty = hasattr(stream, "isatty") and stream.isatty()
    except (OSError, ValueError):
        return False
    if not tty:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return True
