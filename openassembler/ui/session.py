"""A persistent, atomic welcome stamp shared by installers and CLI processes."""

from __future__ import annotations

import os
from pathlib import Path


def _stamp_path() -> Path:
    base = Path(os.environ.get('XDG_STATE_HOME') or Path.home() / '.local' / 'state')
    return base / 'openassembler' / 'welcome-shown'


def consume_show_once() -> bool:
    """Claim the first welcome, including concurrent invocations.

    Unwritable state suppresses the optional welcome; compilation still works.
    """
    try:
        path = _stamp_path()
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        return True
    except OSError:
        return False
