"""The interactive openassembler shell: `openassembler` with no arguments drops into a
persistent `>` prompt (same subcommands as the one-shot CLI, just without
having to retype `openassembler` each time). No AI features -- this is a plain
command dispatcher, like a REPL for any other build tool.
"""

from __future__ import annotations

import shlex
import sys

from openassembler.cli import build_parser, dispatch, source_arguments
from openassembler.ui.banner import render

PROMPT = "\033[1;34m❯\033[0m " if sys.stdout.isatty() else "> "
BUILTIN_HELP = """
Built-in shell commands (in addition to every CLI subcommand below):
  help, ?        show this message
  clear          clear the screen
  exit, quit     leave the shell (Ctrl-D also works)

Type any openassembler subcommand without the leading "openassembler", e.g.:
  compile examples/div_by_const.ptx -o out --verify   # -> output/cubin/out.cubin, output/sass/out.sass
  compile my_repo/ --backend modal                     # -> my_repo/output/{cubin,sass}
  run my_repo/ --backend me@gpubox                     # reuse that repository's saved build
  run --backend modal --dry-run                        # ours vs ptxas on a GPU (nothing is sent)
  cnr my_repo/ --backend me@gpubox                     # compile and run on the same SSH host
  gemm --m 128 --n 128 --k 128                         # native A100 tensor GEMM
  run examples/native_model --backend modal           # saved OA model cubins
  status
"""


def _try_readline() -> None:
    """Best-effort: history + line editing. Not available everywhere (e.g.
    Python builds without readline), so this degrades to plain input() silently."""
    try:
        import atexit
        import os
        import readline  # noqa: F401

        histfile = os.path.expanduser("~/.openassembler_history")
        try:
            readline.read_history_file(histfile)
        except OSError:
            pass
        readline.set_history_length(1000)
        atexit.register(readline.write_history_file, histfile)

        parser = build_parser()
        commands = list(parser._subparsers._group_actions[0].choices)  # noqa: SLF001

        def completer(text, state):
            opts = [c for c in commands if c.startswith(text)]
            return opts[state] if state < len(opts) else None

        readline.set_completer(completer)
        readline.parse_and_bind("tab: complete")
    except ImportError:
        pass


def run() -> int:
    print(render())
    print()
    _try_readline()
    parser = build_parser()

    while True:
        try:
            line = input(PROMPT).strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            continue

        if not line:
            continue
        if line in ("exit", "quit", "q"):
            return 0
        if line in ("help", "?"):
            parser.print_help()
            print(BUILTIN_HELP)
            continue
        if line == "clear":
            print("\033[2J\033[H", end="")
            continue

        try:
            args = parser.parse_args(source_arguments(shlex.split(line), parser))
        except SystemExit:
            # argparse already printed its own usage/error message.
            continue
        except ValueError as exc:
            print(f'error: {exc}', file=sys.stderr)
            continue

        try:
            rc = dispatch(args)
            if isinstance(rc, int) and rc not in (0,):
                print(f"(exit code {rc})")
        except SystemExit as e:
            if e.code not in (0, None):
                print(f"(exit code {e.code})")
        except KeyboardInterrupt:
            print("\n(interrupted)")
        except Exception as e:  # noqa: BLE001
            print(f"error: {e}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(run())
