"""Interactive commands using the same parser and dispatch as the one-shot CLI."""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import shlex
import sys
import time

from openassembler.cli import build_parser, dispatch, source_arguments
from openassembler.ui.banner import render
from openassembler.ui.colors import supports_color

BUILTINS = ("help", "?", "cd", "pwd", "clear", "exit", "quit", "q")
BUILTIN_HELP = """
  compile PATH --backend modal --verify   Compile a file or repository
  compiler PATH --backend modal          Alias for compile
  run PATH --backend modal               Run a saved build
  cnr PATH --backend user@host --verify   Compile and run

  help [COMMAND]   Show commands or a command's flags
  cd PATH / pwd    Change or show the workspace
  clear           Clear the terminal
  exit / quit     Leave OpenAssembler (or press Ctrl-D)

Choose one backend: modal, local, or user@host. Authenticate Modal with
`modal setup` in your terminal before starting OpenAssembler.
Use Tab for commands, flags and paths; ↑/↓ for history.
Ctrl-C cancels input or interrupts work and returns to the prompt.
"""


def _commands(parser):
    return next((a.choices for a in parser._actions if isinstance(a, argparse._SubParsersAction)), {})


def _shell_parser():
    parser = build_parser()

    def rename(current, path):
        current.prog = " ".join(path)
        for field in ("description", "epilog"):
            text = getattr(current, field)
            if text:
                setattr(current, field, text.replace("openassembler ", ""))
        seen = set()
        for name, child in _commands(current).items():
            if id(child) in seen:
                continue
            seen.add(id(child))
            rename(child, [*path, name])

    rename(parser, [])
    return parser


def completion_candidates(line: str, text: str, parser) -> list[str]:
    """Complete command paths, parser flags, backends and quoted filenames."""
    try:
        before = shlex.split(line[:len(line) - len(text)])
    except ValueError:
        return []
    current = parser
    if not before:
        choices = [*BUILTINS, *_commands(parser)]
    elif before[-1] in ("--backend", "--build-backend"):
        choices = ["modal", "local", "auto", "user@host"]
    else:
        if before[0] == "help":
            before = before[1:]
        for token in before:
            child = _commands(current).get(token)
            if child is not None:
                current = child
        choices = list(_commands(current))
        choices += [flag for a in current._actions for flag in a.option_strings]
    matches = [choice + " " for choice in choices if choice.startswith(text)]
    if not text.startswith("-"):
        partial = Path(text).expanduser()
        parent = partial.parent if text and not text.endswith("/") else partial
        prefix = text.rsplit("/", 1)[-1] if not text.endswith("/") else ""
        try:
            for path in sorted(parent.iterdir()):
                if path.name.startswith(prefix) and (prefix.startswith(".") or not path.name.startswith(".")):
                    spelling = (text + path.name if text.endswith("/") else
                                str(Path(text).parent / path.name) if "/" in text else path.name)
                    matches.append(shlex.quote(spelling + ("/" if path.is_dir() else "")) + ("" if path.is_dir() else " "))
        except OSError:
            pass
    return sorted(set(matches))


def _try_readline(parser):
    """Return a best-effort history saver; readline is optional."""
    try:
        import readline
    except ImportError:
        return lambda: None
    histfile = Path.home() / ".openassembler_history"
    try:
        readline.read_history_file(str(histfile))
    except OSError:
        pass
    readline.set_history_length(1000)
    readline.set_completer_delims(" \t\n")

    def completer(text, state):
        matches = completion_candidates(readline.get_line_buffer()[:readline.get_endidx()], text, parser)
        return matches[state] if state < len(matches) else None

    readline.set_completer(completer)
    readline.parse_and_bind("bind ^I rl_complete" if "libedit" in (readline.__doc__ or "") else "tab: complete")

    def save():
        try:
            readline.write_history_file(str(histfile))
            histfile.chmod(0o600)
        except OSError:
            pass

    return save


def _tint(text, code):
    return f"\033[{code}m{text}\033[0m" if supports_color() else text


def run() -> int:
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    parser = _shell_parser()
    if interactive:
        print(render())
        print()
    save_history = _try_readline(parser) if interactive else lambda: None
    last_rc = 0
    try:
        while True:
            try:
                # readline needs nonprinting color markers to calculate cursor position.
                prompt = "\001\033[1;36m\002❯\001\033[0m\002 " if supports_color() else "> "
                line = input(prompt if interactive else "")
                while line.endswith("\\") and (len(line) - len(line.rstrip("\\"))) % 2:
                    line = line[:-1] + input("… " if interactive else "")
            except EOFError:
                if interactive:
                    print()
                return last_rc
            except KeyboardInterrupt:
                print("\nInput cancelled. Type exit to leave.")
                continue
            try:
                raw = shlex.split(line, comments=True)
                if not raw:
                    continue
                if raw[0] == "openassembler":
                    raw = raw[1:]
                    if not raw:
                        print("Already in OpenAssembler. Type help for commands or exit to leave.")
                        continue
                if raw[0] in ("exit", "quit", "q"):
                    if len(raw) != 1:
                        raise ValueError("exit takes no arguments")
                    if interactive:
                        print(_tint("Session closed.", "2"))
                    return last_rc
                if raw[0] in ("help", "?"):
                    if len(raw) == 1:
                        parser.print_help()
                        print(BUILTIN_HELP)
                    else:
                        parser.parse_args([*raw[1:], "--help"])
                    continue
                if raw[0] == "cd":
                    if len(raw) > 2:
                        raise ValueError("usage: cd [PATH] (quote paths containing spaces)")
                    import os
                    os.chdir(Path(raw[1]).expanduser() if len(raw) == 2 else Path.home())
                    print(Path.cwd())
                    continue
                if raw == ["pwd"]:
                    print(Path.cwd())
                    continue
                if raw == ["clear"]:
                    if interactive:
                        print("\033[2J\033[H", end="", flush=True)
                    continue
                raw = [str(Path(token).expanduser()) if token.startswith("~/") else token for token in raw]
                raw = source_arguments(raw, parser)
                if raw[0] not in _commands(parser) and not raw[0].startswith("-"):
                    suggestion = difflib.get_close_matches(raw[0], [*_commands(parser), *BUILTINS], n=1)
                    hint = f" Did you mean {suggestion[0]}?" if suggestion else " Type help for commands."
                    raise ValueError(f"unknown command: {raw[0]}.{hint}")
                args = parser.parse_args(raw)
            except SystemExit as exc:
                last_rc = int(exc.code or 0)
                continue
            except (ValueError, OSError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                last_rc = 2
                continue

            feedback = interactive and "--json" not in raw
            started = time.monotonic()
            if feedback:
                mode = "Plan" if getattr(args, "dry_run", False) else "Working"
                target = getattr(args, "src", None) or getattr(args, "outdir", None)
                backend = getattr(args, "backend", None)
                detail = " · ".join(str(value) for value in (args.cmd, target, backend) if value)
                print(_tint(f"  {mode} · {detail}", "36"), flush=True)
            last_rc = dispatch(args) or 0
            if feedback:
                label = "Plan complete" if getattr(args, "dry_run", False) else "Done"
                label = label if last_rc == 0 else ("Interrupted" if last_rc == 130 else f"Failed (exit {last_rc})")
                print(_tint(f"  {label} · {time.monotonic() - started:.2f}s", "32" if last_rc == 0 else "31"))
                print()
    finally:
        save_history()


if __name__ == "__main__":
    raise SystemExit(run())
