#!/usr/bin/env python3
"""Read-only documentation checks: catalogue, links/anchors and public CLI coverage.

Uses only the standard library. The optional CLI check imports the workspace's
argument parser but never dispatches a command, builds a kernel or contacts a GPU.
"""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

STATUSES = {"implemented", "validated-not-integrated", "tested-not-selected",
            "untested-or-hardware-unavailable"}
IGNORED_PARTS = {".git", ".build", ".venv", "__pycache__", ".pytest_cache",
                 "node_modules", "third_party", "third-party"}


def mask_code(text):
    """Hide fenced/inline code while keeping newlines and source offsets intact."""
    result = []
    fence = None
    for line in text.splitlines(keepends=True):
        match = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence:
            close = match and match[1][0] == fence[0] and len(match[1]) >= fence[1]
            result.append(re.sub(r"[^\r\n]", " ", line))
            if close:
                fence = None
        elif match:
            fence = (match[1][0], len(match[1]))
            result.append(re.sub(r"[^\r\n]", " ", line))
        else:
            result.append(line)
    return "".join(result)


def without_inline_code(text):
    return re.sub(r"(`+)([^\n]*?)\1", lambda m: " " * len(m[0]), text)


def destinations(text):
    """Return (destination, zero-based offset) for ordinary Markdown/HTML links."""
    clean = without_inline_code(mask_code(text))
    refs = {}
    for match in re.finditer(r"^ {0,3}\[([^\]\n]+)\]:\s*(<[^>]+>|\S+)", clean, re.M):
        refs[match[1].strip().casefold()] = match[2]
    result = []
    for match in re.finditer(r"!?\[[^\]\n]*\]\(", clean):
        start, depth, pos, escaped = match.end(), 1, match.end(), False
        while pos < len(clean) and depth:
            character = clean[pos]
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
            pos += 1
        if depth:
            continue
        raw = clean[start:pos - 1].strip()
        if raw.startswith("<") and ">" in raw:
            destination = raw[1:raw.index(">")]
        else:
            destination = re.split(r"(?<!\\)\s", raw, maxsplit=1)[0]
        result.append((destination, match.start()))
    for match in re.finditer(r"!?\[([^\]\n]+)\]\[([^\]\n]*)\]", clean):
        key = (match[2] or match[1]).strip().casefold()
        if key in refs:
            result.append((refs[key].strip("<>"), match.start()))
    # A reference definition may be used by its short form: [label].
    for match in re.finditer(r"!?\[([^\]\n]+)\](?![\[(]|:)", clean):
        key = match[1].strip().casefold()
        if key in refs:
            result.append((refs[key].strip("<>"), match.start()))
    for match in re.finditer(r"\b(?:href|src)\s*=\s*([\"'])(.*?)\1", clean, re.I):
        result.append((match[2], match.start()))
    return result


def anchors(text):
    clean = mask_code(text)
    lines = clean.splitlines()
    headings = []
    for index, line in enumerate(lines):
        match = re.match(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            headings.append(match[1])
        elif index + 1 < len(lines) and line.strip() and re.fullmatch(r" {0,3}(?:=+|-+)\s*", lines[index + 1]):
            headings.append(line.strip())
    used = set()
    for heading in headings:
        # GitHub-like heading IDs: rendered link labels, lowercase, punctuation
        # removed, spaces replaced; duplicate headings get numeric suffixes.
        heading = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)
        heading = re.sub(r"<[^>]+>", "", html.unescape(heading))
        base = re.sub(r"[^\w\- ]", "", heading.lower().replace("`", "")).strip().replace(" ", "-")
        candidate, number = base, 0
        while candidate in used:
            number += 1
            candidate = f"{base}-{number}"
        used.add(candidate)
    for match in re.finditer(r"<[A-Za-z][^<>]*>", without_inline_code(clean)):
        for attribute, _, value in re.findall(r"\b(id|name)\s*=\s*([\"'])(.*?)\2", match[0], re.I):
            if attribute.lower() == "id" or re.match(r"<a(?:\s|>)", match[0], re.I):
                used.add(html.unescape(value))
    return used


class Checker:
    def __init__(self, root):
        self.root = root.resolve()
        self.errors = []
        self.link_count = 0
        self.anchor_cache = {}
        self.idea_count = 0
        self.cli_levels = 0
        self.cli_options = 0

    def display(self, path):
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def fail(self, path, message, line=1):
        self.errors.append(f"{self.display(path)}:{line}: {message}")

    def local_target(self, source, destination):
        destination = html.unescape(re.sub(r"\\([ ()#])", r"\1", destination))
        # External protocols, including mailto/data/app links, are outside this
        # offline check. A protocol-relative URL is external too.
        if destination.startswith("//") or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", destination):
            return None
        parsed = urlsplit(destination)
        location, fragment = unquote(parsed.path), unquote(parsed.fragment)
        if not location:
            target = source
        elif location.startswith("/"):
            absolute = Path(location)
            target = absolute if absolute.is_relative_to(self.root) else self.root / location.lstrip("/")
        else:
            target = source.parent / location
        return target.resolve(), fragment

    def documents(self):
        paths = {self.root / "README.md", self.root / "REPRODUCING.md"}
        paths.update((self.root / "docs").rglob("*.md"))
        paths.update((self.root / "exp").rglob("README.md"))
        paths.update((self.root / "compiler_exp").rglob("README.md"))
        def active(path):
            parts = path.relative_to(self.root).parts
            generated = any(part.lower() in IGNORED_PARTS
                            or part.lower() in {"build", "builds", "dist", "output", "cache"}
                            or part.startswith(("pytest-", "pytest_", "output_"))
                            or re.fullmatch(r".*_build(?:s|_.*)?", part.lower())
                            or part.lower().endswith(("-cache", "_cache"))
                            for part in parts[:-1])
            return path.is_file() and path.resolve().is_relative_to(self.root) and not generated
        return sorted(path for path in paths if active(path))

    def check_links(self, document):
        text = document.read_text(encoding="utf-8")
        for destination, offset in destinations(text):
            self.link_count += 1
            line = text.count("\n", 0, offset) + 1
            resolved = self.local_target(document, destination)
            if resolved is None:
                continue
            target, fragment = resolved
            if not target.is_relative_to(self.root):
                self.fail(document, f"local link escapes repository: {destination}", line)
            elif not target.exists():
                self.fail(document, f"missing local target: {destination}", line)
            elif fragment:
                anchor_target = target / "README.md" if target.is_dir() else target
                if anchor_target.suffix.lower() in {".md", ".markdown"} and anchor_target.is_file():
                    if anchor_target not in self.anchor_cache:
                        self.anchor_cache[anchor_target] = anchors(anchor_target.read_text(encoding="utf-8"))
                    if fragment not in self.anchor_cache[anchor_target]:
                        self.fail(document, f"missing anchor #{fragment} in {self.display(anchor_target)}", line)
                elif target.is_file() and re.fullmatch(r"L\d+(?:-L?\d+)?", fragment):
                    numbers = [int(value) for value in re.findall(r"\d+", fragment)]
                    count = len(target.read_text(encoding="utf-8", errors="replace").splitlines())
                    if min(numbers) < 1 or max(numbers) > count:
                        self.fail(document, f"line anchor outside {self.display(target)} ({count} lines): #{fragment}", line)

    def check_catalogue(self):
        path = self.root / "docs/research/catalogue.json"
        index = path.with_name("README.md")
        ideas_dir = path.parent / "ideas"
        if not path.is_file():
            self.fail(path, "missing research catalogue")
            return
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as error:
            self.fail(path, f"cannot read catalogue: {error}")
            return
        if not isinstance(obj, dict) or obj.get("schema") != 1 or not isinstance(obj.get("ideas"), list):
            self.fail(path, "expected schema=1 and an ideas list")
            return
        ids, listed = [], []
        for position, idea in enumerate(obj["ideas"], 1):
            if not isinstance(idea, dict) or not all(isinstance(idea.get(key), str) and idea[key].strip()
                                                  for key in ("id", "title", "status", "path", "summary")):
                self.fail(path, f"idea {position} needs nonempty id/title/status/path/summary strings")
                continue
            ids.append(idea["id"])
            if idea["status"] not in STATUSES:
                self.fail(path, f"idea {idea['id']}: unknown status {idea['status']!r}")
            raw = Path(idea["path"])
            target = ((self.root / raw) if raw.parts[:1] == ("docs",) else path.parent / raw).resolve()
            listed.append(target)
            if not target.is_relative_to(ideas_dir.resolve()) or target.suffix != ".md":
                self.fail(path, f"idea {idea['id']}: dossier must be an .md file under docs/research/ideas")
            elif not target.is_file():
                self.fail(path, f"idea {idea['id']}: missing dossier {idea['path']}")
            else:
                text = target.read_text(encoding="utf-8")
                matches = list(re.finditer(r"^[ \t]*(?:\*\*)?Status[ \t]*:[ \t]*(?:\*\*)?([a-z][a-z0-9-]*)", text, re.M))
                found = [match[1] for match in matches]
                if found != [idea["status"]]:
                    line = text.count("\n", 0, matches[0].start()) + 1 if matches else 1
                    self.fail(target, f"expected one literal Status: {idea['status']}; found {found}", line)
        for value, count in Counter(ids).items():
            if count > 1:
                self.fail(path, f"duplicate idea id {value!r} ({count} entries)")
        for target, count in Counter(listed).items():
            if count > 1:
                self.fail(path, f"dossier listed {count} times: {self.display(target)}")
        for orphan in sorted(set(ideas_dir.rglob("*.md")) - set(listed)):
            self.fail(orphan, "dossier is not listed in catalogue.json")
        if not index.is_file():
            self.fail(index, "missing research catalogue README")
        else:
            linked = set()
            for destination, _ in destinations(index.read_text(encoding="utf-8")):
                target = self.local_target(index, destination)
                if target:
                    linked.add(target[0])
            for target in sorted(set(listed) - linked):
                self.fail(index, f"catalogue README does not link dossier: {self.display(target)}")
        self.idea_count = len(obj["ideas"])

    def check_status(self):
        path = self.root / "docs/STATUS.md"
        if not path.is_file():
            self.fail(path, "missing architecture/status overview")
            return
        lines = mask_code(path.read_text(encoding="utf-8")).splitlines()
        for number, (line, following) in enumerate(zip(lines, lines[1:]), 1):
            cells = following.strip().strip("|").split("|")
            if (line.lstrip().startswith("|") and cells
                    and all(re.fullmatch(r"\s*:?-{3,}:?\s*", cell) for cell in cells)):
                if not line.startswith("|") or not following.startswith("|---"):
                    self.fail(path, "first table must start at column 1 with a |--- divider for the `status` CLI", number)
                return
        self.fail(path, "expected a Markdown headline table for the `status` command")

    def check_cli(self):
        path = self.root / "docs/COMMANDS.md"
        if not path.is_file():
            self.fail(path, "missing single-file command reference")
            return
        text = path.read_text(encoding="utf-8")
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(self.root))
        try:
            from openassembler.cli import build_parser
            parser = build_parser()
        except (ImportError, SyntaxError) as error:
            self.fail(path, f"cannot import workspace CLI ({error}); use its supported Python or --skip-cli")
            return
        def walk(current, names):
            self.cli_levels += 1
            command = " ".join(names) if names else "openassembler"
            if not re.search(re.escape(command) + r"(?![\w-])", text):
                self.fail(path, f"command path not documented: {command}")
            for action in current._actions:
                if isinstance(action, argparse._SubParsersAction):
                    for name, child in action.choices.items():
                        walk(child, [*names, name])
                else:
                    for spelling in action.option_strings:
                        self.cli_options += 1
                        if not re.search(r"(?<![\w-])" + re.escape(spelling) + r"(?![\w-])", text):
                            self.fail(path, f"option spelling not documented: {command} {spelling}")
        walk(parser, [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-cli", action="store_true", help="check documents without importing the compiler CLI")
    args = parser.parse_args()
    checker = Checker(args.root)
    checker.check_catalogue()
    checker.check_status()
    documents = checker.documents()
    for document in documents:
        checker.check_links(document)
    if not args.skip_cli:
        checker.check_cli()
    for message in checker.errors:
        print(message, file=sys.stderr)
    summary = (f"{len(documents)} Markdown files, {checker.link_count} links, {checker.idea_count} catalogue ideas, "
               f"{checker.cli_levels} CLI parser levels, {checker.cli_options} option/alias occurrences")
    if checker.errors:
        print(f"Documentation check failed: {len(checker.errors)} error(s); checked {summary}", file=sys.stderr)
        return 1
    print(f"Documentation check passed: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
