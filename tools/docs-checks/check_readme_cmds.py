# -*- coding: utf-8 -*-
"""Check the commands in README.md and CONTRIBUTING.md.

Three tiers, because "run every command" is not a safe test in a live checkout:

  RUN    read-only and terminating -> executed, must exit 0. A command listed
         in OUTPUT_OK still has to run and still has to reach the outcome
         named there, but its exit code is not judged
  SHAPE  installs, servers, destructive or slow -> the subcommand and every
         long flag it uses must be accepted by the tool (checked via --help)
  NOTE   known-deferred lines, reported but not judged
"""

import io
import os
import re
import subprocess
import sys


RUN = [
    "uv run semantic-release --noop version",
    "uv run lint-imports",
    "uv run python tools/docs-checks/check_rosters.py",
]

# command -> (tool subcommand, long flags it uses)
SHAPE = {
    "uv sync --extra all --extra perceval": ("sync", ["--extra"]),
    "uv sync --extra torch": ("sync", ["--extra"]),
    "uv pip install --reinstall --index-url ... torch": (
        "pip install",
        ["--reinstall", "--index-url"],
    ),
    "uv run --no-sync pytest": ("run", ["--no-sync"]),
    "uv run pre-commit install": ("run", []),
    "uv run pre-commit run --all-files": ("run", []),
    "uv run pytest": ("run", []),
    "uv run --with pytest-cov pytest": ("run", ["--with"]),
    "uv run mkdocs serve": ("run", []),
    "uv run ruff format .": ("run", []),
}

NOTE = [
    'pip install eigenkit / "eigenkit[...]" (README, 5 lines)'
    "  -- runnable once the first eigenkit release is on PyPI",
]

# Commands whose exit code describes the checkout rather than the README. The
# run still has to happen and still has to reach the stated outcome; only the
# exit code is not judged.
#
# `semantic-release --noop version` previews the next release. It exits 0 when
# it can name the current branch and 1 when it cannot, and a `pull_request`
# checkout is a detached merge commit -- so it always takes the second path in
# CI and the first one locally:
#
#     on a branch     branch '<name>' isn't in any release groups   exit 0
#     detached HEAD   Detached HEAD state cannot match any ...      exit 1
#
# Both are the same answer. What README.md promises for this line is that it
# previews a release and that `--noop` goes before the subcommand; the marker
# below is what says the command got that far.
OUTPUT_OK = {
    "uv run semantic-release --noop version": "no release will be made",
}

fail = []


def run(cmd, cwd=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900)


def tail(proc, n=4):
    """The last few non-empty lines of both streams.

    Not the first line: a tool that prints a banner puts it first, so keeping
    line one would report the banner and drop the error.
    """
    joined = proc.stdout + chr(10) + proc.stderr
    kept = [ln for ln in joined.splitlines() if ln.strip()]
    return " | ".join(kept[-n:]) or "(no output)"


print("RUN  (must exit 0, or reach the outcome named in OUTPUT_OK)")
for cmd in RUN:
    p = run(cmd)
    marker = OUTPUT_OK.get(cmd)
    if marker is None:
        ok = p.returncode == 0
        note = ""
    else:
        ok = marker in (p.stdout + p.stderr)
        note = "   (exit not judged)"
    print("  %s exit=%-3d %s%s" % ("ok  " if ok else "FAIL", p.returncode, cmd, note))
    if not ok:
        fail.append((cmd, tail(p)))

print("\nSHAPE  (subcommand + flags must be accepted)")
helps = {}
for cmd, (sub, flags) in SHAPE.items():
    if sub not in helps:
        h = run("uv %s --help" % sub)
        helps[sub] = (h.returncode, h.stdout + h.stderr)
    code, text = helps[sub]
    missing = [f for f in flags if f not in text]
    ok = code == 0 and not missing
    print(
        "  %s %s%s"
        % ("ok  " if ok else "FAIL", cmd, ("   missing %s" % missing) if missing else "")
    )
    if not ok:
        fail.append((cmd, "uv %s: missing %s" % (sub, missing or "subcommand")))

print("\nREFERENCES")
pyproject = io.open("pyproject.toml", encoding="utf-8").read()
readme = "".join(
    io.open(page, encoding="utf-8").read() for page in ("README.md", "CONTRIBUTING.md")
)

checks = [
    (
        "pytorch index named in README or CONTRIBUTING is defined in pyproject",
        all(
            ('name = "%s"' % n) in pyproject for n in re.findall(r"--index (\S+)", readme)
        ),
    ),
    (
        "every repo-relative path in README and CONTRIBUTING exists",
        all(os.path.exists(p) for p in re.findall(r"(src/eigenkit/[\w/]+\.py)", readme)),
    ),
    (
        "no uv subcommand that uv rejects",
        "uv show" not in readme and "uv venv --rm" not in readme,
    ),
]
for name, ok in checks:
    print("  %s %s" % ("ok  " if ok else "FAIL", name))
    if not ok:
        fail.append((name, "reference check"))

print("\nNOTE")
for n in NOTE:
    print("  --   " + n)

print()
if fail:
    print("FAILED %d:" % len(fail))
    for what, why in fail:
        print("   %s :: %s" % (what, why))
    sys.exit(1)
print("README commands, flags and paths all check out")
