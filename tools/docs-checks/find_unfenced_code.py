# -*- coding: utf-8 -*-
"""Find code inside docstrings that is not in a ```python fence.

An indented block with no language marker renders as an unhighlighted code
block at best, and as a run-together paragraph at worst. This looks for any
docstring line that reads like python but sits outside a fence.
"""

import ast
import io
import os
import re
import sys


SRC = "src/eigenkit"
CODE_LINE = re.compile(
    r"^\s*("
    r"from\s+[\w.]+\s+import\s"
    r"|import\s+[\w.]+\s*$"
    r"|def\s+\w+\s*\("
    r"|class\s+\w+\s*[(:]"
    r"|\w[\w.\[\]]*\s*=\s*\w[\w.]*\("
    r"|(qml|np|torch|plt)\.\w+\("
    r"|print\("
    r"|>>> "
    r")"
)


def docstring_spans(source):
    spans = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            spans.append(
                (
                    body[0].lineno - 1,
                    body[0].end_lineno,
                    getattr(node, "name", "<module>"),
                )
            )
    return sorted(spans)


hits = []
for dp, dn, fns in os.walk(SRC):
    dn[:] = [d for d in dn if d != "__pycache__"]
    for fn in sorted(fns):
        if not fn.endswith(".py"):
            continue
        path = os.path.join(dp, fn)
        source = io.open(path, encoding="utf-8").read()
        lines = source.split("\n")
        for lo, hi, owner in docstring_spans(source):
            inside = False
            for i in range(lo, min(hi, len(lines))):
                ln = lines[i]
                if ln.strip().startswith("```"):
                    inside = not inside
                    continue
                if inside:
                    continue
                if CODE_LINE.match(ln) and '"""' not in ln:
                    hits.append(
                        (
                            os.path.relpath(path, SRC).replace("\\", "/"),
                            i + 1,
                            owner,
                            ln.strip()[:62],
                        )
                    )

for h in hits:
    print("%-40s :%-5d %-24s %s" % h)

# ---- a heading underline joined into its paragraph is a rewrap accident ----
JOINED = re.compile(r"[A-Za-z]\s+(-{3,}|={3,}|~{3,})\s+[A-Za-z]")
UNDERLINE = re.compile(r"^\s*(-{3,}|={3,}|~{3,})\s*$")
struct = []
for dp, dn, fns in os.walk(SRC):
    dn[:] = [d for d in dn if d != "__pycache__"]
    for fn in sorted(fns):
        if not fn.endswith(".py"):
            continue
        path = os.path.join(dp, fn)
        text = io.open(path, encoding="utf-8").read()
        for i, ln in enumerate(text.splitlines(), 1):
            if JOINED.search(ln):
                struct.append(
                    (
                        os.path.relpath(path, SRC).replace("\\", "/"),
                        i,
                        "heading underline joined into the paragraph",
                    )
                )
            elif UNDERLINE.match(ln):
                struct.append(
                    (
                        os.path.relpath(path, SRC).replace("\\", "/"),
                        i,
                        "RST heading underline — use ## instead",
                    )
                )
for r in struct:
    print("%s:%d  %s" % r)
print("TOTAL structural markdown problems:", len(struct))

print("TOTAL unfenced code lines in docstrings:", len(hits))
sys.exit(1 if (hits or struct) else 0)
