# -*- coding: utf-8 -*-
"""Public functions and methods whose parameters are not in an
`Args:` section, and public objects with no `Example:` at all.

Usage: check_func_params.py [--strict] [path-prefix]

`--strict` exits non-zero when a parameter is undocumented, which is what CI
runs. The `Example:` inventory is advisory in both modes.
"""

import ast
import io
import os
import re
import sys


SRC = "src/eigenkit"
SECTION = re.compile(r"^\s*(Args|Attributes|Parameters):\s*$", re.M)
ENTRY = re.compile(r"^\s{4,}([*\w]+)\s*(\([^)]*\))?\s*:", re.M)


def rendered(rel):
    parts = rel.split(os.sep)
    return not any(p.startswith("_") and p != "__init__.py" for p in parts)


def documented(doc):
    if not doc:
        return set()
    names = set()
    for m in SECTION.finditer(doc):
        tail = doc[m.end() :]
        stop = SECTION.search(tail)
        body = tail[: stop.start()] if stop else tail
        names |= {n for n, _ in ENTRY.findall(body)}
    return names


def params(fn):
    a = fn.args
    out = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    return [p for p in out if p not in ("self", "cls") and not p.startswith("_")]


rows, no_example = [], []
for dp, dn, fns in os.walk(SRC):
    dn[:] = [d for d in dn if d != "__pycache__"]
    for fn in sorted(fns):
        if not fn.endswith(".py"):
            continue
        path = os.path.join(dp, fn)
        rel = os.path.relpath(path, SRC)
        if not rendered(rel):
            continue
        page = rel.replace("\\", "/")
        tree = ast.parse(io.open(path, encoding="utf-8").read())
        has_example = "Example:" in io.open(path, encoding="utf-8").read()
        if not has_example:
            no_example.append(page)

        def visit(node, owner="", owner_doc="", page=page):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name.startswith("_") and child.name != "__init__":
                        continue
                    # @overload stubs are typing declarations, not API
                    if any(
                        getattr(d, "id", getattr(d, "attr", "")) == "overload"
                        for d in child.decorator_list
                    ):
                        continue
                    p = params(child)
                    if not p:
                        continue
                    doc = ast.get_docstring(child) or ""
                    # __init__ is rendered from the class docstring's Args
                    named = documented(doc) | (
                        documented(owner_doc) if child.name == "__init__" else set()
                    )
                    missing = [x for x in p if x not in named]
                    if missing:
                        rows.append(
                            (
                                page,
                                (owner + "." if owner else "") + child.name,
                                len(missing),
                                len(p),
                                "no docstring" if not doc else "",
                            )
                        )
                elif isinstance(child, ast.ClassDef) and not child.name.startswith("_"):
                    visit(child, child.name, ast.get_docstring(child) or "")

        visit(tree)

args = sys.argv[1:]
strict = "--strict" in args
positional = [a for a in args if not a.startswith("-")]
only = positional[0] if positional else ""
shown = [r for r in rows if r[0].startswith(only)]
by_page = {}
for page, name, m, n, note in shown:
    by_page.setdefault(page, []).append((name, m, n, note))
for page in sorted(by_page):
    items = by_page[page]
    print("%-42s %d function(s)" % (page, len(items)))
    for name, m, n, note in items[:6]:
        print("      %-38s %d/%d params undocumented %s" % (name, m, n, note))
    if len(items) > 6:
        print("      ... and %d more" % (len(items) - 6))
print()
print("TOTAL functions/methods with undocumented params:", len(shown))
print("rendered modules with no Example: at all:", len(no_example))
for p in no_example:
    if p.startswith(only):
        print("   ", p)

if strict and shown:
    sys.exit(1)
