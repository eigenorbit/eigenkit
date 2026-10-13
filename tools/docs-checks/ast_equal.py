# -*- coding: utf-8 -*-
"""Prove the src/ edits touched docstrings only: strip every docstring, then
compare the AST of the working tree against the AST at git HEAD."""

import ast
import io
import os
import subprocess
import sys


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO, "src/eigenkit")


def strip_docstrings(tree):
    for node in ast.walk(tree):
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
            node.body = body[1:] or [ast.Pass()]
    return tree


def dump(source):
    return ast.dump(strip_docstrings(ast.parse(source)))


bad, checked = [], 0
for dirpath, dirnames, filenames in os.walk(SRC):
    dirnames[:] = [d for d in dirnames if d != "__pycache__"]
    for fn in sorted(filenames):
        if not fn.endswith(".py"):
            continue
        path = os.path.join(dirpath, fn)
        rel = os.path.relpath(path, REPO).replace("\\", "/")
        head = subprocess.run(
            ["git", "show", "HEAD:" + rel],
            cwd=REPO,
            capture_output=True,
        )
        if head.returncode != 0:
            print("NEW FILE (not in HEAD):", rel)
            continue
        checked += 1
        a = dump(head.stdout.decode("utf-8"))
        b = dump(io.open(path, encoding="utf-8").read())
        if a != b:
            bad.append(rel)

print("checked", checked, "files")
if bad:
    print("AST DIFFERS in:")
    for r in bad:
        print("  ", r)
    sys.exit(1)
print("OK: every difference from HEAD is inside a docstring")
