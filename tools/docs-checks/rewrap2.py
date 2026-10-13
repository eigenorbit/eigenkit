# -*- coding: utf-8 -*-
"""Re-wrap docstring paragraphs that a long link or formula pushes past 88 columns.

Safety, in order:
  1. only lines inside an ast docstring node are considered — code is unreachable;
  2. a paragraph is rewrapped only if it holds an autorefs link or a formula and
     a line over the limit;
  3. lines that look like code are refused;
  4. after rewrapping, the docstring's whitespace-normalised token sequence must
     be identical to before. Any mismatch aborts the whole run without writing.
"""

import ast
import io
import os
import re
import sys


SRC = "src/eigenkit"
LIMIT = 88
# Require a dot in the target so that indexing like raw[k][i] can never match.
LINK = re.compile(r"\[[^\]\n]*\]\[[^\]\n]*\.[^\]\n]*\]")
MATH = re.compile(r"\\\(.*?\\\)")
BULLET = re.compile(r"^([-*]\s+|\d+\.\s+)")
CODEISH = re.compile(
    r"^\s*(from |import |def |class |return |print\(|>>>|\.\.\.|#|```|\||[A-Za-z_]\w*\s*=[^=])"
)
FIELD = re.compile(r"^\s*[A-Za-z_][\w .\[\]|,'\"-]*:\s*$|::\s*$")


def tokens(text):
    """Split on whitespace. A link is spaceless, so punctuation glued to it —
    a trailing comma, a closing paren — stays glued, exactly as it was."""
    kept = []
    sent = chr(0xE000)

    def stash(m):
        kept.append(m.group(0))
        return sent + str(len(kept) - 1) + sent

    masked = MATH.sub(stash, LINK.sub(stash, text))
    pat = re.compile(sent + r"(\d+)" + sent)
    return [pat.sub(lambda m: kept[int(m.group(1))], w) for w in masked.split()]


def wrap(words, first_indent, cont_indent):
    lines, cur, indent = [], [], first_indent
    for w in words:
        if cur and len(indent + " ".join(cur + [w])) > LIMIT:
            lines.append(indent + " ".join(cur))
            cur, indent = [w], cont_indent
        else:
            cur.append(w)
    if cur:
        lines.append(indent + " ".join(cur))
    return lines


def rewrap_span(lines, lo, hi):
    """Rewrap paragraphs in lines[lo:hi] (a docstring body). Returns new list."""
    out, i = [], lo
    while i < hi:
        line = lines[i]
        if not line.strip() or FIELD.match(line) or CODEISH.match(line) or '"""' in line:
            out.append(line)
            i += 1
            continue
        base = len(line) - len(line.lstrip())
        block, j = [], i
        while j < hi:
            nxt = lines[j]
            if not nxt.strip() or '"""' in nxt or FIELD.match(nxt) or CODEISH.match(nxt):
                break
            nbase = len(nxt) - len(nxt.lstrip())
            if j > i and nbase != base and not BULLET.match(block[0].strip()):
                break
            if j > i and nbase < base:
                break
            block.append(nxt)
            j += 1
        if not block:
            out.append(line)
            i += 1
            continue
        if any(len(ln) > LIMIT for ln in block) and any(
            LINK.search(ln) or MATH.search(ln) for ln in block
        ):
            indent = " " * base
            units, cur = [], []
            for ln in block:
                if BULLET.match(ln.strip()) and cur:
                    units.append(cur)
                    cur = [ln]
                else:
                    cur.append(ln)
            if cur:
                units.append(cur)
            for unit in units:
                text = " ".join(ln.strip() for ln in unit)
                m = BULLET.match(text)
                if m:
                    out.extend(wrap(tokens(text), indent, indent + " " * len(m.group(1))))
                else:
                    out.extend(wrap(tokens(text), indent, indent))
        else:
            out.extend(block)
        i = j
    return out


def docstring_spans(source):
    spans = []
    tree = ast.parse(source)
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
            spans.append((body[0].lineno - 1, body[0].end_lineno))
    return sorted(spans)


def docstring_tokens(source):
    """Whitespace-normalised tokens of every docstring, in order."""
    out = []
    tree = ast.parse(source)
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
            out.append((body[0].lineno, body[0].value.value.split()))
    return sorted(out)


pending, failures = [], []
for dirpath, dirnames, filenames in os.walk(SRC):
    dirnames[:] = [d for d in dirnames if d != "__pycache__"]
    for fn in sorted(filenames):
        if not fn.endswith(".py"):
            continue
        path = os.path.join(dirpath, fn)
        source = io.open(path, encoding="utf-8").read()
        if not (LINK.search(source) or MATH.search(source)):
            continue
        lines = source.split("\n")
        new = list(lines)
        for lo, hi in reversed(docstring_spans(source)):
            new[lo:hi] = rewrap_span(new, lo, hi)
        candidate = "\n".join(new)
        if candidate == source:
            continue
        try:
            before, after = docstring_tokens(source), docstring_tokens(candidate)
        except SyntaxError as exc:
            failures.append((path, "syntax error after rewrap: %s" % exc))
            continue
        if [t for _, t in before] != [t for _, t in after]:
            failures.append((path, "token sequence changed"))
            continue
        pending.append((path, candidate))

if failures:
    print("ABORTED, nothing written:")
    for p, why in failures:
        print("  ", os.path.relpath(p, SRC).replace("\\", "/"), "-", why)
    sys.exit(1)

if "--apply" in sys.argv:
    for path, text in pending:
        io.open(path, "w", encoding="utf-8", newline="").write(text)

for path, _ in pending:
    print("rewrap", os.path.relpath(path, SRC).replace("\\", "/"))
print(
    "TOTAL", len(pending), "files", "(applied)" if "--apply" in sys.argv else "(dry run)"
)
