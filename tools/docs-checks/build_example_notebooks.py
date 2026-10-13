# -*- coding: utf-8 -*-
"""Rebuild `examples/notebooks/*.ipynb` from the pages in `docs/examples/`.

The example pages are the source of truth: they are what the site publishes and
what the documentation checks run against. A hand-written notebook drifts from
its page as the library changes; a generated one cannot.

Each page becomes one notebook:

  prose            markdown cell
  ```python        code cell, empty output, ready to run
  ```text          markdown cell holding the output the page records, so a
                   re-run can be read against what was published

Relative links and image paths are rewritten so they still resolve from
`examples/notebooks/`, and a `!!!` admonition becomes a blockquote, which is the
closest thing a notebook renders.

Output is deterministic — same page, same bytes — so a rebuild that changes
nothing leaves the working tree clean.

The notebooks ship without outputs. The pages already show every result, so
outputs in the notebooks would only add weight (training progress bars, mostly)
and go stale. A rebuild therefore writes each notebook from scratch, and
`--check` compares the whole file: a notebook that has drifted from its page,
or that was saved after running it, fails.

`examples/notebooks/` is excluded from ruff (`extend-exclude` in `pyproject.toml`), and
has to stay excluded. Ruff formats the Python blocks of the pages themselves;
formatting the generated copy separately would only make it disagree with its
page.

The `example-notebooks` pre-commit hook runs this script, locally and in the
`lint` workflow, so a commit that edits a page rewrites its notebook.

Standard library only.

Usage:
    python tools/docs-checks/build_example_notebooks.py            # rebuild
    python tools/docs-checks/build_example_notebooks.py --check    # drift only
"""

import hashlib
import io
import json
import os
import posixpath
import re
import sys


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)

PAGES_DIR = "docs/examples"
NOTEBOOK_DIR = "examples/notebooks"

FENCE = re.compile(r"^```(\w*)\s*$")
ADMONITION = re.compile(r'^!!!\s+(\w+)(?:\s+"(.*)")?\s*$')
LINK = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)")
IMAGE_ATTR = re.compile(r"\{\s*\.theme-image-(light|dark)\s*\}\s*$")

KERNELSPEC = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}


def rewrite_target(target):
    """Point a page-relative link at the same file from the notebook directory.

    Args:
        target (str): The link target as the page writes it.

    Returns:
        The target rewritten relative to `NOTEBOOK_DIR`, or unchanged when it
        is absolute, external, or a bare anchor.
    """
    if re.match(r"^([a-z][a-z0-9+.-]*:|//|#)", target):
        return target
    resolved = posixpath.normpath(posixpath.join(PAGES_DIR, target))
    return posixpath.relpath(resolved, NOTEBOOK_DIR)


def rewrite_links(line):
    """Rewrite every relative link and image path on one line.

    Args:
        line (str): A line of page markdown.

    Returns:
        The line with page-relative targets pointed at `NOTEBOOK_DIR`.
    """
    return LINK.sub(
        lambda m: "%s[%s](%s)" % (m.group(1), m.group(2), rewrite_target(m.group(3))),
        line,
    )


def convert_prose(lines):
    """Turn page markdown into markdown a notebook renders.

    Drops the dark half of a light/dark image pair — a notebook has no theme to
    switch on, so rendering both would show the same plot twice. An admonition
    becomes a blockquote, its indented body dedented into the quote.

    Args:
        lines (list): The prose lines of one page section.

    Returns:
        The converted lines.
    """
    out = []
    in_admonition = False
    for line in lines:
        attr = IMAGE_ATTR.search(line)
        if attr:
            if attr.group(1) == "dark":
                continue
            line = IMAGE_ATTR.sub("", line).rstrip()

        admonition = ADMONITION.match(line)
        if admonition:
            kind, title = admonition.group(1), admonition.group(2)
            out.append("> **%s**%s" % (kind.capitalize(), " — " + title if title else ""))
            in_admonition = True
            continue

        if in_admonition:
            if not line.strip():
                out.append(">")
                continue
            if line.startswith("    "):
                out.append("> " + rewrite_links(line[4:]))
                continue
            in_admonition = False

        out.append(rewrite_links(line))
    return out


def cell(kind, source, index, slug):
    """Build one notebook cell.

    The id is derived from the page and the cell's position so that rebuilding
    an unchanged page produces byte-identical JSON.

    Args:
        kind (str): `"markdown"` or `"code"`.
        source (str): The cell's text, without a trailing newline.
        index (int): The cell's position in the notebook.
        slug (str): The page's filename stem, which seeds the id.

    Returns:
        The cell as a dict.
    """
    digest = hashlib.sha1(("%s:%d" % (slug, index)).encode("utf-8")).hexdigest()
    lines = source.split("\n")
    out = {
        "cell_type": kind,
        "id": digest[:12],
        "metadata": {},
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]],
    }
    if kind == "code":
        out["execution_count"] = None
        out["outputs"] = []
    return out


def build(path):
    """Convert one page into a notebook.

    Args:
        path (str): Path to the page, relative to the repository root.

    Returns:
        The notebook as a dict, ready for `json.dump`.
    """
    slug = os.path.basename(path)[: -len(".md")]
    lines = io.open(path, encoding="utf-8").read().split("\n")

    # No generated-from banner. That the page is the source of truth is written
    # down in this module's docstring; a cell repeating it inside the artefact
    # is one that people delete by hand, and `--check` would then report an
    # extra cell where there is no real difference.
    cells = []
    prose = []
    last_was_code = False

    def flush_prose():
        """Emit the accumulated prose as one markdown cell, if it has content."""
        body = "\n".join(convert_prose(prose)).strip("\n")
        del prose[:]
        if body:
            cells.append(cell("markdown", body, len(cells), slug))

    index = 0
    while index < len(lines):
        fence = FENCE.match(lines[index])
        if not fence:
            prose.append(lines[index])
            index += 1
            continue

        language = fence.group(1)
        close = index + 1
        while close < len(lines) and not FENCE.match(lines[close]):
            close += 1
        block = "\n".join(lines[index + 1 : close]).strip("\n")
        index = close + 1

        if language == "python":
            flush_prose()
            cells.append(cell("code", block, len(cells), slug))
            last_was_code = True
            continue

        # A `text` block straight after a code block is that block's recorded
        # output. One anywhere else — after prose, after another `text` block —
        # is a literal the page is showing, and stays in the prose.
        if language == "text" and last_was_code and not "".join(prose).strip():
            del prose[:]
            cells.append(
                cell(
                    "markdown",
                    "*Output recorded in the documentation:*\n\n```text\n%s\n```" % block,
                    len(cells),
                    slug,
                )
            )
            last_was_code = False
            continue

        prose.append("```%s" % language)
        prose.extend(block.split("\n"))
        prose.append("```")
        last_was_code = False

    flush_prose()

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": KERNELSPEC,
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render(notebook):
    """Serialise a notebook the way Jupyter writes one.

    Args:
        notebook (dict): The notebook.

    Returns:
        Its JSON text, with the trailing newline Jupyter leaves.
    """
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def signature(cells):
    """Reduce cells to what the page actually determines.

    Args:
        cells (list): The cells of a notebook.

    Returns:
        One `(cell_type, source)` pair per cell.
    """
    return [(c["cell_type"], "".join(c["source"])) for c in cells]


check = "--check" in sys.argv[1:]
pages = sorted(
    os.path.join(PAGES_DIR, name).replace("\\", "/")
    for name in os.listdir(PAGES_DIR)
    if name.endswith(".md")
)

if not check and not os.path.isdir(NOTEBOOK_DIR):
    os.makedirs(NOTEBOOK_DIR)

drift = []
written = 0
for page in pages:
    slug = os.path.basename(page)[: -len(".md")]
    path = "%s/%s.ipynb" % (NOTEBOOK_DIR, slug)
    target = build(page)
    text = render(target)
    current = io.open(path, encoding="utf-8").read() if os.path.exists(path) else None

    if text == current:
        note = "up to date"
    elif check:
        drift.append((path, current, target))
        continue
    else:
        io.open(path, "w", encoding="utf-8", newline="\n").write(text)
        written += 1
        note = "created" if current is None else "rebuilt"
    code = sum(1 for c in target["cells"] if c["cell_type"] == "code")
    print("%-46s %2d cells, %2d runnable  %s" % (path, len(target["cells"]), code, note))

print()
print("pages converted:", len(pages))
if check:
    for path, current, target in drift:
        if current is None:
            print("FAIL  %s does not exist; its page would generate one" % path)
            continue
        old, new = signature(json.loads(current)["cells"]), signature(target["cells"])
        if old == new:
            print(
                "FAIL  %s carries outputs or metadata its page does not generate" % path
            )
        elif len(old) != len(new):
            print(
                "FAIL  %s has %d cells, its page generates %d"
                % (path, len(old), len(new))
            )
        else:
            first = next(i for i in range(len(old)) if old[i] != new[i])
            print(
                "FAIL  %s: cell %d no longer matches the page (%s)"
                % (path, first, new[first][0])
            )
    if drift:
        print("FAILED:", len(drift))
        print("run: python tools/docs-checks/build_example_notebooks.py")
        sys.exit(1)
    print("every notebook matches its page")
else:
    print("notebooks written:", written)
