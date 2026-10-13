# -*- coding: utf-8 -*-
"""The pages use one import idiom.

The rule, written down in `docs/getting-started.md`: a name in the top-level
`__all__` is imported from `eigenkit` itself, and a name that belongs
to a domain package is imported from that package.

Reaching into a submodule for a top-level name reads as a plausible variant of
the same import, so a wrong one goes unnoticed until someone runs it:
`from eigenkit.algorithms import QSVC` looks right, but `QSVC` lives in `qml`,
and the line fails.

Static: `__all__` is parsed out of `__init__.py`, nothing is imported, so this
runs with no dependencies. It cannot tell whether a submodule import resolves;
that needs the package installed and is covered by executing the snippets.
"""

import ast
import io
import os
import re
import sys


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
SRC = "src/eigenkit"


def top_level_names():
    """`__all__` itself, not every quoted word in the file: a regex over
    `__init__.py` also picks up unrelated string lists, and an over-wide set
    would fail a page for importing a name the top level does not actually
    export."""
    tree = ast.parse(io.open(os.path.join(SRC, "__init__.py"), encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise SystemExit("check_doc_imports: no __all__ in %s/__init__.py" % SRC)


top_level = top_level_names()

FROM_SUB = re.compile(r"^from eigenkit\.([\w.]+) import (.+)$", re.M)
hits = []
pages = 0
for dirpath, _dirnames, filenames in os.walk("docs"):
    # the reference pages are generated, and name their own module by design
    if "api" in dirpath.replace("\\", "/").split("/"):
        continue
    for filename in sorted(filenames):
        if not filename.endswith(".md"):
            continue
        path = os.path.join(dirpath, filename).replace("\\", "/")
        pages += 1
        for lineno, line in enumerate(
            io.open(path, encoding="utf-8").read().split("\n"), 1
        ):
            matched = FROM_SUB.match(line)
            if not matched:
                continue
            module, names = matched.group(1), matched.group(2)
            for name in (n.strip() for n in names.split(",")):
                if name in top_level:
                    hits.append((path, lineno, module, name))

for path, lineno, module, name in hits:
    print(
        "FAIL  %s:%d imports `%s` from `eigenkit.%s`, but the top level "
        "exports it" % (path, lineno, name, module)
    )

print()
print("pages scanned:", pages)
print("top-level names in __all__:", len(top_level))
if hits:
    print("FAILED:", len(hits))
    sys.exit(1)
print("every page takes a top-level name from the top level")
