# -*- coding: utf-8 -*-
"""The three conventions of the documentation.

  filenames      a page under `docs/` is named with hyphens, never underscores
  spelling       British `-ise`, in prose only — a name is never respelled, and
                 `optimization` is a package, a page, an algorithm and three
                 cited paper titles, so it is exempt
  trailing head  a chapter under `docs/algorithms/` ends on `## Next`

The spelling half has to leave names alone: respelling
`eigenkit.qml._serialization` inside an autorefs target would break the
build, so code spans, fenced blocks, link targets and `:::` directives are cut
out before anything is matched.

Standard library only.
"""

import io
import os
import re
import sys


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)

PROTECTED = re.compile(
    # Code, code spans, link and autorefs targets, `:::` directives — and an
    # italicised span, which on references.md is a cited paper title. A title is
    # a name: respelling Shor's "Prime Factorization" would misquote it.
    r"(```.*?```|`[^`\n]*`|\]\([^)]*\)|\]\[[^\]]*\]|^:::.*$|(?<!\*)\*(?!\*)[^*]{1,300}\*(?!\*))",
    re.S | re.M,
)
AMERICAN = re.compile(r"\b[A-Za-z]+iz(?:e|es|ed|ing|ation|ations)\b")
# words where `-iz` is not the American spelling of `-is`, plus the one word
# that names things in this repository
EXEMPT = {
    "size",
    "sizes",
    "sized",
    "seize",
    "seizes",
    "prize",
    "prizes",
    "optimization",
    "figsize",
}

fail = []
pages = []
for dirpath, _dirnames, filenames in os.walk("docs"):
    if "api" in dirpath.replace("\\", "/").split("/"):
        continue
    for filename in sorted(filenames):
        if filename.endswith(".md"):
            pages.append(os.path.join(dirpath, filename).replace("\\", "/"))

for path in pages:
    base = os.path.basename(path)
    if "_" in base:
        fail.append(
            "%s is named with an underscore; documentation pages use hyphens" % path
        )

    text = io.open(path, encoding="utf-8").read()
    prose = "".join(
        part for part in PROTECTED.split(text) if part and not PROTECTED.fullmatch(part)
    )
    for word in AMERICAN.findall(prose):
        if word.lower() in EXEMPT:
            continue
        fail.append(
            "%s spells `%s`; the documentation uses the British -ise" % (path, word)
        )

for path in sorted(p for p in pages if p.startswith("docs/algorithms/")):
    if os.path.basename(path) in ("index.md", "references.md"):
        continue
    text = io.open(path, encoding="utf-8").read()
    if "\n## Next\n" not in text:
        fail.append("%s does not end on a `## Next` section" % path)

for message in fail:
    print("FAIL  " + message)

print()
print("pages checked:", len(pages))
if fail:
    print("FAILED:", len(fail))
    sys.exit(1)
print("filenames, spelling and trailing headings all agree")
