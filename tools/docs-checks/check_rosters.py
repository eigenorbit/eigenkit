# -*- coding: utf-8 -*-
"""Every roster in docs/ and README.md must match the code.

Sources of truth
  extras     pyproject.toml [project.optional-dependencies]
  providers  the register() / register_lazy() calls in core/backends/__init__.py
  estimators eigenkit.__all__
  run_torch  grep "def run_torch" src/
  provider=  the signatures of the six torch layers
"""

import io
import os
import re
import sys
import tomllib


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
fail = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  " + detail) if detail else ""))
    if not ok:
        fail.append(name)


# ---- extras ---------------------------------------------------------------
pyproject = tomllib.load(io.open("pyproject.toml", "rb"))
extras = set(pyproject["project"]["optional-dependencies"])
named = extras - {"all"}

for page in ("docs/getting-started.md", "README.md"):
    text = io.open(page, encoding="utf-8").read()
    table = re.findall(r"^\|\s*`([a-z]+)`\s*\|", text, re.M)
    listed = set(table)
    missing = sorted(named - listed)
    extra = sorted(listed - extras)
    check(
        "%s extras table == pyproject" % page,
        not missing and not extra,
        "missing=%s unexpected=%s" % (missing, extra),
    )
    check("%s has an `all` row" % page, "all" in listed)

# `all` must actually pull every named extra, except the ones kept out of it on
# purpose. perceval: its simulator dependency, exqalibur, pins protobuf exactly,
# which through `all` would hold protobuf back for every install.
OUTSIDE_ALL = {"perceval"}
all_spec = " ".join(pyproject["project"]["optional-dependencies"]["all"])
inside = set(re.findall(r"[\w-]+", all_spec.split("[", 1)[1].rsplit("]", 1)[0]))
check(
    "pyproject `all` covers every extra",
    not (named - inside - OUTSIDE_ALL),
    "missing=%s" % sorted(named - inside - OUTSIDE_ALL),
)

# ---- provider aliases ------------------------------------------------------
wiring = io.open("src/eigenkit/core/backends/__init__.py", encoding="utf-8").read()
aliases = set(re.findall(r"register(?:_lazy)?\(\s*\"([^\"]+)\"", wiring))
matrix = io.open("docs/user-guide/providers.md", encoding="utf-8").read()
rows = re.findall(r"^\|\s*[^|]+\|\s*[^|(]*\(`\"([^\"]+)\"`\)", matrix, re.M)
documented = set(rows)
# opt-in aliases carry no (`"name"`) cell; find them as inline code anywhere on the page
for alias in aliases - documented:
    if '"%s"' % alias in matrix:
        documented.add(alias)
missing = sorted(aliases - documented)
check(
    "providers.md names every registered alias",
    not missing,
    "missing=%s" % missing,
)

# ---- estimators ------------------------------------------------------------
init = io.open("src/eigenkit/__init__.py", encoding="utf-8").read()
all_names = set(re.findall(r"^\s*\"(\w+)\",", init, re.M))
estimators = {
    n
    for n in all_names
    if re.search(r"(Classifier|Regressor)$", n) or n in {"QSVC", "QSVR", "HQNC", "HQNR"}
}
estimators -= {
    "BaseQuantumClassifier",
    "BaseQuantumRegressor",
}  # bases, not shipped models
index = io.open("docs/index.md", encoding="utf-8").read()
bullet = re.search(
    r"- \*\*(?:sklearn|[Ss]cikit-learn) estimators\.?\*\*(.+?)\n\n", index, re.S
)
named_on_page = set(re.findall(r"`(\w+)`", bullet.group(1))) if bullet else set()
missing = sorted(estimators - named_on_page)
check(
    "index.md estimator list == __all__",
    not missing,
    "missing=%s" % missing,
)

# ---- run_torch -------------------------------------------------------------
defs = set()
for dp, dn, fns in os.walk("src/eigenkit"):
    dn[:] = [d for d in dn if d != "__pycache__"]
    for fn in fns:
        if fn.endswith(".py"):
            p = os.path.join(dp, fn)
            if "def run_torch" in io.open(p, encoding="utf-8").read():
                defs.add(fn[: -len("_backend.py")] if fn.endswith("_backend.py") else fn)
arch = io.open("docs/architecture-diagrams.md", encoding="utf-8").read()
claims_exclusive = "only one with run_torch" in arch
check(
    "no page claims a single run_torch backend",
    not claims_exclusive,
    "%d backends define it: %s" % (len(defs), sorted(defs)),
)

# ---- which layers take provider= -------------------------------------------
# The page names the layers that take provider= as a list, not as a count, so
# the claim can be compared against the signatures: a count such as "six
# layers" stays plausible even when some of them reject the argument.
layer_dir = "src/eigenkit/torch/layers"
accepts = set()
all_layers = set()
for fn in sorted(os.listdir(layer_dir)):
    if not fn.endswith(".py") or fn.startswith("_"):
        continue
    source = io.open(os.path.join(layer_dir, fn), encoding="utf-8").read()
    for cls, body in re.findall(
        r"^class (\w+)\(nn\.Module\):(.*?)(?=^class |\Z)", source, re.M | re.S
    ):
        all_layers.add(cls)
        init = re.search(r"def __init__\(\s*self,(.*?)\):", body, re.S)
        if init and re.search(r"^\s*provider\s*[:=]", init.group(1), re.M):
            accepts.add(cls)

page = io.open("docs/user-guide/layers.md", encoding="utf-8").read()
sentence = re.search(
    r"The (\w+) layers that build a circuit, (.+?), take the same `provider=`",
    page,
    re.S,
)
claimed = set(re.findall(r"`(\w+)`", sentence.group(2))) if sentence else set()
check(
    "layers.md names every layer that takes provider=",
    sentence is not None and claimed == accepts,
    "page=%s signatures=%s (silent: %s)"
    % (sorted(claimed), sorted(accepts), sorted(all_layers - accepts)),
)

print()
if fail:
    print("FAILED:", len(fail))
    sys.exit(1)
print("all rosters agree with the code")
