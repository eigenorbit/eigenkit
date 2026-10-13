# -*- coding: utf-8 -*-
"""The mermaid diagrams must name things that exist.

Three assertions, all of them read `src/` with `ast` — no import, no deps:

  classDiagram   every `class X { +type field }` resolves: the class exists in
                 src/, and every field, method and library type named inside it
                 exists on that class or on one of its bases
                 and a field's outermost type matches its real annotation
  flowchart      every `<b>name/</b>` package label is a real directory, and
                 every `Family*` shorthand in a label prefixes a real export
  completeness   every top-level package under src/ is named somewhere on
                 architecture-diagrams.md, whose stated job is to map them

The field-name half matters most: a diagram that names `Scale.expr` or
`Sum.terms`, where the real fields are `expression` and `expressions`, still
renders, and nothing else catches it.
"""

import ast
import io
import os
import re
import sys


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
SRC = "src/eigenkit"
MAP_PAGE = "docs/architecture-diagrams.md"

# types a diagram may name without them being library objects
BUILTIN_TYPES = {
    "int",
    "float",
    "str",
    "bool",
    "bytes",
    "complex",
    "dict",
    "list",
    "tuple",
    "set",
    "frozenset",
    "None",
    "Any",
    "ndarray",
    "object",
    "callable",
}

fail = []

# `__all__` of the top-level package, for the `Family*` shorthand in a label
EXPORTS = set(
    re.findall(
        r'^\s*"(\w+)",',
        io.open(os.path.join(SRC, "__init__.py"), encoding="utf-8").read(),
        re.M,
    )
)
FAMILY = re.compile(r"\b([A-Z][A-Za-z]+)\*")


def problem(page, message):
    print("FAIL  %s: %s" % (page, message))
    fail.append(message)


# ---- index every name src/ defines ---------------------------------------
classes = {}  # name -> (bases, members)
annotations = {}  # "Class.field" -> the annotation's source text
module_names = set()
for dirpath, dirnames, filenames in os.walk(SRC):
    dirnames[:] = [d for d in dirnames if d != "__pycache__"]
    for filename in sorted(filenames):
        if not filename.endswith(".py"):
            continue
        tree = ast.parse(
            io.open(os.path.join(dirpath, filename), encoding="utf-8").read()
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                members = set()
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        members.add(child.name)
                    elif isinstance(child, ast.AnnAssign) and isinstance(
                        child.target, ast.Name
                    ):
                        members.add(child.target.id)
                        annotations["%s.%s" % (node.name, child.target.id)] = ast.unparse(
                            child.annotation
                        )
                    elif isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name):
                                members.add(target.id)
                bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
                # a name defined twice (a Protocol and its implementation) merges
                old_bases, old_members = classes.get(node.name, ([], set()))
                classes[node.name] = (old_bases + bases, old_members | members)
                module_names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        module_names.add(target.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                module_names.add(node.name)


def members_of(name, seen=()):
    """Members of a class plus everything it inherits, within src/."""
    if name in seen or name not in classes:
        return set()
    bases, members = classes[name]
    out = set(members)
    for base in bases:
        out |= members_of(base, tuple(seen) + (name,))
    return out


# ---- walk the pages -------------------------------------------------------
FIELD = re.compile(r"^\s*[+\-#~]\s*(?:([\w~]+(?:~[\w~, ]+~)?)\s+)?([\w_]+)(\(.*\))?\s*$")
CLASS_OPEN = re.compile(r"^\s*class\s+([\w_]+)\s*\{\s*$")
TYPE_TOKEN = re.compile(r"[\w_]+")

pages = sorted(
    os.path.join(dp, fn)
    for dp, dn, fns in os.walk("docs")
    for fn in fns
    if fn.endswith(".md")
)
checked_classes = 0
for page in pages:
    text = io.open(page, encoding="utf-8").read()
    rel = page.replace("\\", "/")
    for block in re.findall(r"```mermaid\n(.*?)```", text, re.S):
        if "classDiagram" in block.split("\n", 1)[0] or block.lstrip().startswith(
            "classDiagram"
        ):
            current = None
            for line in block.split("\n"):
                opened = CLASS_OPEN.match(line)
                if opened:
                    current = opened.group(1)
                    checked_classes += 1
                    if current not in classes:
                        # a union alias (`Parameter`) is diagrammed as a class
                        # with a stereotype and has no members of its own
                        if current in module_names:
                            current = None
                        else:
                            problem(
                                rel,
                                "classDiagram names `%s`, which src/ does not define"
                                % current,
                            )
                            current = None
                    continue
                if current is None:
                    continue
                if line.strip() == "}":
                    current = None
                    continue
                if line.strip().startswith("<<"):
                    continue
                matched = FIELD.match(line)
                if not matched:
                    continue
                type_part, member, call = matched.groups()
                known = members_of(current)
                if member not in known:
                    problem(
                        rel,
                        "`%s.%s` is diagrammed, but %s has no such member"
                        % (current, member, current),
                    )
                if type_part:
                    tokens = TYPE_TOKEN.findall(type_part)
                    for token in tokens:
                        if token in BUILTIN_TYPES or token[0].islower():
                            continue
                        if token not in module_names:
                            problem(
                                rel,
                                "`%s.%s` is typed `%s`, which src/ does not define"
                                % (current, member, token),
                            )
                    # the outermost type has to appear in the real annotation.
                    # A diagram may simplify `tuple[tuple[str, int], ...]` to
                    # `tuple`, but it may not call it a `dict`.
                    real = annotations.get("%s.%s" % (current, member))
                    if real and tokens and tokens[0] not in TYPE_TOKEN.findall(real):
                        problem(
                            rel,
                            "`%s.%s` is diagrammed as `%s`, but it is annotated `%s`"
                            % (current, member, tokens[0], real),
                        )
    # `Family*` shorthand: at least one export has to start with it
    for block in re.findall(r"```mermaid\n(.*?)```", text, re.S):
        for family in sorted(set(FAMILY.findall(block))):
            if not any(name.startswith(family) for name in EXPORTS):
                problem(
                    rel,
                    "a label says `%s*`, but nothing exported starts with `%s`"
                    % (family, family),
                )

    # flowchart package labels, which may name a nested package (`backends/`)
    for label in set(re.findall(r"<b>([\w_]+)/</b>", text)):
        if (
            not any(label in dirnames for _, dirnames, _ in os.walk(SRC))
            and os.path.basename(SRC) != label
        ):
            problem(rel, "a box is labelled `%s/`, which is not a package" % label)

# ---- the map has to be a map of everything -------------------------------
page_text = io.open(MAP_PAGE, encoding="utf-8").read()
packages = sorted(
    name
    for name in os.listdir(SRC)
    if os.path.isdir(os.path.join(SRC, name)) and not name.startswith("_")
)
for name in packages:
    if ("%s/" % name) not in page_text:
        problem(
            MAP_PAGE.replace("\\", "/"),
            "package `%s/` is not named anywhere on the library map" % name,
        )

print()
print("classDiagram classes checked:", checked_classes)
print("packages under %s:" % SRC, ", ".join(packages))
if fail:
    print("FAILED:", len(fail))
    sys.exit(1)
print("every class, member, type and package named on a diagram exists")
