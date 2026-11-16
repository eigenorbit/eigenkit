"""Generate API reference pages at build time via mkdocs-gen-files."""

import ast
import json
import re
from pathlib import Path

import mkdocs_gen_files


nav = mkdocs_gen_files.Nav()
pkg_root = Path("src") / "eigenkit"
# The `eigenkit.algorithms` package holds the standalone quantum
# algorithms (HHL, IQAE, portfolio risk). The sklearn-style estimators
# (QSVC, HQNC, generic/timeseries, …) live under `eigenkit.qml`
# and are documented from there.
algorithms_api_members = [
    "HHL",
    "HHLResult",
    "IQAE",
    "IQAEResult",
    "PortfolioRiskEstimator",
    "PortfolioRiskResult",
]
# Estimators that keep `save()` and `load()` in an estimator base class. Their
# curated member list is rendered with `inherited_members`, so the two reach the
# page instead of being dropped as "not defined here".
algorithms_members_with_persistence_inherited_members = {
    "PortfolioRiskEstimator",
    "CustomQuantumClassifier",
    "CustomQuantumRegressor",
    "GenericClassifier",
    "GenericRegressor",
    "HQNC",
    "HQNR",
    "QSVC",
    "QSVR",
    "QuantumTimeSeriesClassifier",
    "QuantumTimeSeriesRegressor",
    "RecurrentHQNC",
    "RecurrentHQNR",
}
# What each curated class renders, and in this order. `__init__` is never
# listed: a constructor's parameters are documented in the class docstring,
# which the class heading already renders, so a second entry would repeat the
# signature with no text of its own.
algorithm_public_members = {
    "HHL": ["solve"],
    "IQAE": ["solve"],
    "PortfolioRiskEstimator": [
        "fit",
        "var_at",
        # Required by the estimator base class; raises and points to var_at.
        "predict",
        "estimate",
        "save",
        "load",
    ],
    "QSVC": ["decision_function", "fit", "load", "predict", "predict_proba", "save"],
    "HQNC": ["fit", "load", "predict", "predict_proba", "save"],
    "QSVR": ["fit", "load", "predict", "save"],
    "HQNR": ["fit", "load", "predict", "save"],
    "RecurrentHQNC": ["fit", "load", "predict", "predict_proba", "save"],
    "RecurrentHQNR": ["fit", "load", "predict", "save"],
    "CustomQuantumClassifier": [
        "fit",
        "load",
        "predict",
        "predict_proba",
        "save",
    ],
    "CustomQuantumRegressor": ["fit", "load", "predict", "save"],
    "GenericClassifier": [
        "accepts_sequence_input",
        "fit",
        "load",
        "predict",
        "predict_proba",
        "save",
    ],
    "GenericRegressor": [
        "accepts_sequence_input",
        "fit",
        "load",
        "predict",
        "save",
    ],
    "QuantumTimeSeriesClassifier": [
        "output",
        "fit",
        "load",
        "predict",
        "predict_proba",
        "score",
        "save",
    ],
    "QuantumTimeSeriesRegressor": [
        "output",
        "fit",
        "load",
        "predict",
        "score",
        "save",
    ],
    "QAOA": ["solve"],
    "VQE": ["solve"],
}
# Modules whose page is written member by member rather than as one `:::`
# directive, so that the curated lists above apply. The estimators do not live
# under `eigenkit.algorithms` — that package holds the standalone
# solvers — so these ids name the packages the classes are actually defined in.
# Every module id is asserted against `src/` below: a moved module has to fail
# the build, because a key that matches nothing silently drops the curation.
algorithm_module_members = {
    "eigenkit.chemistry.vqe": [
        "VQE",
        "VQEResult",
        "hardware_efficient_ansatz",
    ],
    "eigenkit.optimization.qaoa": [
        "QAOA",
        "OptimizationResult",
        "qaoa_ansatz",
    ],
    "eigenkit.qml.qsvc": ["QSVC"],
    "eigenkit.qml.qsvr": ["QSVR"],
    "eigenkit.torch.estimators.custom": [
        "CustomQuantumClassifier",
        "CustomQuantumRegressor",
    ],
    "eigenkit.torch.estimators.generic": [
        "GenericRegressor",
        "GenericClassifier",
    ],
    "eigenkit.torch.estimators.hqnc": ["HQNC"],
    "eigenkit.torch.estimators.hqnr": ["HQNR"],
    "eigenkit.torch.estimators.recurrent_hqnc": ["RecurrentHQNC"],
    "eigenkit.torch.estimators.recurrent_hqnr": ["RecurrentHQNR"],
    "eigenkit.torch.estimators.timeseries": [
        "QuantumTimeSeriesRegressor",
        "QuantumTimeSeriesClassifier",
    ],
}
_absent = [
    module_id
    for module_id in algorithm_module_members
    if not (pkg_root / Path(*module_id.split(".")[1:])).with_suffix(".py").is_file()
]
if _absent:
    raise SystemExit(
        "gen_ref_pages: algorithm_module_members names modules that are not "
        "under src/: " + ", ".join(_absent) + ". Curation for them would be "
        "silently skipped — update the ids to where the classes now live."
    )


def write_member_directive(fd, module_id: str, member: str) -> None:
    fd.write(f"::: {module_id}.{member}\n")
    fd.write("    options:\n")
    # `mkdocs.yml` sets heading_level 3 so that a User Guide page nests its API
    # block under its own h2. Here the object *is* the page, so it opens on h2.
    fd.write("      heading_level: 2\n")
    public_members = algorithm_public_members.get(member)
    if public_members is not None:
        fd.write("      members:\n")
        for public_member in public_members:
            fd.write(f"        - {public_member}\n")
        fd.write("      show_if_no_docstring: true\n")
        if member in algorithms_members_with_persistence_inherited_members:
            fd.write("      inherited_members: true\n")
    fd.write("\n")


# What a search result shows for a page: its <title> and its meta description.
# Both are taken from the module docstring and written as front matter.
# `html_title` is read by `overrides/main.html`. Front matter `title` is not used:
# it would also replace the short module name in the header and in the site search.
TITLE_MAX = 70
TITLE_SUFFIX = " | eigenkit API"
DESCRIPTION_MAX = 160


def main_member_docstring(tree: ast.Module, stem: str) -> str:
    """The docstring of the member that a module with no docstring is named after."""
    members = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and not node.name.startswith("_")
    ]
    # `hqnc.py` is described by `HQNC`. Failing a member of the module's own name,
    # the first one that is not a result container (`HHLResult` before `HHL`).
    members.sort(
        key=lambda node: (
            node.name.lower() != stem.replace("_", ""),
            node.name.endswith("Result"),
        )
    )
    return (ast.get_docstring(members[0]) or "") if members else ""


def summary_sentences(py_file: Path) -> list[str]:
    """The opening sentences of the module docstring, as plain text."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    docstring = ast.get_docstring(tree) or main_member_docstring(tree, py_file.stem)
    prose = []
    for paragraph in docstring.split("\n\n"):
        # A list, a code fence or an `Example:` section ends the opening prose.
        if not re.match(r"[A-Za-z]|`[^`]", paragraph) or re.match(r"\w+:\n", paragraph):
            break
        prose.append(paragraph)
    # A link keeps its text.
    text = re.sub(r"\[([^\]]*)\](\[[^\]]*\]|\([^)]*\))", r"\1", " ".join(prose))
    sentences = []
    for sentence in re.split(
        r"(?<=[.!?])(?<![ei]\.[ge]\.)\s+(?=[A-Z`*])", " ".join(text.split())
    ):
        # A formula has no plain-text form, and a paragraph that ends on a colon
        # only introduces the list or the code block after it.
        if "\\(" in sentence or not sentence.endswith((".", "!", "?")):
            break
        # Code spans and emphasis lose their marks; an asterisk inside a code
        # span is code and stays.
        sentences.append(
            re.sub(r"(`+)(.+?)\1|\*+", lambda match: match.group(2) or "", sentence)
        )
    return sentences


def opening_clause(sentence: str, limit: int) -> str:
    """`sentence` if it fits in `limit` characters, else its opening clause."""
    # Up to the first comma, colon, semicolon or dash and, if that is still too
    # long, up to the first parenthesis. One word ("VQE", "Blocks") says too
    # little, so such a clause runs on to the next mark. A clause that is still too
    # long stays whole: cut inside a clause, a title reads as broken, and search
    # engines shorten a long title themselves.
    # A comma inside a list of names ("Harrow, Hassidim and Lloyd") is not a clause.
    for marks in (r"(?:,(?! \w+ and )|[:;]) | [—–] ", r" \("):
        cuts = [
            match.start()
            for match in re.finditer(marks, sentence)
            if " " in sentence[: match.start()]
        ]
        if len(sentence) > limit and cuts:
            sentence = sentence[: cuts[0]]
    return sentence


def search_title_and_description(py_file: Path, module_id: str) -> tuple[str, str]:
    sentences = summary_sentences(py_file)
    if not sentences:
        return (
            f"{module_id} API reference | eigenkit",
            f"API reference for {module_id} in eigenkit, an open-source Python "
            "library for quantum computing and quantum machine learning.",
        )
    name = module_id.rpartition(".")[2]
    # A summary that opens with the module's own name ("Blocks: reusable parts")
    # would say it twice.
    first = re.sub(rf"^{re.escape(name)}: ", "", sentences[0], flags=re.IGNORECASE)
    lead = opening_clause(first.rstrip("."), TITLE_MAX - len(f"{name}: {TITLE_SUFFIX}"))
    description = sentences[0]
    for sentence in sentences[1:]:
        if len(description) + 1 + len(sentence) > DESCRIPTION_MAX:
            break
        description += " " + sentence
    return f"{name}: {lead}{TITLE_SUFFIX}", description


for py_file in sorted(pkg_root.rglob("*.py")):
    rel = py_file.relative_to(pkg_root)
    parts = rel.parts  # e.g. ('layers', 'q.py') or ('_blocks', ...)

    # Skip private dirs/files — but __init__.py is handled specially below
    if any(p.startswith("_") and p != "__init__.py" for p in parts):
        continue

    if parts[-1] == "__init__.py":
        # Document the package itself; skip the top-level __init__
        if len(parts) == 1:
            continue
        stem_parts = parts[:-1]  # ('layers',) or ('datasets',)
        doc_path = Path("api", *stem_parts[:-1], stem_parts[-1] + ".md")
        nav_path = Path(*stem_parts[:-1], stem_parts[-1] + ".md")
        module_id = "eigenkit." + ".".join(stem_parts)
        nav_key = stem_parts
    else:
        stem_parts = parts[:-1] + (parts[-1][:-3],)  # ('layers', 'q')
        doc_path = Path("api", *parts[:-1], parts[-1][:-3] + ".md")
        nav_path = Path(*parts[:-1], parts[-1][:-3] + ".md")
        module_id = "eigenkit." + ".".join(stem_parts)
        nav_key = stem_parts

    nav[nav_key] = nav_path.as_posix()

    html_title, description = search_title_and_description(py_file, module_id)

    with mkdocs_gen_files.open(doc_path, "w") as fd:
        # A JSON string is a valid YAML scalar, so `json.dumps` does the quoting.
        fd.write("---\n")
        fd.write(f"html_title: {json.dumps(html_title, ensure_ascii=False)}\n")
        fd.write(f"description: {json.dumps(description, ensure_ascii=False)}\n")
        fd.write("---\n\n")
        if module_id == "eigenkit.algorithms":
            fd.write("# algorithms\n\n")
            for member in algorithms_api_members:
                write_member_directive(fd, module_id, member)
        elif module_id in algorithm_module_members:
            # The module itself first, with no members. A page written member by
            # member would otherwise drop the module's own docstring, and the
            # module heading is what every other page in the reference opens on.
            fd.write(f"::: {module_id}\n")
            fd.write("    options:\n")
            fd.write("      heading_level: 2\n")
            fd.write("      members: false\n\n")
            for member in algorithm_module_members[module_id]:
                write_member_directive(fd, module_id, member)
        else:
            fd.write(f"::: {module_id}\n")
            fd.write("    options:\n")
            fd.write("      heading_level: 2\n")

    mkdocs_gen_files.set_edit_path(doc_path, py_file)

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
