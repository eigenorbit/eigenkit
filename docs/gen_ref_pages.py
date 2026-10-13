"""Generate API reference pages at build time via mkdocs-gen-files."""

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

    with mkdocs_gen_files.open(doc_path, "w") as fd:
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
