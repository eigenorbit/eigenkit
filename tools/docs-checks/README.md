# Documentation checkers

Eight checks and two tools that keep the documentation consistent with the code.
Every one of them **passes on the current tree**. Each catches a class of defect
that nothing else checks for, so CI runs them and a change that reintroduces one
fails the build.

Run any of them from the repository root:

```bash
python tools/docs-checks/check_func_params.py
```

| script | asserts | exit |
|---|---|---|
| `check_func_params.py` | every public class and function in a rendered module documents every parameter it takes, in an `Args:` / `Attributes:` section. Also lists modules with no `Example:` | prints a total; fails the build with `--strict` |
| `find_unfenced_code.py` | no code inside a docstring sits outside a ```` ```python ```` fence, and no markdown structure is broken — an RST heading underline, or one folded into its paragraph | non-zero on any hit |
| `check_rosters.py` | every roster in `docs/` and `README.md` equals its source: extras tables against `pyproject.toml`, provider aliases against the `register()` calls, the estimator list against `__all__`, and the layers named as taking `provider=` against the six layer signatures | non-zero on mismatch |
| `check_diagrams.py` | every class, member and field type named in a mermaid `classDiagram` exists in `src/` and matches its annotation; every `<b>name/</b>` box is a real package; every `Family*` shorthand prefixes a real export; every top-level package is named on the library map | non-zero on any mismatch |
| `check_doc_imports.py` | every page imports a name from where the convention says it lives: a name in the top-level `__all__` comes from `eigenkit`, a domain package's own names from that package | non-zero on any hit |
| `check_conventions.py` | the three conventions of the documentation: hyphens in page filenames, British `-ise` in prose (names and `optimization` exempt), and every chapter under `docs/algorithms/` ending on `## Next` | non-zero on any hit |
| `check_readme_cmds.py` | every shell command in `README.md` and `CONTRIBUTING.md` runs, or — for the ones that install, serve or publish — the tool accepts its subcommand and flags. Deferred lines are reported, not failed | non-zero on failure |
| `ast_equal.py` | the working tree differs from `HEAD` **only inside docstrings**: it strips every docstring and compares the AST of every file under `src/eigenkit/`. Run it after any change made only to docstrings | non-zero if code changed |

`rewrap2.py` is a tool, not a check: it re-wraps docstring paragraphs that a long
autorefs link or formula pushed past 88 columns. It only edits inside `ast`
docstring spans, treats links and `\(...\)` as unbreakable tokens, and refuses to
write anything if a docstring's whitespace-normalised token sequence changed.
Run it with `--apply`; without it, it prints what it would touch.

**Its guard is not complete.** A rewrap that folds an RST heading underline into
its paragraph keeps every token, so the guard passes, but the heading and its
dashes end up mid-sentence on the page. `find_unfenced_code.py` also checks
markdown structure and catches this, so run both after using the rewrapper.

`build_example_notebooks.py` is the other tool. It rebuilds
`examples/notebooks/*.ipynb` from the sixteen pages in `docs/examples/`, one
notebook per page, prose as markdown cells and every ```` ```python ```` block
as an empty code cell ready to run. The pages are the source of truth — they are
what the site publishes and what the checks above run against — so the notebooks
are generated from them and never edited by hand. A hand-written copy drifts
from its page as the library changes; a generated one cannot.

```bash
python tools/docs-checks/build_example_notebooks.py            # rebuild
python tools/docs-checks/build_example_notebooks.py --check    # drift only
```

Output is byte-deterministic, so a rebuild that changes nothing leaves the
working tree clean, and `--check` reports drift without writing, naming the
first cell that no longer matches.

The notebooks ship without outputs. The pages already show every result, and
committed outputs would be mostly training progress bars, several hundred
kilobytes per notebook, that go stale with each library change. So a rebuild
writes each notebook from scratch, and `--check` compares
the whole file: a notebook saved after running it fails just like one that has
drifted from its page.

`examples/notebooks/` is excluded from ruff (`extend-exclude` in `pyproject.toml`), and
has to stay excluded. Ruff formats the Python blocks of the pages themselves,
and the notebooks are rebuilt from the pages; formatting the copy separately
would only make the two disagree, and `ruff check --fix` would delete imports
that a cell does not use but the page still needs. Fix the page; the notebook
follows.

The `example-notebooks` hook in `.pre-commit-config.yaml` runs the rebuild, so a
commit that edits an example page rewrites its notebook, locally and in the
`lint` workflow.

Seven of the checks run in CI: `.github/workflows/docs-checks.yml`.
`ast_equal.py` does not — it compares the working tree against `HEAD`, which
asserts nothing when CI runs them as the same tree.

## Conventions these encode

- **Types go in the docstring, not the signature.** `griffe` demands a type once
  a parameter appears in `Args:`, and a `Returns:` on a function with no return
  annotation must use the `name (type):` form or every wrapped line counts as
  another returned value. Signatures are left alone.
- **`Example:` then a fenced ```` ```python ```` block, no blank line between.**
  It is the only form that both keeps the example on the page and highlights it.
  An indented block with no language renders flat; `Examples:` with doctest
  prompts disappears entirely.
- **Maths is maths.** An equation or operator expression uses `\(...\)`; an
  identifier, an array shape or a gate name like `PhaseShift(θ)` stays a code
  span. A docstring carrying LaTeX must be a raw string, or ruff rejects the
  escapes.
- **Links across objects, plain spans within one.** A reference to another class
  is `[`Name`][full.dotted.path]`; a reference to a method of the class being
  read is `` `name()` ``, because it is inches away on the same page.
