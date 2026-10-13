"""`eigenkit.__version__` exists, is accurate, and stays cheap.

It is the first thing a bug report reaches for and the first thing a
compatibility check tests: it is how the version in `pyproject.toml` becomes
observable at runtime.

It is resolved from the installed distribution metadata through the module-level
``__getattr__``, not assigned at import time. That is not a stylistic choice —
importing `importlib.metadata` costs about as much again as importing the whole
core, and the package docstring promises a cheap import. The subprocess test
below is what holds that promise in place; the rest pin the access patterns
callers actually use, all of which route through PEP 562.
"""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
import textwrap

import eigenkit


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
    )


def test_version_exists_and_matches_distribution_metadata():
    assert eigenkit.__version__ == importlib.metadata.version("eigenkit")


def test_version_is_a_dotted_release_string():
    parts = eigenkit.__version__.split(".")
    assert len(parts) >= 2, eigenkit.__version__
    assert parts[0].isdigit() and parts[1].isdigit(), eigenkit.__version__


def test_the_usual_access_patterns_all_work():
    # `hasattr` is what a compatibility check calls; `from ... import` is what a
    # downstream package writes. Both go through the module `__getattr__`.
    result = _run(
        """
        import eigenkit
        from eigenkit import __version__ as imported

        assert hasattr(eigenkit, "__version__")
        assert eigenkit.__version__ == imported
        assert getattr(eigenkit, "__version__", None) == imported
        """
    )
    assert result.returncode == 0, result.stderr


def test_importing_the_package_does_not_load_importlib_metadata():
    # A fresh interpreter: the current session has imported it long ago.
    result = _run(
        """
        import sys
        import eigenkit

        assert "importlib.metadata" not in sys.modules, (
            "importing eigenkit pulled importlib.metadata"
        )
        _ = eigenkit.__version__
        assert "importlib.metadata" in sys.modules
        """
    )
    assert result.returncode == 0, result.stderr


def test_the_first_lookup_caches_into_the_module_globals():
    result = _run(
        """
        import eigenkit

        assert "__version__" not in vars(eigenkit)
        first = eigenkit.__version__
        assert vars(eigenkit)["__version__"] == first
        assert "__version__" in dir(eigenkit)
        """
    )
    assert result.returncode == 0, result.stderr


def test_unknown_dunders_still_raise_attribute_error():
    # The version branch must not swallow every dunder that misses the dict.
    result = _run(
        """
        import eigenkit

        try:
            eigenkit.__release__
        except AttributeError as exc:
            assert "__release__" in str(exc), exc
        else:
            raise AssertionError("expected AttributeError")
        """
    )
    assert result.returncode == 0, result.stderr


def test_without_distribution_metadata_the_attribute_is_absent_not_an_error():
    """An uninstalled source checkout has no metadata to read.

    `PackageNotFoundError` is not an `AttributeError`, so letting it out of
    `__getattr__` would break the two calls the tests above pin: `hasattr` would
    raise instead of returning False, and `getattr(pkg, "__version__", default)`
    would raise instead of returning the default.
    """
    result = _run(
        """
        import importlib.metadata as md

        real = md.version

        def missing(name):
            if name == "eigenkit":
                raise md.PackageNotFoundError(name)
            return real(name)

        md.version = missing

        import eigenkit

        assert not hasattr(eigenkit, "__version__")
        assert getattr(eigenkit, "__version__", "unknown") == "unknown"
        try:
            eigenkit.__version__
        except AttributeError as exc:
            assert "no distribution metadata" in str(exc), exc
        else:
            raise AssertionError("expected AttributeError")
        """
    )
    assert result.returncode == 0, result.stderr
