"""
Download-and-cache utility for remote datasets.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path


DEFAULT_CACHE_DIR = Path.home() / ".eigenkit" / "datasets"


def get_cache_dir() -> Path:
    """Return the dataset cache directory.

    Reads the ``EIGENKIT_DATA_DIR`` environment variable if set;
    otherwise falls back to ``~/.eigenkit/datasets``.

    Returns:
        Path to the cache directory (not guaranteed to exist yet).
    """
    return Path(os.environ.get("EIGENKIT_DATA_DIR", DEFAULT_CACHE_DIR))


def _file_hash(filepath: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def cached_download(
    url: str,
    filename: str,
    expected_hash: str | None = None,
) -> Path:
    """Download a file to the local cache and return its path.

    On subsequent calls the cached file is returned immediately unless the
    SHA-256 hash does not match *expected_hash* (indicating corruption or
    tampering), in which case the file is deleted and re-downloaded.

    Args:
        url: Remote URL to download from.
        filename: Local filename to store under the cache directory.
        expected_hash: Optional SHA-256 hex digest to verify the file. If
            ``None``, integrity checking is skipped.

    Returns:
        Path to the locally cached file.
    """
    cache_dir = get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    # `cache_dir / filename` follows "../" out of the cache, so a registry entry
    # or caller-supplied name could write anywhere the process can. The cache
    # stores flat files, so a name with any path structure in it is a mistake.
    if filename != Path(filename).name or filename in ("", ".", ".."):
        raise ValueError(
            f"filename must be a bare file name with no path separators; "
            f"got {filename!r}."
        )
    filepath = cache_dir / filename

    if filepath.exists():
        if expected_hash is not None and _file_hash(filepath) != expected_hash:
            filepath.unlink()
        else:
            return filepath

    # Download beside the target and rename into place. `urlretrieve` writes
    # directly to its destination, so an interrupted download -- a dropped
    # connection, a Ctrl-C -- would leave a truncated file at the cache path
    # that every later call would return as valid, forever, unless the registry
    # entry carries a hash to check it against.
    with tempfile.NamedTemporaryFile(
        dir=cache_dir, prefix=f".{filename}.", suffix=".part", delete=False
    ) as handle:
        partial = Path(handle.name)
    try:
        urllib.request.urlretrieve(url, partial)  # noqa: S310
        if expected_hash is not None:
            actual = _file_hash(partial)
            if actual != expected_hash:
                raise ValueError(
                    f"Hash mismatch for {filename!r}: expected {expected_hash}, "
                    f"got {actual}"
                )
        partial.replace(filepath)
    finally:
        partial.unlink(missing_ok=True)

    return filepath
