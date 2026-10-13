"""Tests for the dataset cache system (_cache.py)."""

import hashlib
import os
from pathlib import Path
from unittest.mock import patch

from eigenkit.datasets._cache import (
    DEFAULT_CACHE_DIR,
    cached_download,
    get_cache_dir,
)


def test_get_cache_dir_default():
    env_key = "EIGENKIT_DATA_DIR"
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop(env_key, None)
        result = get_cache_dir()
    assert result == DEFAULT_CACHE_DIR
    assert ".eigenkit" in str(result)
    assert "datasets" in str(result)


def test_get_cache_dir_env_var(tmp_path):
    custom = str(tmp_path / "custom_cache")
    with patch.dict(os.environ, {"EIGENKIT_DATA_DIR": custom}):
        result = get_cache_dir()
    assert result == Path(custom)


def test_cached_download_creates_file(tmp_path):
    content = b"hello dataset"
    url = "http://example.com/data.csv"
    filename = "data.csv"

    def fake_urlretrieve(url, dest):
        Path(dest).write_bytes(content)

    with patch.dict(os.environ, {"EIGENKIT_DATA_DIR": str(tmp_path)}):
        with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve) as mock_dl:
            path = cached_download(url, filename)
            assert path.exists()
            assert path.read_bytes() == content
            mock_dl.assert_called_once()


def test_cached_download_uses_cache(tmp_path):
    content = b"cached content"
    url = "http://example.com/data.csv"
    filename = "data.csv"

    # Pre-create the file to simulate a warm cache
    cached_file = tmp_path / filename
    cached_file.write_bytes(content)

    with patch.dict(os.environ, {"EIGENKIT_DATA_DIR": str(tmp_path)}):
        with patch("urllib.request.urlretrieve") as mock_dl:
            path = cached_download(url, filename)
            mock_dl.assert_not_called()
    assert path == cached_file


def test_cached_download_rehashes_corrupt(tmp_path):
    content = b"correct content"
    corrupt = b"corrupt content"
    url = "http://example.com/data.csv"
    filename = "data.csv"

    # Compute hash of the correct content
    expected_hash = hashlib.sha256(content).hexdigest()

    # Write corrupt file to cache
    cached_file = tmp_path / filename
    cached_file.write_bytes(corrupt)

    def fake_urlretrieve(url, dest):
        Path(dest).write_bytes(content)

    with patch.dict(os.environ, {"EIGENKIT_DATA_DIR": str(tmp_path)}):
        with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve) as mock_dl:
            path = cached_download(url, filename, expected_hash=expected_hash)
            mock_dl.assert_called_once()
    assert path.read_bytes() == content


def test_cached_download_hash_verified(tmp_path):
    content = b"good data"
    url = "http://example.com/good.csv"
    filename = "good.csv"
    expected_hash = hashlib.sha256(content).hexdigest()

    def fake_urlretrieve(url, dest):
        Path(dest).write_bytes(content)

    with patch.dict(os.environ, {"EIGENKIT_DATA_DIR": str(tmp_path)}):
        with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            path = cached_download(url, filename, expected_hash=expected_hash)
    assert path.exists()
