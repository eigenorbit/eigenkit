"""Tests for the dataset cache system (_cache.py)."""

import hashlib
import io
import os
import ssl
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

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

    def fake_urlopen(url, context):
        return io.BytesIO(content)

    with patch.dict(os.environ, {"EIGENKIT_DATA_DIR": str(tmp_path)}):
        with patch("urllib.request.urlopen", side_effect=fake_urlopen) as mock_dl:
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
        with patch("urllib.request.urlopen") as mock_dl:
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

    def fake_urlopen(url, context):
        return io.BytesIO(content)

    with patch.dict(os.environ, {"EIGENKIT_DATA_DIR": str(tmp_path)}):
        with patch("urllib.request.urlopen", side_effect=fake_urlopen) as mock_dl:
            path = cached_download(url, filename, expected_hash=expected_hash)
            mock_dl.assert_called_once()
    assert path.read_bytes() == content


def test_cached_download_hash_verified(tmp_path):
    content = b"good data"
    url = "http://example.com/good.csv"
    filename = "good.csv"
    expected_hash = hashlib.sha256(content).hexdigest()

    def fake_urlopen(url, context):
        return io.BytesIO(content)

    with patch.dict(os.environ, {"EIGENKIT_DATA_DIR": str(tmp_path)}):
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            path = cached_download(url, filename, expected_hash=expected_hash)
    assert path.exists()


def test_certifi_roots_are_trusted_when_python_ships_none(tmp_path, monkeypatch):
    """The python.org build of Python for macOS has no CA certificates until
    `Install Certificates.command` runs, so OpenSSL's default paths are empty and
    every HTTPS download fails verification."""
    pytest.importorskip("certifi")
    monkeypatch.setenv("EIGENKIT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing.pem"))
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path / "missing"))
    contexts = []

    def fake_urlopen(url, context):
        contexts.append(context)
        return io.BytesIO(b"a,b\n1,2\n")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    cached_download("https://example.com/data.csv", "data.csv")
    assert contexts[0].cert_store_stats()["x509_ca"] > 0


def test_a_certificate_error_says_how_to_fix_it(tmp_path, monkeypatch):
    monkeypatch.setenv("EIGENKIT_DATA_DIR", str(tmp_path))

    def fake_urlopen(url, context):
        raise urllib.error.URLError(ssl.SSLCertVerificationError("verify failed"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(ssl.SSLCertVerificationError, match="Install Certificates") as e:
        cached_download("https://example.com/data.csv", "data.csv")
    assert str(e.value).startswith("Could not verify"), "printed as a tuple"
    assert "SSL_CERT_FILE" in str(e.value)
    assert list(tmp_path.iterdir()) == [], "a partial download was left cached"
