"""Tests for AutoPreFetcher — no real network calls, uses mocks."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from afetsonar.geo.auto_fetch import AutoPreFetcher


# ============================================================
# Helpers
# ============================================================

def _make_fake_png_bytes() -> bytes:
    """Return a minimal valid PNG file (1×1 red pixel) as bytes."""
    import io
    from PIL import Image
    img = Image.fromarray(np.array([[[200, 100, 50]]], dtype=np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================
# Construction tests
# ============================================================

class TestConstruction:
    def test_default_provider_is_google(self):
        f = AutoPreFetcher(api_key="test")
        assert f.provider == "google"

    def test_mapbox_provider_accepted(self):
        f = AutoPreFetcher(provider="mapbox", api_key="test")
        assert f.provider == "mapbox"

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError, match="provider must be"):
            AutoPreFetcher(provider="bing", api_key="test")

    def test_repr_shows_provider_and_key_status(self):
        f = AutoPreFetcher(provider="google", api_key="secret")
        r = repr(f)
        assert "google" in r
        assert "<set>" in r
        assert "secret" not in r  # key must NOT appear in repr

    def test_repr_no_key(self):
        f = AutoPreFetcher()
        assert "<not set>" in repr(f)

    def test_from_env_reads_google_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_MAPS_KEY", "env_key_123")
        f = AutoPreFetcher.from_env("google")
        assert f.api_key == "env_key_123"

    def test_from_env_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_MAPS_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GOOGLE_MAPS_KEY"):
            AutoPreFetcher.from_env("google")

    def test_from_env_mapbox(self, monkeypatch):
        monkeypatch.setenv("MAPBOX_TOKEN", "mb_token_456")
        f = AutoPreFetcher.from_env("mapbox")
        assert f.api_key == "mb_token_456"


# ============================================================
# URL building
# ============================================================

class TestURLBuilding:
    def test_google_url_contains_lat_lon(self):
        f = AutoPreFetcher(provider="google", api_key="TESTKEY")
        url = f._build_url(41.0, 28.9, 512, 17)
        assert "41.0" in url
        assert "28.9" in url
        assert "satellite" in url
        assert "TESTKEY" in url

    def test_mapbox_url_contains_lon_lat(self):
        # Mapbox format is lon,lat (reversed from Google)
        f = AutoPreFetcher(provider="mapbox", api_key="MB_TOKEN")
        url = f._build_url(41.0, 28.9, 512, 17)
        assert "28.9,41.0" in url  # lon first
        assert "MB_TOKEN" in url

    def test_google_url_contains_size(self):
        f = AutoPreFetcher(provider="google", api_key="K")
        url = f._build_url(0.0, 0.0, 768, 18)
        assert "768x768" in url

    def test_mapbox_url_contains_size(self):
        f = AutoPreFetcher(provider="mapbox", api_key="K")
        url = f._build_url(0.0, 0.0, 768, 18)
        assert "768x768" in url


# ============================================================
# Cache key
# ============================================================

class TestCacheKey:
    def test_same_coords_same_key(self):
        f = AutoPreFetcher(provider="google", api_key="K")
        k1 = f._cache_key(41.0, 28.9, 1024, 18)
        k2 = f._cache_key(41.0, 28.9, 1024, 18)
        assert k1 == k2

    def test_different_coords_different_key(self):
        f = AutoPreFetcher(provider="google", api_key="K")
        k1 = f._cache_key(41.0, 28.9, 1024, 18)
        k2 = f._cache_key(41.1, 28.9, 1024, 18)
        assert k1 != k2

    def test_different_providers_different_key(self):
        fg = AutoPreFetcher(provider="google", api_key="K")
        fm = AutoPreFetcher(provider="mapbox", api_key="K")
        assert fg._cache_key(41.0, 28.9, 1024, 18) != fm._cache_key(41.0, 28.9, 1024, 18)


# ============================================================
# Fetch with mocked HTTP
# ============================================================

class TestFetch:
    def test_fetch_no_key_raises(self):
        f = AutoPreFetcher()
        with pytest.raises(RuntimeError, match="no API key"):
            f.fetch(41.0, 28.9)

    def test_fetch_returns_rgb_array(self, monkeypatch):
        """fetch() should return (H, W, 3) uint8 RGB array."""
        fake_png = _make_fake_png_bytes()

        def fake_urlopen(req, timeout=None):
            mock_resp = MagicMock()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read.return_value = fake_png
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            f = AutoPreFetcher(provider="google", api_key="KEY", cache_dir=None)
            result = f.fetch(41.0, 28.9, size=64, zoom=18)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.ndim == 3
        assert result.shape[2] == 3     # RGB channels
        assert result.shape[0] == 64   # resized to requested size
        assert result.shape[1] == 64

    def test_fetch_uses_memory_cache(self, monkeypatch):
        """Second call with same coords must NOT hit network."""
        fake_png = _make_fake_png_bytes()
        call_count = {"n": 0}

        def fake_urlopen(req, timeout=None):
            call_count["n"] += 1
            mock_resp = MagicMock()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read.return_value = fake_png
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            f = AutoPreFetcher(provider="google", api_key="KEY", cache_dir=None)
            f.fetch(41.0, 28.9, size=64, zoom=18)
            f.fetch(41.0, 28.9, size=64, zoom=18)  # should use memory cache

        assert call_count["n"] == 1  # only one HTTP call

    def test_fetch_uses_disk_cache(self, tmp_path):
        """Second fetcher instance with same cache_dir should read from disk."""
        fake_png = _make_fake_png_bytes()
        call_count = {"n": 0}

        def fake_urlopen(req, timeout=None):
            call_count["n"] += 1
            mock_resp = MagicMock()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read.return_value = fake_png
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            f1 = AutoPreFetcher(provider="google", api_key="KEY", cache_dir=str(tmp_path))
            f1.fetch(41.0, 28.9, size=64, zoom=18)

            # New instance, same cache_dir — should NOT hit network
            f2 = AutoPreFetcher(provider="google", api_key="KEY", cache_dir=str(tmp_path))
            f2.fetch(41.0, 28.9, size=64, zoom=18)

        assert call_count["n"] == 1

    def test_fetch_http_error_raises_runtime(self, tmp_path):
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
            f = AutoPreFetcher(
                provider="google", api_key="KEY",
                cache_dir=str(tmp_path),  # unique dir — no cached files
            )
            with pytest.raises(RuntimeError, match="failed to download"):
                f.fetch(99.0, 99.0, size=64, zoom=18)  # unique coords


# ============================================================
# GPS extraction
# ============================================================

class TestExtractGPS:
    def test_no_exif_returns_none(self, tmp_path):
        """Image without EXIF GPS should return None (not raise)."""
        f = AutoPreFetcher(api_key="K")
        # Create a minimal PNG without any EXIF
        from PIL import Image
        img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        path = str(tmp_path / "no_exif.png")
        img.save(path)
        result = f.extract_gps(path)
        assert result is None

    def test_nonexistent_file_returns_none(self):
        f = AutoPreFetcher(api_key="K")
        result = f.extract_gps("/does/not/exist.jpg")
        assert result is None
