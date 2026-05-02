"""Automatic pre-disaster satellite image fetcher.

When only a post-disaster image is available, ``AutoPreFetcher`` downloads
the corresponding pre-disaster satellite view from Google Maps Static API or
Mapbox Static API using the image's GPS coordinates.

Usage::

    from afetsonar.geo.auto_fetch import AutoPreFetcher

    fetcher = AutoPreFetcher(provider="google", api_key="YOUR_KEY")

    # Option A: provide coordinates directly
    pre_image = fetcher.fetch(lat=41.0082, lon=28.9784)

    # Option B: read GPS from the post-image's EXIF metadata
    coords = fetcher.extract_gps("post_disaster.jpg")
    if coords:
        pre_image = fetcher.fetch(**coords)

References
----------
- Google Maps Static API: https://developers.google.com/maps/documentation/maps-static
- Mapbox Static Images API: https://docs.mapbox.com/api/maps/static-images/
"""

from __future__ import annotations

import hashlib
import io
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple


class AutoPreFetcher:
    """Downloads pre-disaster satellite imagery for a given GPS location.

    Supported providers:
    - ``"google"``  — Google Maps Static API (requires an API key with
      Maps Static API enabled).
    - ``"mapbox"``  — Mapbox Static Images API (requires a Mapbox access
      token).

    Args:
        provider: Imagery provider (``"google"`` or ``"mapbox"``).
        api_key: API key / access token for the chosen provider.
        cache_dir: Directory for caching downloaded images.  Defaults to
            a ``afetsonar_cache`` sub-directory inside the system's temp
            folder.  Set to ``None`` to disable disk caching.
        default_zoom: Default map zoom level (0–21).  Zoom 18 gives
            ~0.6 m/pixel imagery at equatorial latitudes — suitable for
            building-level analysis.
        default_size: Downloaded image side length in pixels.

    Example::

        fetcher = AutoPreFetcher("mapbox", api_key=os.environ["MAPBOX_TOKEN"])
        pre = fetcher.fetch(lat=41.0082, lon=28.9784)
        print(pre.shape)   # (1024, 1024, 3)  uint8 RGB
    """

    SUPPORTED_PROVIDERS = ("google", "mapbox")
    _DEFAULT_CACHE = object()  # sentinel: "use system temp dir"

    def __init__(
        self,
        provider: str = "google",
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = _DEFAULT_CACHE,  # type: ignore[assignment]
        default_zoom: int = 18,
        default_size: int = 1024,
    ) -> None:
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"provider must be one of {self.SUPPORTED_PROVIDERS}, got {provider!r}"
            )
        self.provider = provider
        self.api_key = api_key

        # cache_dir=None  → no disk cache
        # cache_dir=<default sentinel> → use system temp dir
        # cache_dir="path" → use that path
        if cache_dir is AutoPreFetcher._DEFAULT_CACHE:
            import tempfile
            resolved: Optional[str] = os.path.join(
                tempfile.gettempdir(), "afetsonar_cache"
            )
        else:
            resolved = cache_dir  # type: ignore[assignment]

        self.cache_dir = Path(resolved) if resolved else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.default_zoom = default_zoom
        self.default_size = default_size
        self._memory_cache: Dict[str, "np.ndarray"] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        lat: float,
        lon: float,
        size: Optional[int] = None,
        zoom: Optional[int] = None,
    ) -> "np.ndarray":
        """Download a satellite image centred on ``(lat, lon)``.

        Args:
            lat: Latitude in decimal degrees (WGS84).
            lon: Longitude in decimal degrees (WGS84).
            size: Output image side in pixels.  Defaults to
                ``self.default_size``.
            zoom: Map zoom level.  Defaults to ``self.default_zoom``.

        Returns:
            ``(H, W, 3)`` uint8 RGB numpy array ready for model input.

        Raises:
            RuntimeError: If no API key is configured.
            RuntimeError: If the HTTP request fails.
        """
        import numpy as np

        if not self.api_key:
            raise RuntimeError(
                f"AutoPreFetcher: no API key set for provider '{self.provider}'.\n"
                "Pass api_key= when constructing AutoPreFetcher, or set "
                "GOOGLE_MAPS_KEY / MAPBOX_TOKEN environment variables and "
                "use AutoPreFetcher.from_env()."
            )

        size = size or self.default_size
        zoom = zoom or self.default_zoom
        key = self._cache_key(lat, lon, size, zoom)

        # 1. Memory cache
        if key in self._memory_cache:
            return self._memory_cache[key].copy()

        # 2. Disk cache
        if self.cache_dir:
            cached_path = self.cache_dir / f"{key}.png"
            if cached_path.exists():
                img = self._load_image_file(str(cached_path))
                self._memory_cache[key] = img
                return img.copy()

        # 3. Download
        url = self._build_url(lat, lon, size, zoom)
        img = self._download(url, size)

        # Save to disk cache
        if self.cache_dir:
            self._save_image_file(img, str(self.cache_dir / f"{key}.png"))

        self._memory_cache[key] = img
        return img.copy()

    def extract_gps(self, image_path: str) -> Optional[Dict[str, float]]:
        """Read GPS coordinates from a drone image's EXIF metadata.

        Args:
            image_path: Path to a JPEG or TIFF image with GPS EXIF data.

        Returns:
            Dict with keys ``"lat"`` and ``"lon"`` (decimal degrees), or
            ``None`` if no GPS data is found.

        Note:
            Requires the ``exifread`` package (``pip install exifread``).
        """
        from afetsonar.geo.utils import read_exif_gps

        result = read_exif_gps(image_path)
        if result is None:
            return None
        return {"lat": result["latitude"], "lon": result["longitude"]}

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        provider: str = "google",
        **kwargs,
    ) -> "AutoPreFetcher":
        """Construct from environment variables.

        Reads:
        - ``GOOGLE_MAPS_KEY`` for Google provider.
        - ``MAPBOX_TOKEN`` for Mapbox provider.

        Args:
            provider: ``"google"`` or ``"mapbox"``.
            **kwargs: Forwarded to :class:`AutoPreFetcher`.

        Returns:
            Configured :class:`AutoPreFetcher` instance.

        Raises:
            RuntimeError: If the expected environment variable is not set.
        """
        env_var = {
            "google": "GOOGLE_MAPS_KEY",
            "mapbox": "MAPBOX_TOKEN",
        }[provider]
        key = os.environ.get(env_var)
        if not key:
            raise RuntimeError(
                f"Environment variable {env_var} is not set.\n"
                f"Set it with: export {env_var}=your_key_here"
            )
        return cls(provider=provider, api_key=key, **kwargs)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_url(self, lat: float, lon: float, size: int, zoom: int) -> str:
        """Build the API request URL (no key embedded in log-safe repr)."""
        if self.provider == "google":
            # Google Maps Static API
            # Docs: https://developers.google.com/maps/documentation/maps-static/start
            base = "https://maps.googleapis.com/maps/api/staticmap"
            return (
                f"{base}?center={lat},{lon}&zoom={zoom}"
                f"&size={size}x{size}&maptype=satellite&key={self.api_key}"
            )
        else:  # mapbox
            # Mapbox Static Images API
            # Docs: https://docs.mapbox.com/api/maps/static-images/
            base = "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static"
            return (
                f"{base}/{lon},{lat},{zoom}/{size}x{size}"
                f"?access_token={self.api_key}"
            )

    def _download(self, url: str, size: int) -> "np.ndarray":
        """Fetch image bytes from URL and decode to RGB numpy array."""
        import numpy as np
        import urllib.request

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "AFETSONAR/1.0 (disaster response research)"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
        except Exception as exc:
            raise RuntimeError(
                f"AutoPreFetcher: failed to download image from {self.provider}. "
                f"Check your API key and network connection.\nCause: {exc}"
            ) from exc

        img = self._decode_image_bytes(raw)
        if img is None:
            raise RuntimeError(
                "AutoPreFetcher: received response but could not decode image. "
                "The API key may be invalid or quota exceeded."
            )

        # Resize to target size (API may return slightly different dimensions)
        if img.shape[:2] != (size, size):
            img = self._resize(img, size)

        return img

    def _decode_image_bytes(self, raw: bytes) -> Optional["np.ndarray"]:
        """Decode raw bytes → RGB uint8 numpy array."""
        try:
            import cv2
            buf = _np_frombuffer(raw, dtype="uint8")
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except ImportError:
            pass

        # Fallback: use PIL
        try:
            from PIL import Image
            import numpy as np
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            return np.array(img, dtype=np.uint8)
        except Exception:
            pass

        return None

    def _resize(self, img: "np.ndarray", size: int) -> "np.ndarray":
        """Resize image to (size × size)."""
        try:
            import cv2
            return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            pass
        from PIL import Image
        import numpy as np
        pil = Image.fromarray(img).resize((size, size), Image.BILINEAR)
        return np.array(pil, dtype=np.uint8)

    def _load_image_file(self, path: str) -> "np.ndarray":
        """Load a cached PNG file as RGB array."""
        try:
            import cv2
            img = cv2.imread(path)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except ImportError:
            pass
        from PIL import Image
        import numpy as np
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

    def _save_image_file(self, img: "np.ndarray", path: str) -> None:
        """Save RGB array as PNG file."""
        try:
            import cv2
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return
        except ImportError:
            pass
        from PIL import Image
        Image.fromarray(img).save(path)

    def _cache_key(self, lat: float, lon: float, size: int, zoom: int) -> str:
        """Deterministic cache key for a (provider, lat, lon, size, zoom) tuple."""
        payload = f"{self.provider}_{lat:.6f}_{lon:.6f}_{size}_{zoom}"
        return hashlib.md5(payload.encode()).hexdigest()[:16]

    def __repr__(self) -> str:
        has_key = bool(self.api_key)
        return (
            f"AutoPreFetcher(provider={self.provider!r}, "
            f"api_key={'<set>' if has_key else '<not set>'}, "
            f"zoom={self.default_zoom}, size={self.default_size})"
        )


def _np_frombuffer(raw: bytes, dtype: str = "uint8") -> "np.ndarray":
    """numpy.frombuffer wrapper — avoids top-level numpy import."""
    import numpy as np
    return np.frombuffer(raw, dtype=dtype)
