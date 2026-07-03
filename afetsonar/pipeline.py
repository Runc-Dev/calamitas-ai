"""AfetsonarPipeline — end-to-end disaster assessment pipeline.

Single entry point for the full AFETSONAR workflow:

    post image → [auto-fetch pre] → damage mask → buildings → priority → routing → map

Auto-fetch mode (Phase 2) — only a post image is required::

    from afetsonar import AfetsonarPipeline
    from afetsonar.geo.auto_fetch import AutoPreFetcher
    import os

    fetcher = AutoPreFetcher.from_env("google")   # reads GOOGLE_MAPS_KEY
    pipeline = AfetsonarPipeline(
        "checkpoints/student/student_v1_best_ema.pth",
        fetcher=fetcher,
    )

    # GPS from EXIF → auto-fetch pre → full pipeline
    html = pipeline.generate_map(
        post_path="drone_photo.jpg",          # EXIF GPS inside
        bbox_latlon=(41.003, 28.975, 41.008, 28.981),
        hospitals=[{"name": "Cerrahpaşa", "lat": 41.0048, "lon": 28.9510}],
        output_path="results/map.html",
    )

Manual pre-image mode (original)::

    html = pipeline.generate_map(
        post_path="post.png",
        pre_path="pre.png",
        bbox_latlon=...,
        hospitals=...,
    )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from afetsonar.config import DefaultConfig
from afetsonar.geo.utils import haversine_distance, pixel_to_geo


class AfetsonarPipeline:
    """End-to-end AFETSONAR inference and routing pipeline.

    Args:
        model_path: Path to a student or teacher checkpoint (``.pth``).
        config: Configuration object.  Defaults to :class:`DefaultConfig`.
        device: Torch device string (``"cuda"`` / ``"cpu"`` / ``"auto"``).
        fetcher: Optional :class:`~afetsonar.geo.auto_fetch.AutoPreFetcher`
            instance.  When provided, missing pre images are downloaded
            automatically from the configured satellite API.

    Note:
        Pre-image priority order:
        1. ``pre_path`` argument (explicit file path).
        2. Auto-fetch via ``fetcher`` using supplied or EXIF coordinates.
        3. Duplicate the post image (silent fallback when no coords/fetcher).
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[DefaultConfig] = None,
        device: str = "auto",
        fetcher: Optional[Any] = None,
    ) -> None:
        self.config = config or DefaultConfig()

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.fetcher = fetcher
        self.model = self._load_model(model_path)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load a student or teacher checkpoint and return the model in eval mode.

        Auto-detects model type from checkpoint tensor shapes:
        - patch_embeddings.0 with 64 channels → Teacher (SegFormer-B3)
        - patch_embeddings.0 with 32 channels → Student (SegFormer-B0)
        """
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break

        # Auto-detect model type from first patch embedding channel count
        probe_key = "encoder.patch_embeddings.0.proj.weight"
        is_teacher = (
            probe_key in state_dict and state_dict[probe_key].shape[0] == 64
        )

        if is_teacher:
            from afetsonar.models.teacher import SiameseTeacherSegformerV3
            print("[AfetsonarPipeline] Detected teacher checkpoint — loading SiameseTeacherSegformerV3")
            model = SiameseTeacherSegformerV3(
                num_damage_classes=self.config.num_classes,
                num_disaster_classes=self.config.num_disaster_classes,
                pretrained=False,
            )
        else:
            from afetsonar.models.student import StudentSiameseSegformer
            print("[AfetsonarPipeline] Detected student checkpoint — loading StudentSiameseSegformer")
            model = StudentSiameseSegformer(
                num_damage_classes=self.config.num_classes,
                num_disaster_classes=self.config.num_disaster_classes,
                pretrained=False,
            )

        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to(self.device)
        return model

    # ------------------------------------------------------------------
    # Image loading / preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _load_file(path: str) -> np.ndarray:
        """Load an image from disk as RGB uint8 numpy array."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _preprocess_arrays(
        self,
        post: np.ndarray,
        pre: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Normalise and stack pre+post arrays into a model tensor.

        Args:
            post: ``(H, W, 3)`` uint8 RGB post-disaster image.
            pre: ``(H, W, 3)`` uint8 RGB pre-disaster image, or ``None``
                to duplicate the post image.

        Returns:
            ``(1, 6, H, W)`` float32 tensor on ``self.device``.
        """
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        size = self.config.image_size
        post_f = cv2.resize(post, (size, size)).astype(np.float32) / 255.0
        pre_src = pre if pre is not None else post
        pre_f  = cv2.resize(pre_src, (size, size)).astype(np.float32) / 255.0

        post_n = (post_f - mean) / std
        pre_n  = (pre_f  - mean) / std

        combined = np.concatenate([pre_n, post_n], axis=2)  # (H, W, 6)
        tensor = torch.from_numpy(combined).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)

    def _resolve_pre(
        self,
        post_path: str,
        pre_path: Optional[str],
        lat: Optional[float],
        lon: Optional[float],
    ) -> Optional[np.ndarray]:
        """Determine and return the pre-disaster image as a numpy array.

        Resolution order:
        1. ``pre_path`` → load from disk.
        2. ``lat`` / ``lon`` + ``self.fetcher`` → auto-fetch.
        3. EXIF GPS in ``post_path`` + ``self.fetcher`` → auto-fetch.
        4. ``None`` → caller will use post image as fallback.

        Returns:
            ``(H, W, 3)`` uint8 RGB array, or ``None`` if unavailable.
        """
        # 1. Explicit pre file
        if pre_path is not None:
            return self._load_file(pre_path)

        # 2. No fetcher — nothing to do
        if self.fetcher is None:
            if lat is None and lon is None:
                return None
            print(
                "AfetsonarPipeline: lat/lon provided but no fetcher configured. "
                "Pass fetcher=AutoPreFetcher(...) to enable auto-fetch. "
                "Falling back to post image as pre."
            )
            return None

        # 3. Try supplied coordinates first
        if lat is not None and lon is not None:
            print(f"AfetsonarPipeline: auto-fetching pre image at ({lat:.5f}, {lon:.5f})...")
            return self.fetcher.fetch(lat, lon)

        # 4. Try EXIF GPS from the post image
        try:
            coords = self.fetcher.extract_gps(post_path)
        except Exception:
            coords = None

        if coords:
            print(
                f"AfetsonarPipeline: found EXIF GPS in post image "
                f"({coords['lat']:.5f}, {coords['lon']:.5f}). "
                "Auto-fetching pre image..."
            )
            return self.fetcher.fetch(coords["lat"], coords["lon"])

        print(
            "AfetsonarPipeline: no GPS coordinates found (no lat/lon argument "
            "and no EXIF GPS in post image). Falling back to post image as pre."
        )
        return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        post_path: str,
        pre_path: Optional[str] = None,
        *,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> np.ndarray:
        """Run model inference on a post-disaster image.

        Args:
            post_path: Path to the post-disaster image.
            pre_path: Path to the pre-disaster image.  If ``None``, the
                pipeline attempts auto-fetch (when a fetcher is configured)
                or falls back to duplicating the post image.
            lat: Latitude for auto-fetch (keyword-only).
            lon: Longitude for auto-fetch (keyword-only).

        Returns:
            Damage mask ``(H, W)`` uint8 with values 0–5.
        """
        post = self._load_file(post_path)
        pre  = self._resolve_pre(post_path, pre_path, lat, lon)

        tensor = self._preprocess_arrays(post, pre)
        outputs = self.model(tensor)
        logits = outputs["damage_logits"]
        if isinstance(logits, list):
            logits = logits[0]
        return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # ------------------------------------------------------------------
    # Building extraction
    # ------------------------------------------------------------------

    def mask_to_buildings(
        self,
        mask: np.ndarray,
        pixel_size_m: Optional[float] = None,
        bbox_latlon: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Convert a damage mask to a list of building feature dicts.

        Args:
            mask: ``(H, W)`` uint8 array with values 0–5.
            pixel_size_m: Metres per pixel (defaults to ``config.pixel_size_m``).
            bbox_latlon: ``(lat_min, lon_min, lat_max, lon_max)`` for
                geo-referencing pixel centroids.

        Returns:
            List of building dicts: ``building_id``, ``damage_class``,
            ``damage_class_name``, ``area_m2``, ``centroid_pixel``.
            Geographic keys ``lat`` and ``lon`` are added when
            ``bbox_latlon`` is provided.
        """
        px_m   = pixel_size_m or self.config.pixel_size_m
        names  = self.config.class_names
        buildings: List[Dict] = []
        bid = 0

        for cls in range(1, self.config.num_classes):
            binary = (mask == cls).astype(np.uint8)
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                area_px = float(cv2.contourArea(cnt))
                if area_px < 50:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx_px = M["m10"] / M["m00"]
                cy_px = M["m01"] / M["m00"]

                b: Dict[str, Any] = {
                    "building_id": bid,
                    "damage_class": cls,
                    "damage_class_name": names[cls] if cls < len(names) else f"class_{cls}",
                    "area_m2": area_px * (px_m ** 2),
                    "centroid_pixel": (cx_px, cy_px),
                }

                if bbox_latlon is not None:
                    lat_min, lon_min, lat_max, lon_max = bbox_latlon
                    h, w = mask.shape
                    b["lat"] = float(lat_max - (cy_px / h) * (lat_max - lat_min))
                    b["lon"] = float(lon_min + (cx_px / w) * (lon_max - lon_min))

                buildings.append(b)
                bid += 1

        return buildings

    # ------------------------------------------------------------------
    # High-level pipeline
    # ------------------------------------------------------------------

    def analyze(
        self,
        post_path: str,
        pre_path: Optional[str] = None,
        bbox_latlon: Optional[Tuple[float, float, float, float]] = None,
        *,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Inference + building extraction + priority scoring.

        Args:
            post_path: Post-disaster image path.
            pre_path: Pre-disaster image path (or ``None`` to auto-fetch).
            bbox_latlon: ``(lat_min, lon_min, lat_max, lon_max)``.
            lat: Latitude for auto-fetch (keyword-only).
            lon: Longitude for auto-fetch (keyword-only).

        Returns:
            Dict with keys ``"mask"`` and ``"buildings"``.
        """
        from afetsonar.routing.priority import score_buildings

        mask = self.predict(post_path, pre_path, lat=lat, lon=lon)
        buildings = self.mask_to_buildings(mask, bbox_latlon=bbox_latlon)
        buildings = score_buildings(buildings)
        return {"mask": mask, "buildings": buildings}

    def generate_map(
        self,
        post_path: str,
        bbox_latlon: Tuple[float, float, float, float],
        hospitals: List[Dict[str, Any]],
        pre_path: Optional[str] = None,
        output_path: str = "afetsonar_map.html",
        n_teams: Optional[int] = None,
        *,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> str:
        """Full pipeline: image → interactive HTML map.

        Args:
            post_path: Post-disaster image path.
            bbox_latlon: ``(lat_min, lon_min, lat_max, lon_max)``.
            hospitals: List of ``{"name", "lat", "lon"}`` dicts.
            pre_path: Pre-disaster image path (optional — auto-fetched if
                a fetcher is configured and coordinates are available).
            output_path: Destination HTML file path.
            n_teams: Number of rescue teams.  Defaults to ``config.n_teams``.
            lat: Latitude for auto-fetch (keyword-only).
            lon: Longitude for auto-fetch (keyword-only).

        Returns:
            Absolute path to the saved HTML file.
        """
        from afetsonar.routing.team_assignment import assign_hospitals, assign_teams
        from afetsonar.geo.map_builder import FoliumMapBuilder

        n = n_teams or self.config.n_teams
        analysis = self.analyze(
            post_path, pre_path, bbox_latlon, lat=lat, lon=lon
        )
        buildings = analysis["buildings"]

        if not buildings:
            print("Warning: no buildings detected in mask — generating empty map.")

        buildings, teams = assign_teams(buildings, n_teams=n)
        teams = assign_hospitals(teams, hospitals)

        lat_min, lon_min, lat_max, lon_max = bbox_latlon
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2

        builder = FoliumMapBuilder(center_lat, center_lon)
        builder.add_damage_markers(buildings)
        builder.add_hospitals(hospitals)
        return builder.save(output_path)
