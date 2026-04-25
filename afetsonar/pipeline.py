"""AfetsonarPipeline — end-to-end disaster assessment pipeline.

Single entry point for the full AFETSONAR workflow:

    image → damage mask → buildings → priority → routing → map

Usage::

    from afetsonar import AfetsonarPipeline

    pipeline = AfetsonarPipeline(
        model_path="checkpoints/student/student_v1_best_ema.pth"
    )
    html_path = pipeline.generate_map(
        post_image="post_disaster.png",
        pre_image="pre_disaster.png",
        bbox=(41.003, 28.975, 41.008, 28.981),
        hospitals=[{"name": "Cerrahpaşa", "lat": 41.0048, "lon": 28.9510}],
        output_path="results/map.html",
    )
    print(f"Map saved to {html_path}")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

    Example:
        >>> pipeline = AfetsonarPipeline("checkpoints/student/student_v1_best_ema.pth")
        >>> mask = pipeline.predict("post.png", "pre.png")
        >>> mask.shape   # (H, W)  values 0-5
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[DefaultConfig] = None,
        device: str = "auto",
    ) -> None:
        self.config = config or DefaultConfig()

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self._load_model(model_path)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load a student or teacher checkpoint and return the model in eval mode."""
        from afetsonar.models import StudentSiameseSegformer

        checkpoint = torch.load(model_path, map_location=self.device)

        # Support both raw state-dict and wrapped dicts
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break

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
    # Image preprocessing
    # ------------------------------------------------------------------

    def _preprocess(
        self, post_path: str, pre_path: Optional[str] = None
    ) -> torch.Tensor:
        """Load and preprocess images into a model-ready tensor.

        Args:
            post_path: Path to the post-disaster image.
            pre_path: Path to the pre-disaster image (required for Siamese
                models; if ``None``, the post image is duplicated).

        Returns:
            Normalised 6-channel tensor ``(1, 6, H, W)`` on ``self.device``.
        """
        mean = np.array(self.config.__dict__.get(
            "imagenet_mean", [0.485, 0.456, 0.406]
        ), dtype=np.float32)
        std = np.array(self.config.__dict__.get(
            "imagenet_std", [0.229, 0.224, 0.225]
        ), dtype=np.float32)

        def _load(path: str) -> np.ndarray:
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        post = _load(post_path).astype(np.float32) / 255.0
        pre = _load(pre_path).astype(np.float32) / 255.0 if pre_path else post.copy()

        # Resize to model input size
        size = self.config.image_size
        if post.shape[:2] != (size, size):
            post = cv2.resize(post, (size, size))
        if pre.shape[:2] != (size, size):
            pre = cv2.resize(pre, (size, size))

        # ImageNet normalisation
        post = (post - mean) / std
        pre = (pre - mean) / std

        combined = np.concatenate([pre, post], axis=2)  # (H, W, 6)
        tensor = torch.from_numpy(combined).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        post_path: str,
        pre_path: Optional[str] = None,
    ) -> np.ndarray:
        """Run model inference on a single image pair.

        Args:
            post_path: Path to the post-disaster image.
            pre_path: Path to the pre-disaster image.

        Returns:
            Damage mask as ``np.ndarray`` of shape ``(H, W)`` with integer
            values 0–5.
        """
        tensor = self._preprocess(post_path, pre_path)
        outputs = self.model(tensor)
        logits = outputs["damage_logits"]
        if isinstance(logits, list):
            logits = logits[0]
        mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return mask

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
            bbox_latlon: ``(lat_min, lon_min, lat_max, lon_max)`` bounding box
                for geo-referencing pixel centroids.  If provided, ``lat``
                and ``lon`` keys are added to each building.

        Returns:
            List of building dicts with keys: ``building_id``,
            ``damage_class``, ``damage_class_name``, ``area_m2``,
            ``centroid_pixel``.  Geographic keys ``lat``, ``lon`` are added
            when ``bbox_latlon`` is provided.
        """
        px_m = pixel_size_m or self.config.pixel_size_m
        names = self.config.class_names
        buildings: List[Dict] = []
        bid = 0

        for cls in range(1, self.config.num_classes):
            binary = (mask == cls).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area_px = float(cv2.contourArea(cnt))
                if area_px < 50:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx_px = M["m10"] / M["m00"]
                cy_px = M["m01"] / M["m00"]
                area_m2 = area_px * (px_m ** 2)

                b: Dict[str, Any] = {
                    "building_id": bid,
                    "damage_class": cls,
                    "damage_class_name": names[cls] if cls < len(names) else f"class_{cls}",
                    "area_m2": area_m2,
                    "centroid_pixel": (cx_px, cy_px),
                }

                # Geo-reference
                if bbox_latlon is not None:
                    lat_min, lon_min, lat_max, lon_max = bbox_latlon
                    h, w = mask.shape
                    lat = lat_max - (cy_px / h) * (lat_max - lat_min)
                    lon = lon_min + (cx_px / w) * (lon_max - lon_min)
                    b["lat"] = float(lat)
                    b["lon"] = float(lon)

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
    ) -> Dict[str, Any]:
        """Run inference + building extraction + priority scoring.

        Args:
            post_path: Path to the post-disaster image.
            pre_path: Path to the pre-disaster image.
            bbox_latlon: Geographic bounding box for geo-referencing.

        Returns:
            Dict with keys ``"mask"``, ``"buildings"`` (list of building
            dicts with priority scores).
        """
        from afetsonar.routing.priority import score_buildings

        mask = self.predict(post_path, pre_path)
        buildings = self.mask_to_buildings(mask, bbox_latlon=bbox_latlon)
        buildings = score_buildings(buildings)
        return {"mask": mask, "buildings": buildings}

    def generate_map(
        self,
        post_path: str,
        pre_path: Optional[str],
        bbox_latlon: Tuple[float, float, float, float],
        hospitals: List[Dict[str, Any]],
        output_path: str = "afetsonar_map.html",
        n_teams: Optional[int] = None,
    ) -> str:
        """Full pipeline: image → interactive HTML map.

        Args:
            post_path: Path to the post-disaster image.
            pre_path: Path to the pre-disaster image.
            bbox_latlon: ``(lat_min, lon_min, lat_max, lon_max)``.
            hospitals: List of hospital dicts ``{"name", "lat", "lon"}``.
            output_path: Output HTML path.
            n_teams: Number of rescue teams.  Defaults to ``config.n_teams``.

        Returns:
            Absolute path to the saved HTML file.
        """
        from afetsonar.routing.team_assignment import assign_hospitals, assign_teams
        from afetsonar.geo.map_builder import FoliumMapBuilder

        n = n_teams or self.config.n_teams
        analysis = self.analyze(post_path, pre_path, bbox_latlon)
        buildings = analysis["buildings"]

        if not buildings:
            print("Warning: no buildings detected — generating empty map.")

        buildings, teams = assign_teams(buildings, n_teams=n)
        teams = assign_hospitals(teams, hospitals)

        lat_min, lon_min, lat_max, lon_max = bbox_latlon
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2

        builder = FoliumMapBuilder(center_lat, center_lon)
        builder.add_damage_markers(buildings)
        builder.add_hospitals(hospitals)
        html_path = builder.save(output_path)
        return html_path
