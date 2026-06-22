"""
SSIM-based structural-change filter for the Edge Triage Layer.

Compares consecutive frames within each grid window using a lightweight
approximation of the Structural Similarity Index (SSIM).  Grids whose
visual-change score falls below a configurable threshold are flagged as
"static" so that the calling pipeline can skip costly YOLO inference
and VLM escalation for that cell.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from src.core_agent.structured_parser import GridPosition

logger = logging.getLogger("edgeops.edge_triage.ssim_filter")


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

DEFAULT_SSIM_THRESHOLD: float = 0.035
"""Grids whose delta score is below this threshold are considered static.

Lower values = more sensitive (fewer skips).  Range 0.0 – 0.15 typical.
"""

K1: float = 0.01
K2: float = 0.03
"""SSIM stabilisation constants (standard values from the literature)."""

L: int = 255
"""Dynamic range for uint8 pixel values."""


# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────


@dataclass
class GridChangeScores:
    """Per-grid structural change scores for the current frame pair."""

    scores: dict[GridPosition, float] = field(default_factory=dict)

    def should_skip(self, grid: GridPosition, threshold: float | None = None) -> bool:
        """Return True if the grid is static enough to skip downstream inference.

        Args:
            grid:   Target grid.
            threshold:   Override threshold; uses ``DEFAULT_SSIM_THRESHOLD`` if None.

        Returns:
            True when the delta score < threshold (i.e. the grid is static).
        """
        t = threshold if threshold is not None else DEFAULT_SSIM_THRESHOLD
        return self.scores.get(grid, 1.0) < t

    def active_grids(
        self, threshold: float | None = None
    ) -> set[GridPosition]:
        """Return the subset of grids whose change score exceeds the threshold.

        These grids have enough motion / structural change to warrant
        re-running the detector.
        """
        t = threshold if threshold is not None else DEFAULT_SSIM_THRESHOLD
        return {g for g, s in self.scores.items() if s >= t}


# ──────────────────────────────────────────────────────────────────────
# SSIM filter implementation
# ──────────────────────────────────────────────────────────────────────


class SSIMMotionFilter:
    """Lightweight frame-difference filter using a simplified SSIM metric.

    The filter stores the previous frame for each grid and computes a
    single-scale SSIM approximation on the luminance channel every time
    ``update()`` is called.  Grids whose change score is below threshold
    are classified as static.

    Usage:
        ssim = SSIMMotionFilter(threshold=0.05)
        for frame_batch in stream:
            scores = ssim.update(frame_batch)   # dict[GridPosition, float]
            active = scores.active_grids()       # grids with motion
            for g in active:
                run_yolo(frame_batch[g])
    """

    def __init__(self, threshold: float = DEFAULT_SSIM_THRESHOLD) -> None:
        self.threshold = threshold
        self._prev: dict[GridPosition, np.ndarray] = {}
        self._frame_count: int = 0
        logger.info(
            "SSIMMotionFilter initialised (threshold=%.4f)", threshold
        )

    # ── Public API ────────────────────────────────────────────────

    def update(
        self,
        grids: dict[GridPosition, np.ndarray],
    ) -> GridChangeScores:
        """Feed the latest grid crops and retrieve per-grid change scores.

        Args:
            grids:   Mapping of GridPosition → cropped frame (H, W, 3) uint8.

        Returns:
            GridChangeScores with a delta for every grid in *grids*.
            Grids seen for the first time receive a score of 1.0 (always
            considered active).

        Raises:
            ValueError: If any grid frame is not uint8 or has < 2 dims.
        """
        self._frame_count += 1
        result = GridChangeScores()

        for grid, curr_frame in grids.items():
            self._validate_frame(curr_frame, grid)

            if grid not in self._prev:
                result.scores[grid] = 1.0
            else:
                prev_frame = self._prev[grid]
                score = self._compute_delta(prev_frame, curr_frame)
                result.scores[grid] = score

            self._prev[grid] = curr_frame

        logger.debug(
            "Frame %d | scores: %s",
            self._frame_count,
            {g.value: round(s, 4) for g, s in result.scores.items()},
        )
        return result

    def reset(self) -> None:
        """Clear the previous-frame cache (e.g. after a scene cut)."""
        self._prev.clear()
        self._frame_count = 0
        logger.info("SSIMMotionFilter cache reset")

    # ── Internal helpers ──────────────────────────────────────────

    @staticmethod
    def _validate_frame(frame: np.ndarray, grid: GridPosition) -> None:
        if frame.dtype != np.uint8:
            raise ValueError(
                f"{grid.value}: expected uint8 frame, got {frame.dtype}"
            )
        if frame.ndim < 2:
            raise ValueError(
                f"{grid.value}: frame has {frame.ndim} dims, need >= 2"
            )

    @classmethod
    def _compute_delta(cls, prev: np.ndarray, curr: np.ndarray) -> float:
        """Compute 1 - SSIM(prev, curr) as a proxy for visual change.

        Returns a value in [0.0, 1.0] where:
          - 0.0  →  identical frames (no change)
          - 1.0  →  completely different (maximum change)

        The SSIM is computed on the luminance (grayscale) channel using
        a single-scale, single-window approximation suitable for edge
        devices.
        """
        if prev.shape != curr.shape:
            return 1.0

        gray_prev = cls._to_luminance(prev)
        gray_curr = cls._to_luminance(curr)

        c1 = (K1 * L) ** 2
        c2 = (K2 * L) ** 2

        mu_x = gray_prev.mean()
        mu_y = gray_curr.mean()
        sigma_x = gray_prev.var()
        sigma_y = gray_curr.var()
        sigma_xy = ((gray_prev - mu_x) * (gray_curr - mu_y)).mean()

        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)

        ssim = numerator / denominator if denominator != 0 else 0.0
        ssim = np.clip(ssim, 0.0, 1.0)

        return float(1.0 - ssim)

    @staticmethod
    def _to_luminance(frame: np.ndarray) -> np.ndarray:
        """Convert RGB uint8 → grayscale float64 in [0, L]."""
        if frame.ndim == 2:
            return frame.astype(np.float64)
        # Standard luminance weights (ITU-R BT.601)
        return (
            0.299 * frame[:, :, 0].astype(np.float64)
            + 0.587 * frame[:, :, 1].astype(np.float64)
            + 0.114 * frame[:, :, 2].astype(np.float64)
        )
