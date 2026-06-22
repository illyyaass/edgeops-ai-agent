"""
GridVideoProcessor — edge-layer video ingestion and mock anomaly detection.

Accepts a single video file representing a 5-grid split-screen view, crops
each grid using hardcoded ROI coordinates, runs a mock high-recall detector
(simulating YOLOv11 / YOLO26), and returns structured anomaly reports
compatible with the core agent's Pydantic schema.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import numpy as np

from src.core_agent.structured_parser import GridPosition, ZoneName

logger = logging.getLogger("edgeops.edge_triage.detector")

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

FACILITY_ZONES: dict[GridPosition, str] = {
    GridPosition.grid_1: "Ambar_Warehouse",
    GridPosition.grid_2: "Ana_Giris_Gate",
    GridPosition.grid_3: "Uretim_Hatti_Line",
    GridPosition.grid_4: "Otopark_Parking",
    GridPosition.grid_5: "Cevre_Perimeter",
}

ZONE_TO_ENUM: dict[str, ZoneName] = {
    "Ambar_Warehouse": ZoneName.kuzey_kapi,
    "Ana_Giris_Gate": ZoneName.dogu_cephe,
    "Uretim_Hatti_Line": ZoneName.guney_mutfak,
    "Otopark_Parking": ZoneName.bati_merdiven,
    "Cevre_Perimeter": ZoneName.merkez_hol,
}

ANOMALY_TYPES: list[str] = [
    "worker_fall",
    "vehicle_tip_over",
    "forklift_tip_over",
    "unauthorized_entry",
    "crowd_formation",
    "equipment_malfunction",
]

# 5-grid split-screen ROI layout (x, y, w, h) as fractions of the frame
# Top row:  3 equal tiles  (grid_1, grid_2, grid_3)
# Bottom row: 2 centred tiles (grid_4, grid_5)
ROI_LAYOUT: dict[GridPosition, tuple[float, float, float, float]] = {
    GridPosition.grid_1: (0.00, 0.00, 0.333, 0.50),
    GridPosition.grid_2: (0.333, 0.00, 0.333, 0.50),
    GridPosition.grid_3: (0.666, 0.00, 0.334, 0.50),
    GridPosition.grid_4: (0.125, 0.50, 0.375, 0.50),
    GridPosition.grid_5: (0.500, 0.50, 0.375, 0.50),
}


# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────


@dataclass
class DetectionResult:
    """Result from scanning a single grid window.

    Fields are designed to map directly into the OlayDetayi Pydantic model
    for downstream processing.
    """

    grid: GridPosition
    zone: ZoneName
    facility_zone: str
    frame: np.ndarray
    timestamp: str
    frame_index: int
    anomaly_type: str | None
    confidence: float
    bbox: tuple[int, int, int, int] | None  # x1, y1, x2, y2


@dataclass
class GridScanReport:
    """Aggregated scan results for all 5 grids at a single frame step."""

    frame_index: int
    timestamp: str
    detections: dict[GridPosition, DetectionResult] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────
# GridVideoProcessor
# ──────────────────────────────────────────────────────────────────────


class GridVideoProcessor:
    """Ingests a 5-grid split-screen video and yields per-frame scan reports.

    Usage:
        processor = GridVideoProcessor("path/to/video.mp4")
        for report in processor.process(anomaly_prob=0.02):
            for det in report.detections.values():
                if det.anomaly_type is not None:
                    print(f"{det.grid.value}: {det.anomaly_type}")
    """

    def __init__(self, video_path: str | Path) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self._validate_roi_layout()

    # ── Public API ────────────────────────────────────────────────

    def process(
        self,
        *,
        anomaly_prob: float = 0.015,
        skip_grids: set[GridPosition] | None = None,
        start_frame: int = 0,
        max_frames: int | None = None,
    ) -> Generator[GridScanReport, None, None]:
        """Yield a scan report for every frame in the video.

        Args:
            anomaly_prob:  Per-grid per-frame probability of triggering a
                           mock anomaly detection (simulated YOLO recall).
            skip_grids:    Set of grids to skip (e.g. from SSIM filter).
            start_frame:   Frame index to begin reading from.
            max_frames:    Maximum number of frames to process (None = all).
        """
        cap = self._open_video()
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            end_frame = min(total_frames, start_frame + max_frames) if max_frames else total_frames

            for idx in range(start_frame, end_frame):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                timestamp = self._format_timestamp(idx, fps)
                report = self._scan_frame(
                    frame=frame,
                    frame_index=idx,
                    timestamp=timestamp,
                    anomaly_prob=anomaly_prob,
                    skip_grids=skip_grids or set(),
                )
                yield report
        finally:
            cap.release()

    # ── Internal helpers ──────────────────────────────────────────

    def _validate_roi_layout(self) -> None:
        """Verify ROI layout covers 0-1 range without overlaps (sanity check)."""
        x_ranges: list[tuple[float, float, GridPosition]] = []
        for grid, (rx, ry, rw, rh) in ROI_LAYOUT.items():
            if abs(rx + rw) > 1.001 or abs(ry + rh) > 1.001:
                logger.warning("ROI %s exceeds frame boundary", grid.value)
            x_ranges.append((rx, rx + rw, grid))
        x_ranges.sort()
        for i in range(len(x_ranges) - 1):
            _, end_a, _ = x_ranges[i]
            start_b, _, _ = x_ranges[i + 1]
            if end_a > start_b + 0.01:
                logger.warning("ROI overlap detected between adjacent grids")

    def _open_video(self) -> cv2.VideoCapture:
        """Open video file with OpenCV."""
        import cv2

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        return cap

    @staticmethod
    def _format_timestamp(frame_index: int, fps: float) -> str:
        """Convert frame index to 'ss:dd:dd.xxx' timestamp string."""
        total_seconds = frame_index / fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        secs = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    @staticmethod
    def _crop_roi(
        frame: np.ndarray,
        roi: tuple[float, float, float, float],
    ) -> np.ndarray:
        """Crop a region of interest from the frame using fractional coords."""
        h, w = frame.shape[:2]
        rx, ry, rw, rh = roi
        x1 = int(rx * w)
        y1 = int(ry * h)
        x2 = int((rx + rw) * w)
        y2 = int((ry + rh) * h)
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _scan_frame(
        frame: np.ndarray,
        frame_index: int,
        timestamp: str,
        anomaly_prob: float,
        skip_grids: set[GridPosition],
    ) -> GridScanReport:
        """Run mock detection on every grid (except skipped ones)."""
        report = GridScanReport(frame_index=frame_index, timestamp=timestamp)

        for grid in GridPosition:
            roi = ROI_LAYOUT[grid]
            grid_frame = GridVideoProcessor._crop_roi(frame, roi)
            facility_name = FACILITY_ZONES[grid]
            zone_enum = ZONE_TO_ENUM[facility_name]

            if grid in skip_grids:
                report.detections[grid] = DetectionResult(
                    grid=grid,
                    zone=zone_enum,
                    facility_zone=facility_name,
                    frame=grid_frame,
                    timestamp=timestamp,
                    frame_index=frame_index,
                    anomaly_type=None,
                    confidence=0.0,
                    bbox=None,
                )
                continue

            anomaly = GridVideoProcessor._mock_detect(grid_frame, anomaly_prob)
            report.detections[grid] = DetectionResult(
                grid=grid,
                zone=zone_enum,
                facility_zone=facility_name,
                frame=grid_frame,
                timestamp=timestamp,
                frame_index=frame_index,
                anomaly_type=anomaly["type"],
                confidence=anomaly["confidence"],
                bbox=anomaly["bbox"],
            )

        return report

    @staticmethod
    def _mock_detect(
        frame: np.ndarray,
        anomaly_prob: float,
    ) -> dict:
        """Simulate a high-recall YOLO detector.

        With probability *anomaly_prob*, fabricates a detection with a
        random anomaly type and a realistic bounding box.  Otherwise
        returns an empty detection (no anomaly).
        """
        h, w = frame.shape[:2]

        if random.random() < anomaly_prob:
            anom = random.choice(ANOMALY_TYPES)
            confidence = round(random.uniform(0.65, 0.98), 3)
            # realistic-ish bbox (not touching edges)
            margin_x, margin_y = int(w * 0.05), int(h * 0.05)
            x1 = random.randint(margin_x, w // 2)
            y1 = random.randint(margin_y, h // 2)
            x2 = x1 + random.randint(w // 8, w // 3)
            y2 = y1 + random.randint(h // 8, h // 3)
            x2 = min(x2, w - margin_x)
            y2 = min(y2, h - margin_y)
            return {"type": anom, "confidence": confidence, "bbox": (x1, y1, x2, y2)}

        return {"type": None, "confidence": 0.0, "bbox": None}
