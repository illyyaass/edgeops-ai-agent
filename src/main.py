#!/usr/bin/env python3
"""
EdgeOps AI Agent — Hybrid Edge-Server Cascade Pipeline Orchestrator.

End-to-end loop:
  1. Synthetic or real video frame -> 5-grid ROI crops
  2. SSIMMotionFilter  ->  skip static grids
  3. GridVideoProcessor ->  anomaly detections
  4. ReIDExtractor     ->  appearance embeddings
  5. VectorRegistry    ->  cross-grid identity matching
  6. YerelVideoAjanYonetici ->  VLM reasoning (mock or live vLLM)
  7. KararDestekRaporu ->  execute recommended actions
"""

from __future__ import annotations

# ── Path auto-discovery (runs before ANY local import) ────────────────
import os
import sys

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

# ── Standard library ──────────────────────────────────────────────────
import base64
import io
import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ── Project imports ───────────────────────────────────────────────────
from src.core_agent.prompt_templates import SYSTEM_PROMPT_TR, build_agent_prompt
from src.core_agent.structured_parser import (
    GridPosition,
    KararDestekRaporu,
    OlayDetayi,
    OlayTuru,
    parse_rapor,
)
from src.edge_triage.detector import (
    FACILITY_ZONES,
    GridVideoProcessor,
    ROI_LAYOUT,
)
from src.edge_triage.ssim_filter import SSIMMotionFilter
from src.tools.security_actions import lock_down_zone, guvenli_bolge_uyarisi
from src.tools.emergency_actions import (
    bilgi_amacli_bildirim,
    call_emergency_services,
)
from src.tracking_reid.reid_extractor import ReIDExtractor
from src.tracking_reid.vector_storage import VectorRegistry

logger = logging.getLogger("edgeops.pipeline")

# ──────────────────────────────────────────────────────────────────────
# Action dispatch map
# ──────────────────────────────────────────────────────────────────────

ACTION_DISPATCH: dict[str, Any] = {
    "lock_down_zone": lock_down_zone,
    "call_emergency_services": call_emergency_services,
    "guvenli_bolge_uyarisi": guvenli_bolge_uyarisi,
    "bilgi_amacli": bilgi_amacli_bildirim,
}


# ──────────────────────────────────────────────────────────────────────
# GridPosition -> ZoneName helper
# ──────────────────────────────────────────────────────────────────────

_GRID_TO_ZONE: dict[GridPosition, str] = {
    GridPosition.grid_1: "kuzey_kapi",
    GridPosition.grid_2: "dogu_cephe",
    GridPosition.grid_3: "guney_mutfak",
    GridPosition.grid_4: "bati_merdiven",
    GridPosition.grid_5: "merkez_hol",
}


def grid_to_zone(gp: GridPosition) -> str:
    return _GRID_TO_ZONE.get(gp, "belirtilmemis")


# ──────────────────────────────────────────────────────────────────────
# YerelVideoAjanYonetici — mock / live vLLM agent client
# ──────────────────────────────────────────────────────────────────────


class YerelVideoAjanYonetici:
    """Local Video Agent Manager.

    In **mock mode** the agent generates a synthetic but structurally valid
    ``KararDestekRaporu`` so the pipeline can be tested without a server.

    In **live mode** it POSTs the cropped frame + text context to a local
    vLLM endpoint running Qwen2.5-VL-7B-Instruct with guided decoding.
    """

    def __init__(
        self,
        mock: bool = False,
        vllm_endpoint: str = "http://127.0.0.1:8000/v1",
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    ) -> None:
        self.mock = mock
        self.vllm_endpoint = vllm_endpoint
        self.model_name = model_name
        self.conversation_history: list[dict[str, Any]] = []

        if not self.mock:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(
                    base_url=self.vllm_endpoint,
                    api_key="EMPTY",
                )
                logger.info("vLLM client initialised at %s", vllm_endpoint)
            except ImportError:
                logger.warning(
                    "openai package not found — falling back to mock mode. "
                    "Run: pip install openai"
                )
                self.mock = True

    def reason(
        self,
        detection_context: str,
        timestamp: str,
        frame_crop: np.ndarray | None = None,
    ) -> KararDestekRaporu:
        """Feed edge-layer findings to the VLM and return a validated report.

        Args:
            detection_context:  Turkish text (grid, anomaly, bbox, track id).
            timestamp:          Current video timestamp.
            frame_crop:         Optional image crop to send to the VLM.

        Returns:
            ``KararDestekRaporu`` instance.
        """
        if self.mock:
            raw = self._mock_vlm_response(detection_context, timestamp)
        else:
            raw = self._call_vllm(detection_context, timestamp, frame_crop)

        rapor = parse_rapor(raw)
        self.conversation_history.append({"role": "assistant", "content": raw})
        return rapor

    # ── Mock fallback ─────────────────────────────────────────────

    def _mock_vlm_response(self, ctx: str, timestamp: str) -> str:
        """Generate a plausible ``KararDestekRaporu`` JSON from context."""
        event_count = min(ctx.count("anomaly_type") + ctx.count("Anomaly"), 5)

        events = []
        for i in range(event_count):
            events.append(
                OlayDetayi(
                    event_id=i + 1,
                    tur=OlayTuru.suspicious_person,
                    aciklama=f"Grid {i+1}'de tespit edilen anomali degerlendiriliyor.",
                    grid_position=GridPosition(f"grid_{i+1}"),
                    zone=grid_to_zone(GridPosition(f"grid_{i+1}")),
                    baslangic_zamani=f"frame_index:{i * 10 + 5}",
                    bitis_zamani="devam_ediyor",
                    seviye=0.85 - i * 0.05,
                ).model_dump()
            )

        risk = random.choice(["dusuk", "orta", "yuksek", "kritik"])

        report = {
            "summary": (
                f"Edge katmanindan {event_count} olay tespit edildi. "
                f"Genel risk seviyesi: {risk}."
            ),
            "events": events,
            "risk_analizi": {
                "genel_risk_seviyesi": risk,
                "en_tehlikeli_grid": f"grid_{event_count or 1}",
                "aciklama": f"{event_count} olay tespit edildi, degerlendirme yapildi.",
            },
            "eylem_onerileri": [
                {
                    "fonksiyon": "lock_down_zone",
                    "parametreler": {"zone_name": "kuzey_kapi"},
                    "gerekce": "Anomali tespiti nedeniyle bolge kapatiliyor.",
                }
            ]
            if risk in ("yuksek", "kritik")
            else [
                {
                    "fonksiyon": "bilgi_amacli",
                    "parametreler": {
                        "baslik": "Durum Raporu",
                        "icerik": f"{event_count} olay gozlemlendi, risk: {risk}",
                    },
                    "gerekce": "Bilgilendirme amacli bildirim.",
                }
            ],
        }
        return json.dumps(report, ensure_ascii=False)

    # ── Live vLLM client ──────────────────────────────────────────

    def _call_vllm(
        self,
        detection_context: str,
        timestamp: str,
        frame_crop: np.ndarray | None = None,
    ) -> str:
        """Call the local vLLM OpenAI-compatible endpoint with guided decoding.

        If the connection fails (server not running, network error, etc.) the
        method falls back gracefully to ``_mock_vlm_response`` so the pipeline
        never halts.

        Args:
            detection_context:  Text describing the detection.
            timestamp:          Current video timestamp (used by fallback).
            frame_crop:         Optional image crop to send as a vision input.

        Returns:
            Raw JSON string from the model (real or synthetic).
        """
        try:
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT_TR},
            ]

            user_content: list[dict[str, Any]] = []
            if frame_crop is not None:
                encoded = self._encode_image(frame_crop)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                })
            user_content.append({"type": "text", "text": detection_context})

            messages.append({"role": "user", "content": user_content})

            self.conversation_history.append(
                {"role": "user", "content": detection_context}
            )

            response = self._openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=2048,
                temperature=0.1,
                extra_body={
                    "guided_json": KararDestekRaporu.model_json_schema(),
                },
            )

            return response.choices[0].message.content

        except Exception as exc:
            logger.warning(
                "vLLM connection failed (%s) — falling back to mock mode",
                exc,
            )
            self.mock = True
            return self._mock_vlm_response(detection_context, timestamp)

    @staticmethod
    def _encode_image(crop: np.ndarray) -> str:
        """Encode a numpy image crop to a base64 JPEG string."""
        try:
            from PIL import Image
            pil_img = Image.fromarray(crop)
        except ImportError:
            import cv2
            _, buf = cv2.imencode(".jpg", crop)
            return base64.b64encode(buf.tobytes()).decode("ascii")
        else:
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("ascii")


# ──────────────────────────────────────────────────────────────────────
# Action executor
# ──────────────────────────────────────────────────────────────────────


def execute_actions(rapor: KararDestekRaporu) -> list[str]:
    """Dispatch every recommended action in the report.

    Args:
        rapor:  Validated decision-support report.

    Returns:
        List of confirmation strings from each executed action.
    """
    results: list[str] = []
    for oneri in rapor.eylem_onerileri:
        func = ACTION_DISPATCH.get(oneri.fonksiyon.value)
        if func is None:
            logger.warning("Unknown action: %s", oneri.fonksiyon.value)
            results.append(f"[UYARI] Bilinmeyen fonksiyon: {oneri.fonksiyon.value}")
            continue
        try:
            result = func(**oneri.parametreler)
            results.append(result)
        except TypeError as exc:
            logger.error("Action %s failed: %s", oneri.fonksiyon.value, exc)
            results.append(f"[HATA] {oneri.fonksiyon.value} basarisiz: {exc}")
    return results


# ──────────────────────────────────────────────────────────────────────
# Pipeline orchestrator
# ──────────────────────────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    """Configuration for a pipeline run."""

    ssim_threshold: float = 0.035
    anomaly_prob: float = 0.02
    mock_agent: bool = True
    show_display: bool = True
    max_frames: int = 300
    video_path: str = "data/mock_videos/test_saha_kaydi.mp4"
    frame_size: tuple[int, int] = (960, 720)


@dataclass
class PipelineStep:
    """Snapshot of one frame for debugging / metrics."""

    frame_index: int
    timestamp: str
    static_grids: set[GridPosition]
    detections: int
    anomalies: int
    track_ids: list[int]
    agent_report: KararDestekRaporu | None
    actions_executed: list[str]


def _make_synthetic_frame(h: int, w: int, frame_index: int) -> np.ndarray:
    """Generate a synthetic frame with per-grid motion-like noise."""
    frame = np.full((h, w, 3), 32, dtype=np.uint8)
    for grid, (rx, ry, rw, rh) in ROI_LAYOUT.items():
        x1, y1 = int(rx * w), int(ry * h)
        x2, y2 = int((rx + rw) * w), int((ry + rh) * h)
        phase = (frame_index + int(grid.value.split("_")[1]) * 20) % 40
        brightness = 32 + phase * 2
        frame[y1:y2, x1:x2] = np.random.randint(
            max(0, brightness - 5),
            min(255, brightness + 5),
            (y2 - y1, x2 - x1, 3),
            dtype=np.uint8,
        )
    return frame


# ──────────────────────────────────────────────────────────────────────
# OpenCV visualisation
# ──────────────────────────────────────────────────────────────────────

WINDOW_NAME = "ANKA Team - EdgeOps AI Agent"

_GRID_COLORS: dict[GridPosition, tuple[int, int, int]] = {
    GridPosition.grid_1: (255, 0, 0),    # blue
    GridPosition.grid_2: (0, 255, 0),    # green
    GridPosition.grid_3: (0, 165, 255),  # orange
    GridPosition.grid_4: (255, 255, 0),  # cyan
    GridPosition.grid_5: (255, 0, 255),  # magenta
}


def _draw_overlays(
    frame: np.ndarray,
    frame_index: int,
    timestamp: str,
    anomaly_detections: list,
    grid_to_track: dict,
) -> np.ndarray:
    """Return a display copy of *frame* with grid lines, anomaly highlights,
    and track-info text overlaid."""
    import cv2

    display = frame.copy()
    h, w = display.shape[:2]

    # 1. Grid boundary lines and ROI labels
    for grid, (rx, ry, rw, rh) in ROI_LAYOUT.items():
        x1 = int(rx * w)
        y1 = int(ry * h)
        x2 = int((rx + rw) * w)
        y2 = int((ry + rh) * h)

        det = next((d for d in anomaly_detections if d.grid == grid), None)
        if det is not None:
            color = (0, 0, 255)  # red BGR
            thickness = 3
            tid = grid_to_track.get(grid)
            label = f"{det.facility_zone} - {det.anomaly_type}"
            if tid is not None:
                label += f" - Track #{tid}"
        else:
            color = _GRID_COLORS.get(grid, (100, 100, 100))
            thickness = 1
            label = FACILITY_ZONES.get(grid, "")

        cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            display, label, (x1 + 6, y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )

    # 2. Top info bar
    info = f"Frame: {frame_index} | {timestamp} | Press 'q' to exit"
    cv2.putText(
        display, info, (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
    )

    return display


def run_pipeline(cfg: PipelineConfig | None = None) -> list[PipelineStep]:
    """Execute one full end-to-end pipeline pass.

    Args:
        cfg:  Pipeline configuration (defaults to production settings).

    Returns:
        List of ``PipelineStep`` records.
    """
    if cfg is None:
        cfg = PipelineConfig()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    logger.info(
        "Pipeline starting | max_frames=%d mock=%s",
        cfg.max_frames, cfg.mock_agent,
    )

    ssim = SSIMMotionFilter(threshold=cfg.ssim_threshold)
    reid = ReIDExtractor()
    registry = VectorRegistry()
    agent = YerelVideoAjanYonetici(mock=cfg.mock_agent)

    steps: list[PipelineStep] = []

    # ── Open video source (fallback to synthetic if unavailable) ───
    import cv2 as _cv2

    cap = _cv2.VideoCapture(cfg.video_path) if os.path.exists(cfg.video_path) else None
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        logger.warning("Video not found at '%s' — falling back to synthetic frames", cfg.video_path)
        cap = None
        frame_limit = cfg.max_frames
        fps = 30.0
    else:
        fps = cap.get(_cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
        frame_limit = min(total_frames, cfg.max_frames) if cfg.max_frames else total_frames
        logger.info("Video opened | file=%s fps=%.1f frames=%d", cfg.video_path, fps, total_frames)

    frame_idx: int = 0
    while frame_idx < frame_limit:
        if cap is not None:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
        else:
            frame = _make_synthetic_frame(cfg.frame_size[1], cfg.frame_size[0], frame_idx)

        ts = GridVideoProcessor._format_timestamp(frame_idx, fps)

        grids: dict[GridPosition, np.ndarray] = {}
        for g in GridPosition:
            grids[g] = GridVideoProcessor._crop_roi(frame, ROI_LAYOUT[g])

        change_scores = ssim.update(grids)
        static = {g for g in GridPosition if change_scores.should_skip(g, cfg.ssim_threshold)}

        report = GridVideoProcessor._scan_frame(frame, frame_idx, ts, cfg.anomaly_prob, static)

        anomaly_detections = [
            d for d in report.detections.values()
            if d.anomaly_type is not None and d.bbox is not None
        ]
        track_ids: list[int] = []
        grid_to_track: dict = {}
        for det in anomaly_detections:
            crop = det.frame[det.bbox[1]:det.bbox[3], det.bbox[0]:det.bbox[2]]
            if crop.size == 0:
                continue
            emb = reid.extract(crop)
            tid = registry.register(
                emb,
                grid=det.grid,
                timestamp=ts,
                frame_index=frame_idx,
                metadata={"anomaly": det.anomaly_type, "confidence": det.confidence},
            )
            track_ids.append(tid)
            grid_to_track[det.grid] = tid

        agent_report: KararDestekRaporu | None = None
        actions: list[str] = []
        if anomaly_detections:
            context_lines = [
                f"Zaman: {ts} | Kare: {frame_idx} | "
                f"Grid: {d.grid.value} | Bolge: {d.facility_zone} | "
                f"Anomali: {d.anomaly_type} | Guven: {d.confidence}"
                for d in anomaly_detections
            ]
            if track_ids:
                context_lines.append(f"Takip Edilen Kisiler: {track_ids}")
            ctx = "\n".join(context_lines)

            # Pass the first anomalous frame crop to the VLM
            first_crop = anomaly_detections[0].frame
            agent_report = agent.reason(ctx, ts, frame_crop=first_crop)
            actions = execute_actions(agent_report)

        # ── Live display ──────────────────────────────────────────
        if cfg.show_display:
            try:
                import cv2
                display = _draw_overlays(frame, frame_idx, ts, anomaly_detections, grid_to_track)
                cv2.imshow(WINDOW_NAME, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Display exit requested at frame %d", frame_idx)
                    break
            except ImportError:
                logger.debug("cv2 not available — display disabled")
                cfg.show_display = False

        steps.append(
            PipelineStep(
                frame_index=frame_idx,
                timestamp=ts,
                static_grids=static,
                detections=len([d for d in report.detections.values() if d.anomaly_type is not None]),
                anomalies=len(anomaly_detections),
                track_ids=track_ids,
                agent_report=agent_report,
                actions_executed=actions,
            )
        )

        frame_idx += 1

    cap.release()
    if cfg.show_display:
        try:
            import cv2
            cv2.destroyWindow(WINDOW_NAME)
        except ImportError:
            pass

    logger.info(
        "Pipeline complete | frames=%d anomalies=%d tracks=%d",
        len(steps), sum(s.anomalies for s in steps), registry.track_count,
    )
    return steps


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Run pipeline and print summary."""
    steps = run_pipeline()
    total_anomalies = sum(s.anomalies for s in steps)
    total_actions = sum(len(s.actions_executed) for s in steps)
    print(
        f"\nPipeline finished: {len(steps)} frames, "
        f"{total_anomalies} anomalies, "
        f"{total_actions} actions dispatched"
    )
    for s in steps:
        if s.anomalies > 0:
            print(
                f"  Frame {s.frame_index:4d} | {s.timestamp} | "
                f"static={len(s.static_grids)} | "
                f"anomalies={s.anomalies} | "
                f"tracks={s.track_ids} | "
                f"actions={len(s.actions_executed)}"
            )


if __name__ == "__main__":
    main()
