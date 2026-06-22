"""
Mathematical verification metrics for the EdgeOps AI Agent pipeline.

Implements the three key performance indicators defined in the system
specifications:

  - E_temp  : Temporal Detection Error
  - R_parsing : Schema Parsing Rate
  - S_rec   : Recommendation Alignment Score (dummy cosine similarity)
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from src.core_agent.structured_parser import (
    EylemFonksiyonu,
    KararDestekRaporu,
    parse_rapor,
)


# ──────────────────────────────────────────────────────────────────────
# E_temp — Temporal Detection Error
# ──────────────────────────────────────────────────────────────────────


def calculate_temporal_error(
    detected_events: list[dict[str, Any]],
    ground_truth_events: list[dict[str, Any]],
    time_field: str = "baslangic_zamani",
) -> float:
    """Compute the mean absolute temporal detection error.

    For every ground-truth event, the closest detected event (by timestamp)
    is found and the absolute time difference is averaged across all events.

    Timestamps can be in ``ss:dd:dd.xxx`` or ``frame_index:N`` format.

    Args:
        detected_events:    Events produced by the pipeline.
        ground_truth_events:  Reference (ground truth) events.
        time_field:           Key used to extract the timestamp.

    Returns:
        Mean temporal error in **seconds** (``float``).
        Returns ``0.0`` if no ground-truth events exist.
    """
    if not ground_truth_events:
        return 0.0

    gt_times = [_parse_timestamp(e[time_field]) for e in ground_truth_events]
    det_times = [_parse_timestamp(e[time_field]) for e in detected_events]

    if not det_times:
        return float("inf")

    gt_arr = np.array(gt_times, dtype=np.float64)      # (N,)
    det_arr = np.array(det_times, dtype=np.float64)     # (M,)

    # For each GT time, find the closest detection time
    # |gt - det|  ->  outer diff  ->  row-wise min
    diffs = np.abs(gt_arr[:, None] - det_arr[None, :])  # (N, M)
    best_diffs = diffs.min(axis=1)                       # (N,)

    return float(best_diffs.mean())


def _parse_timestamp(value: str) -> float:
    """Convert a timestamp string to seconds.

    Accepted formats:
      - ``ss:dd:dd.xxx``        (hours:minutes:seconds.fraction)
      - ``frame_index:<N>``     (N / 30.0  — assuming 30 FPS)
    """
    if value.startswith("frame_index:"):
        idx = int(value.split(":")[1])
        return idx / 30.0

    parts = value.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)

    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)

    try:
        return float(value)
    except ValueError:
        return 0.0


# ──────────────────────────────────────────────────────────────────────
# R_parsing — Schema Parsing Rate
# ──────────────────────────────────────────────────────────────────────


def calculate_parsing_rate(raw_outputs: list[str]) -> float:
    """Compute the fraction of model outputs that parse successfully.

    Each raw string is attempted against ``parse_rapor()``.  Outputs that
    are valid JSON **and** pass Pydantic validation count as successes.

    Args:
        raw_outputs:  List of raw strings produced by the VLM agent.

    Returns:
        Rate in **[0.0, 1.0]** (1.0 = 100% compliance).
    """
    if not raw_outputs:
        return 0.0

    successes = 0
    for raw in raw_outputs:
        cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            parse_rapor(cleaned)
            successes += 1
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

    return successes / len(raw_outputs)


# ──────────────────────────────────────────────────────────────────────
# S_rec — Recommendation Alignment Score
# ──────────────────────────────────────────────────────────────────────


# Action-type → one-hot style vector index (dummy encoding)
_ACTION_INDEX: dict[str, int] = {
    "lock_down_zone": 0,
    "call_emergency_services": 1,
    "guvenli_bolge_uyarisi": 2,
    "bilgi_amacli": 3,
}
_ACTION_DIM: int = len(_ACTION_INDEX)


def _actions_to_matrix(
    reports: list[KararDestekRaporu],
) -> np.ndarray:
    """Convert a list of reports to an ``(N, D)`` binary action matrix.

    Each row corresponds to one report; each column to one action type.
    A cell is 1 if that action appears at least once in the report's
    ``eylem_onerileri`` list, 0 otherwise.
    """
    mat = np.zeros((len(reports), _ACTION_DIM), dtype=np.float64)
    for i, rapor in enumerate(reports):
        for oneri in rapor.eylem_onerileri:
            col = _ACTION_INDEX.get(oneri.fonksiyon.value)
            if col is not None:
                mat[i, col] = 1.0
    return mat


def calculate_recommendation_alignment(
    predicted: list[KararDestekRaporu],
    ground_truth: list[KararDestekRaporu],
) -> float:
    """Compute cosine-similarity alignment between predicted and ground-truth
    recommendation matrices.

    Both lists must have the same length (frame-by-frame correspondence).

    Args:
        predicted:    Reports produced by the agent pipeline.
        ground_truth: Reference reports from the ground-truth dataset.

    Returns:
        Mean cosine similarity in **[0.0, 1.0]** across all frames.
    """
    if not predicted or not ground_truth:
        return 0.0

    n = min(len(predicted), len(ground_truth))
    P = _actions_to_matrix(predicted[:n])
    G = _actions_to_matrix(ground_truth[:n])

    # Row-wise cosine similarity
    norms_p = np.linalg.norm(P, axis=1, keepdims=True)
    norms_g = np.linalg.norm(G, axis=1, keepdims=True)

    both_zero = (norms_p.squeeze() == 0.0) & (norms_g.squeeze() == 0.0)
    norms_p_safe = np.where(norms_p == 0, 1.0, norms_p)
    norms_g_safe = np.where(norms_g == 0, 1.0, norms_g)

    P_norm = P / norms_p_safe
    G_norm = G / norms_g_safe
    cos_sims = (P_norm * G_norm).sum(axis=1)
    cos_sims[both_zero] = 1.0  # identical all-zero action profiles

    return float(cos_sims.mean())


# ──────────────────────────────────────────────────────────────────────
# Convenience runner
# ──────────────────────────────────────────────────────────────────────


def evaluate_all(
    detected_events: list[dict[str, Any]],
    ground_truth_events: list[dict[str, Any]],
    raw_agent_outputs: list[str],
    predicted_reports: list[KararDestekRaporu],
    ground_truth_reports: list[KararDestekRaporu],
) -> dict[str, float]:
    """Run all three metrics and return a results dictionary.

    Args:
        detected_events:       Pipeline-detected events.
        ground_truth_events:   Ground-truth events.
        raw_agent_outputs:     Raw VLM output strings.
        predicted_reports:     Parsed agent reports.
        ground_truth_reports:  Ground-truth reports.

    Returns:
        ``{"E_temp": ..., "R_parsing": ..., "S_rec": ...}``
    """
    return {
        "E_temp": calculate_temporal_error(detected_events, ground_truth_events),
        "R_parsing": calculate_parsing_rate(raw_agent_outputs),
        "S_rec": calculate_recommendation_alignment(
            predicted_reports, ground_truth_reports
        ),
    }
