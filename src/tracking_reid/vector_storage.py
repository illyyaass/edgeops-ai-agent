"""
In-memory vector registry for cross-grid identity matching.

Stores appearance embeddings (512-dim) alongside metadata (grid, timestamp,
track ID) and matches incoming probes via cosine similarity.  Enables the
system to recognise the same person reappearing in different grid windows
or at different times.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.core_agent.structured_parser import GridPosition

logger = logging.getLogger("edgeops.tracking_reid.vector_storage")

COSINE_THRESHOLD: float = 0.75
"""Minimum cosine similarity to consider two embeddings the same identity."""


@dataclass
class TrackEntry:
    """A single stored observation of a tracked identity."""

    track_id: int
    embedding: np.ndarray
    grid: GridPosition
    timestamp: str
    frame_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchResult:
    """Outcome of a probe against the registry."""

    matched: bool
    track_id: int | None
    similarity: float
    entry: TrackEntry | None


class VectorRegistry:
    """In-memory gallery of appearance embeddings with cosine matching.

    Usage:
        registry = VectorRegistry(threshold=0.75)
        tid = registry.register(embedding, grid_1, "00:01:02.000", 42)
        result = registry.search(probe_embedding)
        if result.matched:
            print(f"Re-identified track {result.track_id}")
    """

    def __init__(self, threshold: float = COSINE_THRESHOLD) -> None:
        self.threshold = threshold
        self._entries: list[TrackEntry] = []
        self._next_track_id: int = 1
        self._track_id_to_entries: dict[int, list[TrackEntry]] = {}

    # ── Public API ────────────────────────────────────────────────

    def register(
        self,
        embedding: np.ndarray,
        grid: GridPosition,
        timestamp: str,
        frame_index: int,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Register a new embedding, optionally associating with a known identity.

        If the embedding matches an existing track (cosine >= threshold),
        the new observation is appended to that track.  Otherwise a new
        track ID is created.

        Args:
            embedding:  L2-normalised probe vector (dim,).
            grid:       Grid where the person was observed.
            timestamp:  Video timestamp string.
            frame_index: Frame index.
            metadata:   Optional extra data (anomaly type, bbox, etc.).

        Returns:
            The ``track_id`` this observation was assigned to.
        """
        result = self.search(embedding)
        track_id = result.track_id if result.matched else self._next_track_id

        entry = TrackEntry(
            track_id=track_id,
            embedding=embedding.copy(),
            grid=grid,
            timestamp=timestamp,
            frame_index=frame_index,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        self._track_id_to_entries.setdefault(track_id, []).append(entry)

        if not result.matched:
            self._next_track_id += 1
            logger.info(
                "New track %d | grid=%s frame=%d",
                track_id, grid.value, frame_index,
            )
        else:
            logger.debug(
                "Track %d updated | grid=%s frame=%d sim=%.4f",
                track_id, grid.value, frame_index, result.similarity,
            )

        return track_id

    def search(
        self, embedding: np.ndarray
    ) -> MatchResult:
        """Find the best match for a probe in the gallery.

        Args:
            embedding:  L2-normalised probe vector (dim,).

        Returns:
            MatchResult with the highest-similarity gallery entry (if any).
        """
        if not self._entries:
            return MatchResult(matched=False, track_id=None, similarity=0.0, entry=None)

        gallery = np.stack([e.embedding for e in self._entries], axis=0)
        sims = gallery @ embedding
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= self.threshold:
            entry = self._entries[best_idx]
            return MatchResult(
                matched=True,
                track_id=entry.track_id,
                similarity=best_sim,
                entry=entry,
            )

        return MatchResult(matched=False, track_id=None, similarity=best_sim, entry=None)

    def get_track_history(self, track_id: int) -> list[TrackEntry]:
        """Retrieve all observations for a given track ID."""
        return self._track_id_to_entries.get(track_id, [])

    @property
    def track_count(self) -> int:
        """Number of distinct identities tracked."""
        return len(self._track_id_to_entries)

    @property
    def total_observations(self) -> int:
        """Total number of registered observations."""
        return len(self._entries)

    def reset(self) -> None:
        """Clear all data (e.g. for a new video session)."""
        self._entries.clear()
        self._track_id_to_entries.clear()
        self._next_track_id = 1
        logger.info("VectorRegistry reset")
