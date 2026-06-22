"""
Mock Re-ID embedding extractor for cross-view person matching.

Simulates a trained appearance-based ReID network (e.g. OSNet / ResNet-50)
that maps a person crop to a 512-dim descriptor.  The mock is
deterministic — the same pixel crop always produces the same embedding,
enabling consistent identity matching across frames and grid views.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np

logger = logging.getLogger("edgeops.tracking_reid.extractor")

EMBEDDING_DIM: int = 512
"""Dimensionality of the output embedding vector."""


class ReIDExtractor:
    """Appearance-based feature extractor for cross-view person Re-ID.

    Usage:
        extractor = ReIDExtractor()
        crop = frame[y1:y2, x1:x2]
        emb = extractor.extract(crop)
        # emb.shape -> (512,)  L2-normalized unit vector
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def extract(self, person_crop: np.ndarray) -> np.ndarray:
        """Extract a fixed-size appearance embedding from a person crop.

        The mock uses a pixel-content hash to seed a deterministic PRNG,
        then generates a 512-dim vector and L2-normalises it.

        Args:
            person_crop:  RGB or grayscale image crop (H, W [, C]) uint8.

        Returns:
            L2-normalised ``(dim,)`` float64 embedding.

        Raises:
            ValueError: If ``person_crop`` is empty or has unknown dtype.
        """
        if person_crop.size == 0:
            raise ValueError("Cannot extract embedding from empty crop")

        seed = self._content_hash(person_crop)
        rng = np.random.default_rng(seed)
        vec = rng.normal(0.0, 1.0 / self.dim, size=self.dim)
        vec /= np.linalg.norm(vec) + 1e-12
        return vec

    def extract_batch(
        self, crops: list[np.ndarray]
    ) -> np.ndarray:
        """Extract embeddings for a batch of person crops.

        Args:
            crops:  List of image crops.

        Returns:
            ``(N, dim)`` matrix where each row is an L2-normalised embedding.
        """
        return np.stack([self.extract(c) for c in crops], axis=0)

    @staticmethod
    def _content_hash(crop: np.ndarray) -> int:
        """Produce a deterministic integer seed from the pixel content.

        Uses SHA-256 on the raw bytes and truncates to 64 bits so the
        same visual input always yields the same embedding.
        """
        raw = np.ascontiguousarray(crop).tobytes()
        digest = hashlib.sha256(raw).digest()
        return int.from_bytes(digest[:8], "little")
