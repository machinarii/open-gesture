"""The Backend protocol every model family implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from open_gesture_annotate.io import Gesture


class BackendUnavailable(Exception):
    """Raised when a backend's library or weights cannot be resolved."""


@runtime_checkable
class Backend(Protocol):
    """One model family. Owns exactly one sidecar file and is its only writer."""

    name: str  # stable key, e.g. "uniface-face"
    version: str  # library version, recorded in _meta.json
    sidecar: str  # output filename, e.g. "faces.json"

    def available(self) -> tuple[bool, str]:
        """(importable and weights resolvable?, human-readable reason)."""

    def provenance(self) -> dict:
        """Model names, weight files, SHA256 and license of each weight."""

    def annotate(self, image: np.ndarray, gesture: Gesture) -> dict:
        """Annotate one BGR image. Raise on failure; the runner isolates it."""
