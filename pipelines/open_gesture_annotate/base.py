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

    # Optional: a backend MAY implement `set_output_dir(self, out_dir: Path) -> None`.
    # If present, `runner.run_backend` calls it once, BEFORE calling `available()`
    # and before the annotation loop, passing the actual `--out` directory the run
    # is using. This lets a backend that reads another backend's sidecar (e.g. `va`
    # reading `face`'s faces.json) resolve that sidecar beneath the same output
    # directory instead of hard-coding `repo_root() / "annotations"` -- which would
    # silently diverge from a non-default `--out`. It must run before `available()`
    # so that an availability check depending on the output directory (e.g. `va`
    # refusing when `faces.json` is wholly missing) inspects the directory this run
    # is actually using, not the default. Backends that do not read another
    # backend's output need not implement this; the runner's `getattr(backend,
    # "set_output_dir", None)` lookup makes it a no-op for them.
