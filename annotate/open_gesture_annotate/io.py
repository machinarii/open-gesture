"""Reading curated data and writing machine-generated sidecars."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from open_gesture_annotate.schema import new_sidecar, validate_sidecar


@dataclass(frozen=True)
class Gesture:
    """One curated gesture record from manifest.json."""

    id: str
    index: int
    name: str
    category: str
    file: str  # repo-relative POSIX path
    raw: dict = field(repr=False)


def repo_root() -> Path:
    """The open-gesture checkout root, found by walking up to manifest.json."""
    for candidate in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (candidate / "manifest.json").is_file():
            return candidate
    raise FileNotFoundError("could not locate manifest.json above this package")


def load_manifest(root: Path) -> list[Gesture]:
    """Load manifest.json into Gesture records, ordered by `index`."""
    data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    gestures = [
        Gesture(
            id=rec["id"],
            index=rec["index"],
            name=rec["name"],
            category=rec["category"],
            file=rec["file"],
            raw=rec,
        )
        for rec in data["gestures"]
    ]
    return sorted(gestures, key=lambda g: g.index)


def sidecar_path(out_dir: Path, backend) -> Path:
    return Path(out_dir) / backend.sidecar


def read_sidecar(path: Path, backend) -> dict:
    """Load an existing sidecar, or a fresh empty one if it does not exist."""
    path = Path(path)
    if not path.is_file():
        return new_sidecar(backend)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_sidecar(data)
    return data


def write_sidecar(path: Path, data: dict) -> None:
    """Validate then atomically write a sidecar, so a crash cannot truncate it."""
    validate_sidecar(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def completed_ids(data: dict) -> set[str]:
    """Gesture ids already annotated successfully — the basis for resume."""
    return {gid for gid, rec in data["records"].items() if rec.get("status") == "ok"}


def load_image(root: Path, gesture: Gesture) -> np.ndarray:
    """Read a gesture image as a BGR uint8 array."""
    path = Path(root) / gesture.file
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {gesture.file}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode image: {gesture.file}")
    return img
