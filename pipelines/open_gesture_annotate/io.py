"""Reading curated data and writing machine-generated sidecars."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


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
