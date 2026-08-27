"""Record what produced the annotations, and under which licence.

The repo promises MIT code and CC-BY-4.0 images with commercial use permitted.
Pretrained weights are licensed separately from the libraries that load them, so
a run that silently pulls a non-commercial weight would quietly falsify that
promise. This module makes every weight's licence explicit and warns on any that
is not permissive.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from open_gesture_annotate import registry

PERMISSIVE = {"MIT", "Apache-2.0", "Apache 2.0", "BSD-3-Clause", "BSD-2-Clause",
              "Apache-2.0 / MIT"}


def is_permissive(license_str: str | None) -> bool:
    return bool(license_str) and license_str.strip() in PERMISSIVE


def collect(keys: list[str]) -> dict:
    backends = {}
    for key in keys:
        entry: dict = {}
        try:
            backend = registry.get(key)
            available, reason = backend.available()
            entry = {
                "name": backend.name,
                "version": backend.version,
                "sidecar": backend.sidecar,
                "available": bool(available),
                "reason": reason,
                "provenance": backend.provenance(),
            }
        except Exception as exc:
            entry = {"available": False, "reason": f"{type(exc).__name__}: {exc}",
                     "provenance": {}}
        backends[key] = entry
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backends": backends,
    }


def licence_warnings(meta: dict) -> list[str]:
    """One warning per model whose licence is not known-permissive."""
    warnings = []
    for key, entry in meta["backends"].items():
        for model in entry.get("provenance", {}).get("models", []):
            lic = model.get("license")
            if not is_permissive(lic):
                warnings.append(
                    f"{key}: model {model.get('name', '?')!r} has licence {lic!r} — "
                    "verify before commercial redistribution"
                )
    return warnings


def write_meta(out_dir: Path, keys: list[str]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = collect(keys)
    meta["licence_warnings"] = licence_warnings(meta)
    path = out_dir / "_meta.json"
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
