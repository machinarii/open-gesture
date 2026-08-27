#!/usr/bin/env python3
"""Generate per-gesture motion-spec stubs from manifest.json.

The manifest describes *end poses*, not motion. BlenderProc/SMPL-X needs a
temporal spec -- how the body moves over time -- which doesn't exist yet and
is the real gating task before any rendering. This script bootstraps that:
for each gesture it emits a motion-spec stub that a human (or a downstream
motion-synthesis step) fills in.

Each stub is seeded from the manifest:
  - `dynamic`: heuristic guess from movement language in physical_description.
    Static gestures (thumbs up, OK sign) need a single hold pose; dynamic ones
    (wave, beckon, hand chop) need keyframed motion. The heuristic only flags
    candidates -- always eyeball the `dynamic` field before authoring.
  - `suggested_duration_s`: short hold for static, longer window for dynamic.
  - `keyframes`: placeholder list. Static -> one pose; dynamic -> start/peak/end
    scaffold to be filled with SMPL-X pose params.
  - `body_parts` / `participants`: carried from the manifest so the rig setup
    and multi-person staging are pre-populated.

Output: pipeline/motion_specs/<id>.json, one per gesture.

Usage:
    python3 generate_motion_specs.py [--manifest PATH] [--out-dir DIR] [--force]

stdlib only.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Movement language that implies the gesture is defined by motion, not a pose.
# Tuned against the 99-gesture manifest -- err toward flagging dynamic, since a
# false "static" silently drops the motion that makes the gesture legible.
MOTION_WORDS = [
    "wave", "waving", "beckon", "swing", "swinging", "shake", "shaking",
    "move", "moving", "circular motion", "rotate", "rotating", "twist",
    "flick", "flicking", "tap", "tapping", "chop", "chopping", "pump",
    "pumping", "back and forth", "up and down", "side to side", "repeated",
    "repeatedly", "rhythm", "rhythmic", "draw", "drawing", "trace", "tracing",
    "raise then", "raise and lower", "sweep", "sweeping", "snap", "snapping",
    "clap", "clapping", "point and", "nod", "nodding", "bounce", "bouncing",
]

_MOTION_RE = re.compile(r"|".join(re.escape(w) for w in MOTION_WORDS), re.IGNORECASE)


def is_dynamic(physical_description: str) -> bool:
    return bool(_MOTION_RE.search(physical_description))


def keyframe_scaffold(dynamic: bool) -> list[dict]:
    """Placeholder keyframes. SMPL-X pose params get filled in by the author."""
    if not dynamic:
        return [
            {"t": 0.0, "phase": "hold", "smplx_pose": None, "note": "single end pose"},
        ]
    return [
        {"t": 0.0, "phase": "start", "smplx_pose": None, "note": "neutral / pre-stroke"},
        {"t": 0.5, "phase": "peak", "smplx_pose": None, "note": "stroke apex (most legible frame)"},
        {"t": 1.0, "phase": "end", "smplx_pose": None, "note": "retraction / rest"},
    ]


def build_spec(g: dict) -> dict:
    dynamic = is_dynamic(g["physical_description"])
    return {
        "id": g["id"],
        "name": g["name"],
        "category": g["category"],
        "source_physical_description": g["physical_description"],
        "dynamic": dynamic,
        "dynamic_source": "heuristic",  # set to "manual" once a human confirms
        "suggested_duration_s": 2.0 if dynamic else 1.0,
        "loop": dynamic,  # cyclic gestures (wave, pump) can loop for more frames
        "participants": g["number_of_people"],
        "body_parts": g["body_parts"],
        "keyframes": keyframe_scaffold(dynamic),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "manifest.json",
        help="Path to manifest.json (default: repo root)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "motion_specs",
        help="Output directory (default: pipeline/motion_specs/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing spec files (default: skip files that exist, "
        "so hand-authored specs are never clobbered on re-run)",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    gestures = manifest["gestures"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = skipped = dynamic_count = 0
    for g in gestures:
        path = args.out_dir / f"{g['id']}.json"
        if path.exists() and not args.force:
            skipped += 1
            continue
        spec = build_spec(g)
        dynamic_count += spec["dynamic"]
        path.write_text(json.dumps(spec, indent=2) + "\n")
        written += 1

    print(f"Specs written: {written}, skipped (exist): {skipped}")
    if written:
        print(f"Flagged dynamic (heuristic): {dynamic_count}/{written}")
    print(f"Output: {args.out_dir}")
    if skipped:
        print("Re-run with --force to regenerate skipped specs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
