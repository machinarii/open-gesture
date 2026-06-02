#!/usr/bin/env python3
"""Snap captured hand presets onto motion-spec keyframes.

Composes the two halves of a gesture pose: this fills the FINGER fields
(left_hand_pose / right_hand_pose) of every keyframe from the assigned preset,
leaving the ARM fields (global_orient / body_pose) for the IK-posed body capture
to fill. That decoupling is the whole point of presets -- author a hand shape
once, reuse it across every gesture that needs it, and pose only the arm per
gesture.

Reads:
  hand_presets/library.json      captured presets (smplx_capture.py)
  hand_presets/gesture_map.json  gesture -> {left_hand, right_hand} preset names
  motion_specs/<id>.json         keyframes to fill

For each keyframe it ensures an smplx_pose dict exists (creating one with null
arm fields if absent) and writes the hand fields. Presets still uncaptured
(null) are reported and skipped -- never zero-filled.

Usage:
    python3 apply_hand_presets.py [--presets DIR] [--motion-specs DIR] [--only ID...]

stdlib only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EMPTY_POSE = {"global_orient": None, "body_pose": None, "left_hand_pose": None, "right_hand_pose": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--presets", type=Path, default=here / "hand_presets")
    parser.add_argument("--motion-specs", type=Path, default=here / "motion_specs")
    parser.add_argument("--only", nargs="*", help="Limit to these gesture ids")
    args = parser.parse_args()

    library = json.loads((args.presets / "library.json").read_text())["presets"]
    gesture_map = json.loads((args.presets / "gesture_map.json").read_text())

    def captured(preset_name: str, hand: str):
        """Return the captured pose for (preset, hand), or None if uncaptured."""
        preset = library.get(preset_name)
        return preset.get(hand) if preset else None

    updated, blocked = 0, {}  # blocked: preset_name -> set(hand) still uncaptured
    targets = args.only or list(gesture_map.keys())

    for gid in targets:
        spec_path = args.motion_specs / f"{gid}.json"
        if not spec_path.exists():
            continue
        assignment = gesture_map[gid]
        left = captured(assignment["left_hand"], "left")
        right = captured(assignment["right_hand"], "right")

        # Track which assigned presets can't be applied yet.
        if left is None:
            blocked.setdefault(assignment["left_hand"], set()).add("left")
        if right is None:
            blocked.setdefault(assignment["right_hand"], set()).add("right")
        if left is None and right is None:
            continue

        spec = json.loads(spec_path.read_text())
        for kf in spec["keyframes"]:
            pose = kf.get("smplx_pose") or dict(EMPTY_POSE)
            if left is not None:
                pose["left_hand_pose"] = left
            if right is not None:
                pose["right_hand_pose"] = right
            kf["smplx_pose"] = pose
        spec_path.write_text(json.dumps(spec, indent=2) + "\n")
        updated += 1

    print(f"Updated hand fields in {updated} motion specs.")
    if blocked:
        print("Uncaptured presets (capture with smplx_capture.py before they apply):")
        for name in sorted(blocked):
            print(f"  {name:16s} hands: {', '.join(sorted(blocked[name]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
