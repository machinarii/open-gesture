#!/usr/bin/env python3
"""Build the MANO hand-preset library + suggested per-gesture hand mapping.

Many of the 99 gestures share a small set of hand shapes (fist, open palm,
index point, the finger counts, pinch, crossed fingers...). Rather than pose
the fingers 99 times, capture each distinct shape ONCE as a named preset
(smplx_capture.py) and snap it onto every gesture that uses it
(apply_hand_presets.py).

This script produces two seed files:

  hand_presets/library.json    The preset vocabulary. Each preset has a
                               description and left/right pose slots that are
                               null until captured from the SMPL-X rig. The
                               numeric MANO values are NOT fabricated here --
                               even "flat" depends on the add-on's mean-pose
                               setting, so all slots start null and are filled
                               by capturing real poses.

  hand_presets/gesture_map.json  A SUGGESTED hand-preset per gesture, derived
                                 heuristically from physical_description. Like
                                 the motion-spec `dynamic` flag, treat it as a
                                 starting point to review, not ground truth.

Usage:
    python3 build_hand_presets.py [--manifest PATH] [--out-dir DIR] [--force]

stdlib only.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Grounded in the manifest's hand-shape language (see survey in build notes).
PRESETS = {
    "relaxed":         "Neutral resting hand; fingers gently curled. Default when hands aren't the focus.",
    "flat_open":       "Open palm, fingers extended and together. Waves, stop, high-five, push.",
    "five_spread":     "Open hand, all five fingers spread apart. Number five/ten, splayed displays.",
    "fist":            "Fully closed fist. Solidarity, knock, resolve.",
    "thumb_up":        "Closed fist with thumb extended upward. Thumbs up / approval.",
    "ok_ring":         "Thumb and index tips touching in a ring, other three fingers extended. OK sign.",
    "point_index":     "Index extended, remaining fingers curled into the palm. Pointing, number one, beckon.",
    "two_v":           "Index and middle extended in a V, others curled. Number two, victory/peace.",
    "three_fingers":   "Index, middle, ring extended; thumb holds pinky. Number three.",
    "four_fingers":    "All four fingers extended, thumb folded across palm. Number four.",
    "pinch":           "Thumb and index tips nearly touching, small gap. Money rub, small-size, precision.",
    "crossed_fingers": "Index and middle crossed. Luck / promise.",
    "finger_gun":      "Index extended forward, thumb up, others curled. Finger gun / playful point.",
}

# Heuristic id/keyword -> preset rules, checked in order; first match wins.
# Explicit-by-id rules come first where the description alone is ambiguous.
ID_RULES = {
    "number-01": "point_index", "number-02": "two_v", "number-03": "three_fingers",
    "number-04": "four_fingers", "number-05": "five_spread", "number-10": "five_spread",
}
KEYWORD_RULES = [
    (r"\bok\b|thumb and index.*(touch|ring|circle)|index finger to the tip of the thumb", "ok_ring"),
    (r"all five|five fingers spread|open the hand fully|all ten fingers", "five_spread"),
    (r"thumb (straight )?up|thumb.*extend.*up", "thumb_up"),
    (r"cross(ed)? .*finger|finger.*cross", "crossed_fingers"),
    (r"pinch|thumb and index.*small|rub.*thumb", "pinch"),
    (r"index and middle|middle and index|v-shape|v shape", "two_v"),
    (r"index finger straight|extend the index|point|index extended", "point_index"),
    (r"\bfist\b", "fist"),
    (r"flat|palm (facing )?(out|forward)|open (palm|hand)|fingers (together|extended)", "flat_open"),
]


def suggest_preset(g: dict) -> tuple[str, bool]:
    """Return (preset, is_confident). Low-confidence picks fall back to relaxed."""
    if g["id"] in ID_RULES:
        return ID_RULES[g["id"]], True
    desc = g["physical_description"].lower()
    for pattern, preset in KEYWORD_RULES:
        if re.search(pattern, desc):
            return preset, True
    return "relaxed", False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--manifest", type=Path, default=here.parent / "manifest.json")
    parser.add_argument("--out-dir", type=Path, default=here / "hand_presets")
    parser.add_argument("--force", action="store_true", help="Overwrite library.json (keeps captured poses otherwise)")
    args = parser.parse_args()

    gestures = json.loads(args.manifest.read_text())["gestures"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Library: don't clobber captured poses on re-run unless --force.
    lib_path = args.out_dir / "library.json"
    if lib_path.exists() and not args.force:
        library = json.loads(lib_path.read_text())
        print(f"Kept existing {lib_path.name} (has captured poses); --force to rebuild.")
    else:
        library = {
            "mano_layout": {"left_hand_pose": 45, "right_hand_pose": 45, "note": "15 joints × 3 axis-angle, PCA OFF"},
            "presets": {name: {"description": desc, "left": None, "right": None} for name, desc in PRESETS.items()},
        }
        lib_path.write_text(json.dumps(library, indent=2) + "\n")
        print(f"Wrote {lib_path}  ({len(PRESETS)} presets, all uncaptured)")

    # Suggested mapping: dominant hand gets the preset; left stays relaxed unless
    # the description is explicitly two-handed-distinct (the 6-9 counts).
    mapping, low_conf = {}, []
    for g in gestures:
        preset, confident = suggest_preset(g)
        entry = {"right_hand": preset, "left_hand": "relaxed", "review": not confident}
        # Two-handed asymmetric counts: one hand shows five, other shows the count.
        if g["id"] in {"number-06", "number-07", "number-08", "number-09"}:
            other = {"number-06": "point_index", "number-07": "two_v",
                     "number-08": "three_fingers", "number-09": "four_fingers"}[g["id"]]
            entry = {"right_hand": "five_spread", "left_hand": other, "review": True}
        mapping[g["id"]] = entry
        if not confident:
            low_conf.append(g["id"])

    map_path = args.out_dir / "gesture_map.json"
    map_path.write_text(json.dumps(mapping, indent=2) + "\n")
    print(f"Wrote {map_path}  ({len(mapping)} gestures)")
    print(f"  low-confidence (defaulted to relaxed, review these): {len(low_conf)}")
    if low_conf:
        print("   ", ", ".join(low_conf[:10]) + (" ..." if len(low_conf) > 10 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
