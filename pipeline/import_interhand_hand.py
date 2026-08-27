#!/usr/bin/env python3
"""Import a hand pose from InterHand2.6M MANO fits into the preset library.

InterHand2.6M ships MANO parameter fits (the NeuralAnnot release). MANO IS the
SMPL-X hand model, so a MANO hand_pose drops almost directly into a SMPL-X
left/right_hand_pose preset slot -- this is the external source for the finger
shapes (OK sign, the counts, etc.) without posing them by hand.

It is the InterHand counterpart of smplx_capture.py: instead of reading a posed
Blender armature, it reads a MANO fit for a chosen (capture, frame, hand) and
writes the 45-dim hand_pose into the named preset, TAGGED with non-commercial
provenance so downstream tiering can keep it out of the commercial dataset.

═══════════════════════════════════════════════════════════════════════════
LICENSE: InterHand2.6M is CC-BY-NC 4.0 (non-commercial). Presets imported here
are stamped license_tier="non-commercial"; apply_hand_presets.py propagates
that onto any motion spec that uses them. Do NOT mix into a commercial release.

VERIFY-ON-REAL-DATA (no InterHand data in the authoring env):
  * NeuralAnnot JSON shape assumed:
      { capture_id: { frame_idx: { "right": {"pose":[48],...}|null,
                                    "left":  {"pose":[48],...}|null } } }
    MANO `pose` = 48 = global_orient[3] + hand_pose[45]. We drop the first 3
    (the wrist orientation lives in SMPL-X body_pose, not hand_pose) and keep
    the 45 finger params.
  * MANO<->SMPL-X hand axis frames match for the fingers, but confirm the LEFT
    hand: InterHand stores left fits already in the left-hand frame; SMPL-X
    left_hand_pose expects the same, so no extra mirroring SHOULD be needed --
    eyeball one imported left preset on the rig before trusting it.
═══════════════════════════════════════════════════════════════════════════

Usage:
    # list captures / frames available in a MANO annotation file
    python3 import_interhand_hand.py --mano FILE.json --list

    # import one pose into a preset slot
    python3 import_interhand_hand.py --mano FILE.json \
        --capture 0 --frame 12345 --hand right --preset ok_ring

stdlib only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MANO_POSE_LEN = 48          # global_orient(3) + hand_pose(45)
HAND_POSE_LEN = 45
LICENSE = {"dataset": "InterHand2.6M", "license": "CC-BY-NC-4.0", "tier": "non-commercial"}


def load_mano(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit("error: unexpected MANO file structure (expected capture->frame->hand dict)")
    return data


def extract_hand_pose(fit: dict, ref: str) -> list:
    """MANO pose[3:48] -> the 45-dim SMPL-X hand_pose (wrist dropped)."""
    pose = fit.get("pose")
    if pose is None or len(pose) != MANO_POSE_LEN:
        raise SystemExit(f"error: {ref}: MANO 'pose' missing or not length {MANO_POSE_LEN}")
    return [float(x) for x in pose[3:]]  # drop global_orient(3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--mano", type=Path, required=True, help="InterHand2.6M MANO NeuralAnnot JSON")
    parser.add_argument("--library", type=Path, default=here / "hand_presets" / "library.json")
    parser.add_argument("--list", action="store_true", help="List available captures/frames and exit")
    parser.add_argument("--capture", help="capture_id key")
    parser.add_argument("--frame", help="frame_idx key")
    parser.add_argument("--hand", choices=["left", "right"], default="right")
    parser.add_argument("--preset", help="Library preset to write into")
    args = parser.parse_args()

    mano = load_mano(args.mano)

    if args.list:
        print(f"{len(mano)} captures:")
        for cap in list(mano)[:20]:
            frames = mano[cap]
            hands = {h for fr in frames.values() for h, v in fr.items() if v}
            print(f"  capture {cap}: {len(frames)} frames, hands={sorted(hands)}")
        if len(mano) > 20:
            print(f"  ... (+{len(mano) - 20} more captures)")
        return 0

    if not (args.capture and args.frame and args.preset):
        raise SystemExit("error: --capture, --frame and --preset are required (or use --list)")

    try:
        fit = mano[args.capture][args.frame][args.hand]
    except KeyError:
        raise SystemExit(f"error: no {args.hand} fit at capture {args.capture} frame {args.frame}")
    if fit is None:
        raise SystemExit(f"error: {args.hand} fit is null at capture {args.capture} frame {args.frame}")

    ref = f"capture{args.capture}/frame{args.frame}/{args.hand}"
    hand_pose = extract_hand_pose(fit, ref)

    library = json.loads(args.library.read_text())
    if args.preset not in library["presets"]:
        raise SystemExit(f"error: preset '{args.preset}' not in library (add it in build_hand_presets.py)")

    preset = library["presets"][args.preset]
    preset[args.hand] = hand_pose
    # Record provenance + NC tier on the preset so propagation can find it.
    preset.setdefault("source", {})[args.hand] = {**LICENSE, "ref": ref}
    library.setdefault("license_tiers", {})[args.preset] = LICENSE["tier"]

    args.library.write_text(json.dumps(library, indent=2) + "\n")
    print(f"Imported {ref} -> preset '{args.preset}'.{args.hand} ({HAND_POSE_LEN}-dim)")
    print(f"  tagged license_tier=non-commercial (InterHand2.6M CC-BY-NC-4.0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
