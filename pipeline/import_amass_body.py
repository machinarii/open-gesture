#!/usr/bin/env python3
"""Import body/arm motion from an AMASS sequence into a motion spec's keyframes.

AMASS stores mocap as SMPL-H/SMPL-X parameters per frame. This fills the ARM
half of a gesture pose -- global_orient + body_pose (the 21 body joints,
including shoulders/elbows/wrists) -- by sampling an AMASS sequence at each
keyframe's normalized time. It deliberately does NOT touch the hand fields:
those come from the InterHand import / hand presets, so the two motion sources
compose cleanly (arms from AMASS, fingers from InterHand or captured presets).

Pose layout (both SMPL-H 156 and SMPL-X 165 share this prefix):
    poses[:, 0:3]   global_orient
    poses[:, 3:66]  body_pose   (21 joints × 3)

═══════════════════════════════════════════════════════════════════════════
LICENSE: AMASS is research-only (non-commercial, per-subdataset terms). Specs
filled here are stamped license_tier="non-commercial". Keep out of a commercial
release.

VERIFY-ON-REAL-DATA: AMASS .npz keys assumed: 'poses' [N, >=66], 'trans',
'mocap_framerate' (some releases: 'mocap_frame_rate'). Confirm the body-pose
slice matches your bodies; check that the chosen frame range covers the gesture
stroke (use --info to inspect, then --start-frame/--end-frame to trim).
═══════════════════════════════════════════════════════════════════════════

Usage:
    python3 import_amass_body.py --npz SEQ.npz --info
    python3 import_amass_body.py --npz SEQ.npz --spec motion_specs/greet-01.json \
        [--start-frame N] [--end-frame N]

Requires: numpy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EMPTY_POSE = {"global_orient": None, "body_pose": None, "left_hand_pose": None, "right_hand_pose": None}
LICENSE = {"dataset": "AMASS", "license": "research-only", "tier": "non-commercial"}
GLOBAL_LEN, BODY_LEN = 3, 63


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True, help="AMASS sequence .npz")
    parser.add_argument("--spec", type=Path, help="motion_specs/<id>.json to fill")
    parser.add_argument("--info", action="store_true", help="Print sequence info and exit")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None, help="Default: last frame")
    args = parser.parse_args()

    try:
        import numpy as np
    except ImportError:
        sys.exit("error: numpy required. pip install -r requirements.txt")

    data = np.load(args.npz, allow_pickle=True)
    if "poses" not in data:
        sys.exit(f"error: {args.npz.name} has no 'poses' key (keys: {list(data.keys())})")
    poses = data["poses"]
    n_frames, p = poses.shape
    if p < GLOBAL_LEN + BODY_LEN:
        sys.exit(f"error: poses width {p} < {GLOBAL_LEN + BODY_LEN}; not a body pose array")
    fps = float(data["mocap_framerate"]) if "mocap_framerate" in data else (
        float(data["mocap_frame_rate"]) if "mocap_frame_rate" in data else None)

    if args.info:
        print(f"{args.npz.name}: {n_frames} frames, pose-dim {p}, fps {fps}")
        print(f"  duration ~{n_frames / fps:.2f}s" if fps else "  (no framerate key)")
        print(f"  body model: {'SMPL-X (165)' if p >= 165 else 'SMPL-H (156)' if p >= 156 else f'width {p}'}")
        return 0

    if not args.spec:
        sys.exit("error: --spec required (or use --info)")

    end = args.end_frame if args.end_frame is not None else n_frames - 1
    if not (0 <= args.start_frame <= end < n_frames):
        sys.exit(f"error: frame range [{args.start_frame},{end}] out of bounds (0..{n_frames - 1})")

    spec = json.loads(args.spec.read_text())
    for kf in spec["keyframes"]:
        t = float(kf["t"])
        frame = args.start_frame + round(t * (end - args.start_frame))
        row = poses[frame]
        pose = kf.get("smplx_pose") or dict(EMPTY_POSE)
        pose["global_orient"] = [float(x) for x in row[:GLOBAL_LEN]]
        pose["body_pose"] = [float(x) for x in row[GLOBAL_LEN:GLOBAL_LEN + BODY_LEN]]
        kf["smplx_pose"] = pose  # hands untouched

    spec["dynamic_source"] = "amass"
    spec.setdefault("source", {})["body"] = {**LICENSE, "ref": f"{args.npz.name}[{args.start_frame}:{end}]"}
    spec["license_tier"] = "non-commercial"  # AMASS is research-only; never downgrade
    args.spec.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Filled {len(spec['keyframes'])} keyframes' arm pose from {args.npz.name}"
          f"[{args.start_frame}:{end}] -> {args.spec.name}")
    print("  tagged license_tier=non-commercial (AMASS research-only); hands untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
