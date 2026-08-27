#!/usr/bin/env python3
"""Audit the commercial / research split across motion specs before rendering.

Each clip inherits its motion spec's license_tier (set when InterHand- or
AMASS-derived motion is applied). This scans the specs, cross-references the
render configs to project the split at the CLIP level (specs × variants), and
reports pose-authoring completeness so you know exactly what will land in the
commercial vs. research dataset before burning render time.

Pose completeness: a spec is "renderable" only when every keyframe has all four
smplx_pose fields filled (the same gate render_clip.py enforces).

Usage:
    python3 report_license_tiers.py [--motion-specs DIR] [--render-configs DIR]
                                    [--json]

stdlib only.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

POSE_FIELDS = ("global_orient", "body_pose", "left_hand_pose", "right_hand_pose")


def is_renderable(spec: dict) -> bool:
    for kf in spec["keyframes"]:
        pose = kf.get("smplx_pose")
        if pose is None or any(pose.get(f) is None for f in POSE_FIELDS):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--motion-specs", type=Path, default=here / "motion_specs")
    parser.add_argument("--render-configs", type=Path, default=here / "render_configs")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    # Clips-per-gesture from the render index (default 0 if not generated yet).
    variants = Counter()
    index_path = args.render_configs / "index.json"
    if index_path.exists():
        for job in json.loads(index_path.read_text())["jobs"]:
            variants[job["output_clip"].split("__", 1)[0]] += 1

    specs_by_tier = Counter()
    clips_by_tier = Counter()
    renderable_specs = Counter()   # keyed by tier, only counts fully-authored
    renderable_clips = Counter()
    by_gesture = {}

    for spec_path in sorted(args.motion_specs.glob("*.json")):
        spec = json.loads(spec_path.read_text())
        gid = spec["id"]
        tier = spec.get("license_tier", "untagged")
        n_clips = variants.get(gid, 0)
        renderable = is_renderable(spec)

        specs_by_tier[tier] += 1
        clips_by_tier[tier] += n_clips
        if renderable:
            renderable_specs[tier] += 1
            renderable_clips[tier] += n_clips
        by_gesture[gid] = {"tier": tier, "renderable": renderable, "clips": n_clips}

    report = {
        "tiers": sorted(set(specs_by_tier) | {"commercial", "non-commercial"}),
        "specs_by_tier": dict(specs_by_tier),
        "clips_by_tier": dict(clips_by_tier),
        "renderable_specs_by_tier": dict(renderable_specs),
        "renderable_clips_by_tier": dict(renderable_clips),
    }

    if args.json:
        print(json.dumps({"summary": report, "by_gesture": by_gesture}, indent=2))
        return 0

    total_specs = sum(specs_by_tier.values())
    print(f"Motion specs: {total_specs}   Render variants indexed: {sum(variants.values())}\n")
    header = f"{'tier':16s}{'specs':>8s}{'renderable':>12s}{'clips':>8s}{'renderable clips':>18s}"
    print(header)
    print("-" * len(header))
    for tier in sorted(specs_by_tier, key=lambda t: (-specs_by_tier[t], t)):
        print(f"{tier:16s}{specs_by_tier[tier]:>8d}{renderable_specs[tier]:>12d}"
              f"{clips_by_tier[tier]:>8d}{renderable_clips[tier]:>18d}")
    print("-" * len(header))
    nc = renderable_clips.get("non-commercial", 0)
    com = renderable_clips.get("commercial", 0)
    print(f"\nRenderable now: {com} commercial clips, {nc} research-only clips.")
    if total_specs and sum(renderable_specs.values()) == 0:
        print("(Nothing renderable yet — author/import poses first.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
