#!/usr/bin/env python3
"""Generate BlenderProc render configs — the domain-randomization sweep.

Consumes the motion specs and emits one render-job config per synthetic clip,
each a different randomized view of the same gesture: lighting, background,
camera distance/angle, occlusion, and avatar body. BlenderProc reads one config,
drives an SMPL-X avatar through the referenced motion spec, and writes a clip.

The point is variation: the classifier should see each gesture under many
conditions so it generalizes to real outdoor/industrial/AR scenes. This script
owns the *scene* randomization; the *pose* comes from the motion spec (the
`smplx_pose` keyframes a human/motion-synthesis step fills in). The two are
orthogonal — you can generate the full sweep now and the renders become valid
as soon as the poses are authored.

Output naming is the contract with the extraction stage: each clip is named
`<gesture_id>__v<NNN>.mp4`, so extract_landmarks.py recovers the gesture id
(everything before `__`) and joins the clip to its labels. The render config
that produces it is `<gesture_id>__v<NNN>.json`.

Determinism: a seed (default 0) plus the per-job index drives all choices, so
re-running yields byte-identical configs — reproducible datasets, and you can
regenerate a single variant without perturbing the others.

Usage:
    python3 generate_render_configs.py [--motion-specs DIR] [--out DIR]
                                       [--variants N] [--seed N]

stdlib only.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Randomization vocabularies. Picked to span the deployment contexts the README
# cares about (outdoor, industrial, AR) rather than studio-clean renders.
BACKGROUNDS = [
    "street_daytime", "street_dusk", "sidewalk_crowd", "parking_lot",
    "warehouse_interior", "factory_floor", "office_indoor", "park_outdoor",
    "construction_site", "retail_storefront", "ar_living_room", "ar_meeting_room",
]
LIGHTING = ["overcast", "direct_sun", "golden_hour", "fluorescent_indoor", "mixed_indoor", "low_light"]
OCCLUDERS = ["none", "none", "none", "foreground_pole", "passing_person", "partial_frame_edge", "held_object"]


def randomize_job(rng: random.Random, spec: dict, variant: int) -> dict:
    """One render config: scene randomization + a reference to the motion spec."""
    dynamic = spec.get("dynamic", False)
    duration = spec.get("suggested_duration_s", 1.0)
    fps = 30
    return {
        "gesture_id": spec["id"],
        "name": spec["name"],
        "variant": variant,
        "output_clip": f"{spec['id']}__v{variant:03d}.mp4",
        "motion_spec": f"motion_specs/{spec['id']}.json",
        "participants": spec.get("participants", "single"),
        "duration_s": duration,
        "fps": fps,
        "num_frames": max(1, round(duration * fps)),
        "loop_motion": spec.get("loop", False),
        "camera": {
            # Distances span close AR-headset framing to far roadside viewpoints.
            "distance_m": round(rng.uniform(1.5, 8.0), 2),
            "azimuth_deg": round(rng.uniform(-75, 75), 1),
            "elevation_deg": round(rng.uniform(-10, 25), 1),
            "focal_length_mm": rng.choice([24, 35, 50]),
        },
        "lighting": {
            "preset": rng.choice(LIGHTING),
            "intensity": round(rng.uniform(0.4, 1.6), 2),
            "color_temp_k": rng.choice([3200, 4000, 5000, 5600, 6500]),
        },
        "background": rng.choice(BACKGROUNDS),
        "occlusion": rng.choice(OCCLUDERS),
        "avatar": {
            # Vary body so the classifier doesn't latch onto one morphology.
            "gender": rng.choice(["male", "female", "neutral"]),
            "body_shape_seed": rng.randrange(10_000),
            "skin_tone_seed": rng.randrange(10_000),
        },
        "outputs": ["rgb", "depth", "keypoints", "segmentation"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--motion-specs", type=Path, default=here / "motion_specs")
    parser.add_argument("--out", type=Path, default=here / "render_configs")
    parser.add_argument("--variants", type=int, default=8, help="Render variants per gesture (default: 8)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    spec_paths = sorted(args.motion_specs.glob("*.json"))
    if not spec_paths:
        raise SystemExit(f"error: no motion specs in {args.motion_specs} (run generate_motion_specs.py first)")

    args.out.mkdir(parents=True, exist_ok=True)
    index = {"variants_per_gesture": args.variants, "seed": args.seed, "jobs": []}
    dynamic_jobs = 0

    for spec_path in spec_paths:
        spec = json.loads(spec_path.read_text())
        # Per-gesture RNG seeded by (seed, gesture id) so one gesture can be
        # regenerated independently and reproducibly.
        rng = random.Random(f"{args.seed}:{spec['id']}")
        for v in range(args.variants):
            job = randomize_job(rng, spec, v)
            (args.out / f"{spec['id']}__v{v:03d}.json").write_text(json.dumps(job, indent=2) + "\n")
            index["jobs"].append({"config": f"{spec['id']}__v{v:03d}.json", "output_clip": job["output_clip"]})
            dynamic_jobs += spec.get("dynamic", False)

    (args.out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    total = len(spec_paths) * args.variants
    print(f"Wrote {total} render configs ({len(spec_paths)} gestures × {args.variants} variants) -> {args.out}")
    print(f"  {dynamic_jobs} jobs render dynamic gestures, {total - dynamic_jobs} static.")
    print(f"Index: {args.out / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
