#!/usr/bin/env python3
"""Generate Cosmos Transfer augmentation jobs from the rendered clips.

Cosmos Transfer photorealistically re-renders each BlenderProc clip into new
environments while preserving the gesture motion -- it's conditioned on the
depth + segmentation control signals BlenderProc already exports, so the body
and hands stay put while the scene becomes a real-looking outdoor / industrial /
AR context. This multiplies domain coverage right where it helps: BEFORE
MediaPipe, so the appearance-sensitive landmark detector sees realistic frames.

For each rendered clip this emits one job per context variant, as a
Cosmos-Transfer1 controlnet spec plus pipeline metadata.

Naming preserves the join contract: an input clip `<id>__v<NNN>.mp4` becomes
`<id>__v<NNN>__cosmos_<variant>.mp4`. The extractor still recovers the gesture
id as everything before the first `__`, so labels join unchanged.

License: each job inherits the source motion spec's license_tier (research-only
motion stays research-only after augmentation). NOTE the Cosmos model itself
carries the NVIDIA Open Model License -- a separate constraint from the motion
data tier; both are recorded on the job.

═══════════════════════════════════════════════════════════════════════════
VERIFY: the controlnet-spec keys (prompt / depth / seg / edge / vis with
control_weight + input_control) follow Cosmos-Transfer1; confirm against your
checkout. Control-signal paths assume BlenderProc wrote depth/ and seg/ next to
the clips -- adjust --depth-dir / --seg-dir to match your render output layout.
═══════════════════════════════════════════════════════════════════════════

Usage:
    python3 generate_cosmos_jobs.py [--render-configs DIR] [--motion-specs DIR]
        [--clips-dir DIR] [--out DIR] [--variants NAME...]

stdlib only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Context variants: each re-renders the same motion into a new real environment.
# Prompts are written for the deployment contexts the README targets.
CONTEXT_VARIANTS = {
    "outdoor": "A photorealistic person performing a hand gesture on a busy city sidewalk, "
               "overcast daylight, pedestrians and traffic in the background, candid street scene.",
    "industrial": "A photorealistic worker performing a hand gesture on a factory floor, "
                  "high-visibility vest, fluorescent lighting, machinery and shelving behind them.",
    "ar_indoor": "A photorealistic person performing a hand gesture in a modern living room, "
                 "soft natural window light, seen as if through AR glasses, shallow depth of field.",
    "low_light": "A photorealistic person performing a hand gesture outdoors at dusk, "
                 "dim ambient light, mixed streetlight color, slight motion blur, security-camera look.",
}
NEGATIVE_PROMPT = ("cartoon, illustration, 3d render, cgi, plastic skin, deformed hands, "
                   "extra fingers, blurry, low quality, watermark, text")

# Control weights: depth + segmentation anchor the body/hands so the gesture
# survives the restyle; edge/vis are lighter so the scene can change freely.
CONTROL_WEIGHTS = {"depth": 0.5, "seg": 0.4, "edge": 0.2, "vis": 0.2}


def spec_tier(motion_specs: Path, gesture_id: str) -> str:
    spec_path = motion_specs / f"{gesture_id}.json"
    if spec_path.exists():
        return json.loads(spec_path.read_text()).get("license_tier", "untagged")
    return "untagged"


def build_job(clip: str, variant: str, prompt: str, tier: str, clips_dir: str,
              depth_dir: str, seg_dir: str) -> dict:
    stem = clip[:-4] if clip.endswith(".mp4") else clip
    out_clip = f"{stem}__cosmos_{variant}.mp4"
    return {
        "input_clip": clip,
        "output_clip": out_clip,
        "variant": variant,
        "license_tier": tier,
        "model_license": "NVIDIA-Open-Model-License",
        # Cosmos-Transfer1 controlnet spec:
        "cosmos_spec": {
            "prompt": prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "input_video_path": f"{clips_dir}/{clip}",
            "depth": {"input_control": f"{depth_dir}/{stem}.mp4", "control_weight": CONTROL_WEIGHTS["depth"]},
            "seg": {"input_control": f"{seg_dir}/{stem}.mp4", "control_weight": CONTROL_WEIGHTS["seg"]},
            "edge": {"control_weight": CONTROL_WEIGHTS["edge"]},
            "vis": {"control_weight": CONTROL_WEIGHTS["vis"]},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--render-configs", type=Path, default=here / "render_configs")
    parser.add_argument("--motion-specs", type=Path, default=here / "motion_specs")
    parser.add_argument("--out", type=Path, default=here / "cosmos_jobs")
    parser.add_argument("--clips-dir", default="clips", help="Path (as seen by Cosmos) to input clips")
    parser.add_argument("--depth-dir", default="clips/depth", help="Path to BlenderProc depth control videos")
    parser.add_argument("--seg-dir", default="clips/seg", help="Path to BlenderProc segmentation control videos")
    parser.add_argument("--variants", nargs="*", default=list(CONTEXT_VARIANTS),
                        help=f"Context variants per clip (default: all of {list(CONTEXT_VARIANTS)})")
    args = parser.parse_args()

    unknown = [v for v in args.variants if v not in CONTEXT_VARIANTS]
    if unknown:
        raise SystemExit(f"error: unknown variant(s) {unknown}; choose from {list(CONTEXT_VARIANTS)}")

    index_path = args.render_configs / "index.json"
    if not index_path.exists():
        raise SystemExit(f"error: {index_path} missing — run generate_render_configs.py first")
    clips = [job["output_clip"] for job in json.loads(index_path.read_text())["jobs"]]

    args.out.mkdir(parents=True, exist_ok=True)
    index = {"variants": args.variants, "jobs": [], "tier_counts": {}}
    tier_counts: dict[str, int] = {}

    for clip in clips:
        gesture_id = clip.split("__", 1)[0]
        tier = spec_tier(args.motion_specs, gesture_id)
        for variant in args.variants:
            job = build_job(clip, variant, CONTEXT_VARIANTS[variant], tier,
                            args.clips_dir, args.depth_dir, args.seg_dir)
            (args.out / f"{job['output_clip'][:-4]}.json").write_text(json.dumps(job, indent=2) + "\n")
            index["jobs"].append({"job": f"{job['output_clip'][:-4]}.json",
                                  "output_clip": job["output_clip"], "license_tier": tier})
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

    index["tier_counts"] = tier_counts
    (args.out / "index.json").write_text(json.dumps(index, indent=2) + "\n")

    total = len(clips) * len(args.variants)
    print(f"Wrote {total} Cosmos jobs ({len(clips)} clips × {len(args.variants)} variants) -> {args.out}")
    print(f"  by license tier: {tier_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
