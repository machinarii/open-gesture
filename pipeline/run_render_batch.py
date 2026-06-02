#!/usr/bin/env python3
"""Orchestrate the BlenderProc render sweep over render_configs/.

Reads render_configs/index.json and invokes `blenderproc run render_clip.py`
once per job. Each render runs in its own BlenderProc process (Blender state
doesn't survive between clips), so this just sequences and guards those calls.

Defaults to a DRY RUN: it plans the work, reports which gestures still have
unauthored poses (those can't render yet), and prints the commands without
executing. Pass --execute to actually render.

Usage:
    # plan only (safe; default)
    python3 run_render_batch.py --smplx-model /path/SMPLX_NEUTRAL.npz --out clips/

    # render for real, skipping clips already produced
    python3 run_render_batch.py --smplx-model /path/SMPLX_NEUTRAL.npz --out clips/ \
        --execute --skip-existing [--limit N]

stdlib only.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def poses_authored(spec_path: Path) -> bool:
    spec = json.loads(spec_path.read_text())
    return all(kf.get("smplx_pose") is not None for kf in spec["keyframes"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--render-configs", type=Path, default=here / "render_configs")
    parser.add_argument("--motion-specs", type=Path, default=here / "motion_specs")
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output dir for clips")
    parser.add_argument("--hdri-dir", type=Path, default=None)
    parser.add_argument("--execute", action="store_true", help="Actually run (default: dry run)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip clips already on disk")
    parser.add_argument("--limit", type=int, default=None, help="Render at most N clips")
    args = parser.parse_args()

    index_path = args.render_configs / "index.json"
    if not index_path.exists():
        sys.exit(f"error: {index_path} missing — run generate_render_configs.py first")
    jobs = json.loads(index_path.read_text())["jobs"]

    blenderproc = shutil.which("blenderproc")
    if args.execute and blenderproc is None:
        sys.exit("error: 'blenderproc' not on PATH. pip install blenderproc (needs Blender).")

    # Classify every gesture's pose-authoring state once (full pass, limit-
    # independent) so the reported blocked counts are always accurate.
    gesture_ids = {job["output_clip"].split("__", 1)[0] for job in jobs}
    authored = {gid: poses_authored(args.motion_specs / f"{gid}.json") for gid in gesture_ids}
    blocked_specs = {gid for gid, ok in authored.items() if not ok}
    blocked_jobs = sum(1 for job in jobs if not authored[job["output_clip"].split("__", 1)[0]])

    planned, skipped_existing = [], 0
    for job in jobs:
        gesture_id = job["output_clip"].split("__", 1)[0]
        if not authored[gesture_id]:
            continue
        if args.skip_existing and (args.out / job["output_clip"]).exists():
            skipped_existing += 1
            continue
        planned.append(job)
        if args.limit and len(planned) >= args.limit:
            break

    print(f"Jobs in index:        {len(jobs)}")
    print(f"Blocked (poses TODO): {blocked_jobs}  across {len(blocked_specs)} gestures")
    print(f"Skipped (exist):      {skipped_existing}")
    print(f"Renderable now:       {len(planned)}{f' (capped at --limit {args.limit})' if args.limit and len(planned) >= args.limit else ''}")
    if blocked_specs:
        sample = ", ".join(sorted(blocked_specs)[:6])
        print(f"  -> author poses for: {sample}{' ...' if len(blocked_specs) > 6 else ''}")

    if not planned:
        print("\nNothing to render. Author SMPL-X poses in motion_specs/ first.")
        return 0

    for job in planned:
        gesture_id = job["output_clip"].split("__", 1)[0]
        cmd = [
            blenderproc or "blenderproc", "run", str(here / "render_clip.py"),
            "--config", str(args.render_configs / job["config"]),
            "--spec", str(args.motion_specs / f"{gesture_id}.json"),
            "--smplx-model", str(args.smplx_model),
            "--out", str(args.out),
        ]
        if args.hdri_dir:
            cmd += ["--hdri-dir", str(args.hdri_dir)]

        if not args.execute:
            print("  [dry-run]", " ".join(cmd))
            continue
        print(f"  rendering {job['output_clip']} ...")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  ! failed ({result.returncode}) on {job['output_clip']}; continuing")

    if not args.execute:
        print(f"\nDry run. Re-run with --execute to render {len(planned)} clips.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
