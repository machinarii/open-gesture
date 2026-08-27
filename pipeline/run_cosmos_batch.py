#!/usr/bin/env python3
"""Orchestrate Cosmos Transfer augmentation over the generated jobs (RunPod).

Reads cosmos_jobs/index.json and runs Cosmos Transfer once per job. Cosmos needs
a GPU + the model weights, so this is meant to run on a RunPod pod; locally it
defaults to a DRY RUN that plans the work and reports the license-tier split.

The actual invocation is a configurable command template (--cmd) so it adapts to
however your Cosmos-Transfer1 checkout is launched. Tokens substituted per job:
    {spec}   absolute path to the job's controlnet-spec JSON
    {out}    absolute path to the output clip

Because every job carries its source motion's license_tier, --tier lets you
render ONE tier at a time -- e.g. build the commercial dataset without ever
touching research-only clips.

Usage:
    # plan only (default)
    python3 run_cosmos_batch.py --out augmented/

    # render just the commercial tier, skipping done clips
    python3 run_cosmos_batch.py --out augmented/ --tier commercial \
        --execute --skip-existing \
        --cmd "python cosmos_transfer1/inference.py --spec {spec} --output {out}"

stdlib only.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--cosmos-jobs", type=Path, default=here / "cosmos_jobs")
    parser.add_argument("--out", type=Path, required=True, help="Output dir for augmented clips")
    parser.add_argument("--tier", choices=["commercial", "non-commercial", "untagged"],
                        help="Only run jobs of this license tier (default: all)")
    parser.add_argument("--cmd", default="python cosmos_transfer1/inference.py --spec {spec} --output {out}",
                        help="Command template; {spec} and {out} are substituted per job")
    parser.add_argument("--execute", action="store_true", help="Actually run (default: dry run)")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    index_path = args.cosmos_jobs / "index.json"
    if not index_path.exists():
        sys.exit(f"error: {index_path} missing — run generate_cosmos_jobs.py first")
    jobs = json.loads(index_path.read_text())["jobs"]

    # Full-pass tier tally (limit-independent), then plan.
    tier_totals: dict[str, int] = {}
    for job in jobs:
        tier_totals[job["license_tier"]] = tier_totals.get(job["license_tier"], 0) + 1

    planned, skipped = [], 0
    for job in jobs:
        if args.tier and job["license_tier"] != args.tier:
            continue
        if args.skip_existing and (args.out / job["output_clip"]).exists():
            skipped += 1
            continue
        planned.append(job)
        if args.limit and len(planned) >= args.limit:
            break

    print(f"Jobs in index:   {len(jobs)}   by tier: {tier_totals}")
    if args.tier:
        print(f"Tier filter:     {args.tier}")
    print(f"Skipped (exist): {skipped}")
    print(f"Planned:         {len(planned)}{f' (capped at --limit {args.limit})' if args.limit and len(planned) >= args.limit else ''}")

    if not planned:
        print("\nNothing to run.")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    for job in planned:
        spec = args.cosmos_jobs / job["job"]
        out_clip = args.out / job["output_clip"]
        cmd = args.cmd.format(spec=str(spec), out=str(out_clip))
        if not args.execute:
            print("  [dry-run]", cmd)
            continue
        print(f"  augmenting {job['output_clip']} ({job['license_tier']}) ...")
        result = subprocess.run(shlex.split(cmd))
        if result.returncode != 0:
            print(f"  ! failed ({result.returncode}) on {job['output_clip']}; continuing")

    if not args.execute:
        print(f"\nDry run. Re-run with --execute to augment {len(planned)} clips.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
