#!/usr/bin/env python3
"""Fabricate a synthetic sequences/ dir to smoke-test the training stage.

Stands in for the BlenderProc -> Cosmos -> MediaPipe path so the training and
INT8-export wiring can be validated WITHOUT real video or MediaPipe installed.
It writes the exact same on-disk format extract_landmarks.py produces (.npz
shards + index.json), so the thing under test is the real train_classifier.py,
not a reimplementation.

The landmark values are random noise with a small per-gesture bias, so the
model can reach above-chance accuracy in a few epochs -- enough to prove the
heads, losses, and INT8 converter all wire up. It is NOT a real signal.

Usage:
    python3 make_smoke_data.py [--labels DIR] [--out DIR]
                               [--clips-per-gesture N] [--gestures N]
                               [--min-frames N] [--max-frames N] [--seed N]
    # then:
    python3 train_classifier.py --sequences sequences_smoke --epochs 3 --seq-len 48

Requires: numpy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import landmark_format as lf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--labels", type=Path, default=here / "labels")
    parser.add_argument("--out", type=Path, default=here / "sequences_smoke")
    parser.add_argument("--gestures", type=int, default=12, help="How many distinct gestures to sample")
    parser.add_argument("--clips-per-gesture", type=int, default=8)
    parser.add_argument("--min-frames", type=int, default=40)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        import numpy as np
    except ImportError:
        sys.exit("error: numpy required. pip install -r requirements.txt")

    schema = json.loads((args.labels / "label_schema.json").read_text())
    head_order = list(schema["heads"].keys())

    records = [json.loads(line) for line in (args.labels / "labels.jsonl").open()]
    records = records[: args.gestures]

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    index = {"head_order": head_order, "feature_dim": lf.FEATURE_DIM, "shards": []}
    for rec in records:
        gid = rec["id"]
        labels = np.asarray([rec["labels"][h] for h in head_order], dtype=np.int32)
        # Per-gesture bias vector so clips of the same gesture are separable.
        bias = rng.normal(0, 1, size=lf.FEATURE_DIM).astype(np.float32)
        for c in range(args.clips_per_gesture):
            n_frames = int(rng.integers(args.min_frames, args.max_frames + 1))
            seq = (rng.normal(0, 0.1, size=(n_frames, lf.FEATURE_DIM)) + bias).astype(np.float32)
            name = f"{gid}__smoke{c:02d}"
            np.savez_compressed(
                args.out / f"{name}.npz",
                sequence=seq,
                labels=labels,
                gesture_id=gid,
            )
            index["shards"].append({"file": f"{name}.npz", "gesture_id": gid, "frames": n_frames})

    (args.out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"Wrote {len(index['shards'])} smoke clips for {len(records)} gestures -> {args.out}")
    print("Next: python3 train_classifier.py --sequences", args.out.name, "--epochs 3 --seq-len 48")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
