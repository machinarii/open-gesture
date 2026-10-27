#!/usr/bin/env python3
"""Extract landmark sequences from rendered/augmented gesture clips.

This is the MediaPipe Holistic stage. It runs the SAME extraction path used at
real inference time, so the classifier trains on exactly the representation it
will see in production -- that's what lets synthetic RGB transfer to real use.

Input: a directory of video clips named so the gesture id is recoverable, e.g.

    clips/affirm-01__lighting3_angle12_occl0.mp4
          ^^^^^^^^  gesture id (everything before the first '__')

Each clip is decoded frame by frame, run through MediaPipe Holistic, flattened
via landmark_format.frame_features (the shared layout), and saved as one .npz:

    sequence  float32 [T, FEATURE_DIM]   landmark sequence
    labels    int32   [num_heads]        from labels/label_schema.json, in a
                                          fixed head order recorded in the index
    gesture_id str                       source gesture
    aspect    float32                    frame width / height (landmark_format.normalize)

An index.json ties the shards together and records the head order so the
trainer reads labels positionally without guessing.

IMPORTANT ordering: run this AFTER Cosmos Transfer. MediaPipe is appearance-
sensitive, so it must see the photorealistic frames, not the raw Blender
renders -- otherwise landmark detection degrades and the labels attach to
garbage sequences. See README.

Requires: mediapipe, opencv-python, numpy  (see requirements.txt)

Usage:
    python3 extract_landmarks.py --clips DIR [--labels DIR] [--out DIR]
                                 [--min-detections FRAC] [--id-sep STR] [--workers N]

Resumable: a clip whose shard already exists in --out is skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import landmark_format as lf
from export_labels import NONE


def _require(module: str, pip_name: str):
    try:
        return __import__(module)
    except ImportError:
        sys.exit(
            f"error: '{module}' is required but not installed.\n"
            f"       pip install {pip_name}   (or: pip install -r requirements.txt)"
        )


def load_label_lookup(labels_dir: Path):
    """Return (head_order, {gesture_id: [int label per head]})."""
    schema = json.loads((labels_dir / "label_schema.json").read_text())
    head_order = list(schema["heads"].keys())

    lookup: dict[str, list[int]] = {}
    with (labels_dir / "labels.jsonl").open() as fh:
        for line in fh:
            rec = json.loads(line)
            lookup[rec["id"]] = [rec["labels"][h] for h in head_order]
    return head_order, lookup


def extract_clip(clip: Path, shard: Path, gesture_id: str, labels: list[int],
                 min_detections: float, min_hands: float) -> str | None:
    """Run Holistic over one clip and write its shard; return a drop reason or None.

    The worker writes the shard itself so only a short string crosses back to
    the parent: returning the frames let finished clips pile up in the parent's
    memory while it waited on a slow one (gigabytes on IPN).

    A fresh Holistic per clip: in tracking mode it carries state between
    frames, so a shared instance would start each clip tracking the last one.
    """
    import cv2
    import mediapipe as mp
    import numpy as np

    cap = cv2.VideoCapture(str(clip))
    aspect = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / max(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 1.0)
    frames, detected, hands = [], 0, 0
    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=False,  # face is dropped from the feature layout
    ) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
                detected += 1
            if results.left_hand_landmarks or results.right_hand_landmarks:
                hands += 1
            frames.append(lf.frame_features(results))
    cap.release()
    n = max(len(frames), 1)
    if detected / n < min_detections:
        return f"detection rate {detected / n:.0%} < {min_detections:.0%}"
    if hands / n < min_hands:
        return f"hand detection rate {hands / n:.0%} < {min_hands:.0%}"
    tmp = shard.with_suffix(".partial.npz")  # rename after write: a kill never leaves a half shard
    np.savez_compressed(
        tmp,
        sequence=np.asarray(frames, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int32),
        gesture_id=gesture_id,
        aspect=np.float32(aspect),
    )
    tmp.rename(shard)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--clips", type=Path, required=True, help="Directory of video clips")
    parser.add_argument("--labels", type=Path, default=here / "labels", help="labels/ dir from export_labels.py")
    parser.add_argument("--out", type=Path, default=here / "sequences", help="Output dir for .npz shards")
    parser.add_argument("--id-sep", default="__", help="Separator after gesture id in filenames (default: __)")
    parser.add_argument(
        "--min-detections",
        type=float,
        default=0.5,
        help="Drop a clip if fewer than this fraction of frames had any hand/pose "
        "landmarks -- guards against attaching labels to undetected sequences (default: 0.5)",
    )
    parser.add_argument(
        "--min-hand-detections",
        type=float,
        default=0.0,
        help="Also drop a clip if fewer than this fraction of frames had a hand -- for hand-gesture "
        "datasets where pose alone passes --min-detections. Not applied to the none class (default: 0)",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="clips processed in parallel (default: half the CPUs)")
    args = parser.parse_args()

    np = _require("numpy", "numpy")
    _require("cv2", "opencv-python")
    _require("mediapipe", "mediapipe")

    head_order, lookup = load_label_lookup(args.labels)
    args.out.mkdir(parents=True, exist_ok=True)

    clips = sorted(p for p in args.clips.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"})
    if not clips:
        sys.exit(f"error: no video clips found in {args.clips}")

    todo, dropped = [], 0
    for clip in clips:
        gesture_id = clip.stem.split(args.id_sep, 1)[0]
        if gesture_id not in lookup:
            print(f"  skip {clip.name}: gesture id '{gesture_id}' not in labels")
            dropped += 1
        elif not (args.out / f"{clip.stem}.npz").exists():  # resumable: finished shards are kept
            todo.append((clip, gesture_id))
    print(f"{len(clips) - len(todo) - dropped} clips already extracted, {len(todo)} to go")

    # A fresh pool per batch recycles the workers: each clip builds a fresh
    # Holistic, and a long-lived worker holds on to whatever those leave behind.
    # (Not max_tasks_per_child: it deadlocked here mid-run on Python 3.12.)
    batch = 25 * args.workers
    for start in range(0, len(todo), batch):
        with ProcessPoolExecutor(args.workers) as pool:
            jobs = {pool.submit(extract_clip, clip, args.out / f"{clip.stem}.npz", gid, lookup[gid],
                                args.min_detections, 0.0 if gid == NONE else args.min_hand_detections): clip
                    for clip, gid in todo[start : start + batch]}
            for job in as_completed(jobs):
                reason = job.result()
                if reason:
                    print(f"  drop {jobs[job].name}: {reason}")
                    dropped += 1
        print(f"  {min(start + batch, len(todo))}/{len(todo)}", flush=True)

    # Index every finished shard for this clip set, including ones from earlier runs.
    index = {"head_order": head_order, "feature_dim": lf.FEATURE_DIM, "shards": []}
    for clip in clips:
        shard = args.out / f"{clip.stem}.npz"
        if shard.exists():
            with np.load(shard) as d:
                index["shards"].append({"file": shard.name, "gesture_id": str(d["gesture_id"]),
                                        "frames": int(d["sequence"].shape[0])})
    (args.out / "index.json").write_text(json.dumps(index, indent=2) + "\n")

    print(f"\nKept {len(index['shards'])} clips, dropped {dropped}.")
    print(f"Head order: {head_order}")
    print(f"Index: {args.out / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
