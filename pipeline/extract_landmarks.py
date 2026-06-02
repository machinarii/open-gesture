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

An index.json ties the shards together and records the head order so the
trainer reads labels positionally without guessing.

IMPORTANT ordering: run this AFTER Cosmos Transfer. MediaPipe is appearance-
sensitive, so it must see the photorealistic frames, not the raw Blender
renders -- otherwise landmark detection degrades and the labels attach to
garbage sequences. See README.

Requires: mediapipe, opencv-python, numpy  (see requirements.txt)

Usage:
    python3 extract_landmarks.py --clips DIR [--labels DIR] [--out DIR]
                                 [--min-detections FRAC] [--id-sep STR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import landmark_format as lf


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
    args = parser.parse_args()

    np = _require("numpy", "numpy")
    cv2 = _require("cv2", "opencv-python")
    mp = _require("mediapipe", "mediapipe")

    head_order, lookup = load_label_lookup(args.labels)
    args.out.mkdir(parents=True, exist_ok=True)

    clips = sorted(p for p in args.clips.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"})
    if not clips:
        sys.exit(f"error: no video clips found in {args.clips}")

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=False,  # face is dropped from the feature layout
    )

    index = {"head_order": head_order, "feature_dim": lf.FEATURE_DIM, "shards": []}
    kept = dropped = 0

    for clip in clips:
        gesture_id = clip.stem.split(args.id_sep, 1)[0]
        if gesture_id not in lookup:
            print(f"  skip {clip.name}: gesture id '{gesture_id}' not in labels")
            dropped += 1
            continue

        cap = cv2.VideoCapture(str(clip))
        frames, detected = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
                detected += 1
            frames.append(lf.frame_features(results))
        cap.release()

        if not frames or detected / len(frames) < args.min_detections:
            rate = (detected / len(frames)) if frames else 0.0
            print(f"  drop {clip.name}: detection rate {rate:.0%} < {args.min_detections:.0%}")
            dropped += 1
            continue

        shard = args.out / f"{clip.stem}.npz"
        np.savez_compressed(
            shard,
            sequence=np.asarray(frames, dtype=np.float32),
            labels=np.asarray(lookup[gesture_id], dtype=np.int32),
            gesture_id=gesture_id,
        )
        index["shards"].append({"file": shard.name, "gesture_id": gesture_id, "frames": len(frames)})
        kept += 1

    holistic.close()
    (args.out / "index.json").write_text(json.dumps(index, indent=2) + "\n")

    print(f"\nKept {kept} clips, dropped {dropped}.")
    print(f"Head order: {head_order}")
    print(f"Index: {args.out / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
