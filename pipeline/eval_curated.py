#!/usr/bin/env python3
"""Score a trained classifier on the 99 curated images (the held-out test set).

Reads model/open_gesture.keras (train_classifier.py) and the shards written by
`import_hagrid.py --curated`, and reports the gesture_id head on the gestures
HaGRID covers. A prediction counts when it lands in the expected gesture's
group: gestures sharing one hand shape (e.g. neg-05 and greet-02 are both
HaGRID `stop`) cannot be told apart from a still image, and the importer
labels each group by its first id.

A covered gesture with no shard (Holistic found no hand) counts as a miss.

Requires tensorflow, numpy.

Usage:
    python3 eval_curated.py [--model PATH] [--sequences DIR] [--seq-len N] [--all]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import landmark_format as lf
from hagrid_hands import HAGRID_TO_GESTURES

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=HERE / "model" / "open_gesture.keras")
    parser.add_argument("--sequences", type=Path, default=HERE / "sequences_curated")
    parser.add_argument("--labels", type=Path, default=HERE / "labels")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--all", action="store_true",
                        help="score every gesture with a shard (e.g. record_gestures.py clips), not just HaGRID's 25")
    args = parser.parse_args()

    import tensorflow as tf

    classes = json.loads((args.labels / "label_schema.json").read_text())["heads"]["gesture_id"]["classes"]
    id_of = {i: g for g, i in classes.items()}
    group = {g: set(gs) for gs in HAGRID_TO_GESTURES.values() for g in gs}

    index = json.loads((args.sequences / "index.json").read_text())
    shards: dict[str, list[str]] = {}
    for s in index["shards"]:
        shards.setdefault(s["gesture_id"], []).append(s["file"])
    model = tf.keras.models.load_model(args.model)
    gesture_head = model.output_names.index("gesture_id")

    hits = total = 0
    for gid in sorted(shards) if args.all else sorted(group):
        if gid not in shards:
            print(f"MISS {gid:11} no hand detected by Holistic")
            total += 1
            continue
        for name in shards[gid]:
            data = np.load(args.sequences / name)
            seq = data["sequence"][: args.seq_len]
            aspect = float(data["aspect"]) if "aspect" in data.files else 1.0
            x = np.zeros((1, args.seq_len, seq.shape[1]), dtype=np.float32)
            x[0, : len(seq)] = lf.normalize(seq, aspect)
            probs = model.predict(x, verbose=0)[gesture_head][0]
            pred = id_of[int(probs.argmax())]
            ok = pred in group.get(gid, {gid})
            hits += ok
            total += 1
            print(f"{'ok  ' if ok else 'MISS'} {gid:11} got {pred} ({probs.max():.2f})  {name}")
    print(f"\n{'All' if args.all else 'Covered'} gestures: {hits}/{total} clips correct (by group)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
