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
    python3 eval_curated.py [--model PATH] [--sequences DIR] [--seq-len N]
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
    args = parser.parse_args()

    import tensorflow as tf

    classes = json.loads((args.labels / "label_schema.json").read_text())["heads"]["gesture_id"]["classes"]
    id_of = {i: g for g, i in classes.items()}
    group = {g: set(gs) for gs in HAGRID_TO_GESTURES.values() for g in gs}

    index = json.loads((args.sequences / "index.json").read_text())
    shards = {s["gesture_id"]: s["file"] for s in index["shards"]}
    model = tf.keras.models.load_model(args.model)
    gesture_head = model.output_names.index("gesture_id")

    hits = 0
    for gid in sorted(group):
        if gid not in shards:
            print(f"MISS {gid:11} no hand detected by Holistic")
            continue
        data = np.load(args.sequences / shards[gid])
        seq = data["sequence"][: args.seq_len]
        aspect = float(data["aspect"]) if "aspect" in data.files else 1.0
        x = np.zeros((1, args.seq_len, seq.shape[1]), dtype=np.float32)
        x[0, : len(seq)] = lf.normalize(seq, aspect)
        probs = model.predict(x, verbose=0)[gesture_head][0]
        pred = id_of[int(probs.argmax())]
        ok = pred in group[gid]
        hits += ok
        print(f"{'ok  ' if ok else 'MISS'} {gid:11} got {pred} ({probs.max():.2f})")
    print(f"\nCovered gestures: {hits}/{len(group)} correct (by group)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
