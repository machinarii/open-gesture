#!/usr/bin/env python3
"""Train the multi-head landmark-sequence classifier and export INT8 TFLite.

Reads the .npz shards from extract_landmarks.py and the head definitions from
labels/label_schema.json, builds a small temporal CNN over the landmark
sequence, and trains one softmax head per metadata dimension (gesture id,
category, emotional state, arousal, gesture type, cultural context).

Why this architecture: the input is a sequence of 258-D landmark vectors, not
pixels. A stack of 1D (temporal) convolutions is tiny, fast, and quantizes
cleanly to INT8 -- far cheaper than anything that needs the raw image. The
heavy runtime cost lives in MediaPipe on the host, not here (see README's Hailo
note); this model is small enough that the Hailo-8L is optional.

Sequences are padded/truncated to a fixed length so the whole batch is one
dense tensor (also what the INT8 converter and Hailo expect -- static shapes).

Export path:
    Keras -> TFLite INT8 (full-integer, representative dataset)
    For Hailo-8L: TFLite/ONNX -> Hailo Dataflow Compiler -> .hef
    (Hailo does NOT run .tflite directly; the .tflite here is the compiler input
     and also runs as-is on CPU/Coral for comparison.)

Requires: tensorflow, numpy  (see requirements.txt)

Usage:
    python3 train_classifier.py [--sequences DIR [DIR ...]] [--labels DIR] [--out DIR]
                                [--seq-len N] [--epochs N] [--batch N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import landmark_format as lf
from extract_landmarks import load_label_lookup


def _require(module: str, pip_name: str):
    try:
        return __import__(module)
    except ImportError:
        sys.exit(
            f"error: '{module}' is required but not installed.\n"
            f"       pip install {pip_name}   (or: pip install -r requirements.txt)"
        )


def load_dataset(np, sequences_dirs: list[Path], seq_len: int, labels_dir: Path):
    """Load shards into X [N, seq_len, FEATURE_DIM] and per-head label arrays.

    Labels come from each shard's gesture_id looked up in the CURRENT labels/,
    not the indices stored at extraction: adding gestures shifts the indices,
    and stale ones would train silently on the wrong classes.

    Sequences are right-padded with zeros or truncated to seq_len. Zero padding
    is consistent with how frame_features zero-fills undetected groups, so the
    model sees padding as 'nothing detected' rather than a spurious pose.
    """
    head_order, lookup = load_label_lookup(labels_dir)
    shards = []  # (dir, entry) across every --sequences dir, loaded into one array
    for d in sequences_dirs:
        entries = json.loads((d / "index.json").read_text())["shards"]
        if not entries:
            sys.exit(f"error: no shards listed in {d / 'index.json'}")
        shards += [(d, e) for e in entries]

    X = np.zeros((len(shards), seq_len, lf.FEATURE_DIM), dtype=np.float32)
    Y = np.zeros((len(shards), len(head_order)), dtype=np.int32)

    for i, (d, entry) in enumerate(shards):
        data = np.load(d / entry["file"])
        seq = data["sequence"][:seq_len]
        aspect = float(data["aspect"]) if "aspect" in data.files else 1.0
        X[i, : len(seq)] = lf.normalize(seq, aspect)  # inference must apply the same
        Y[i] = lookup[entry["gesture_id"]]

    return X, Y, head_order


def build_model(keras, layers, seq_len: int, heads: dict):
    """Temporal CNN trunk + one softmax head per metadata dimension."""
    inp = keras.Input(shape=(seq_len, lf.FEATURE_DIM), name="landmarks")
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = [
        layers.Dense(spec["num_classes"], activation="softmax", name=name)(x)
        for name, spec in heads.items()
    ]
    return keras.Model(inp, outputs, name="open_gesture_multihead")


def representative_dataset(np, X):
    """Generator over real samples so the INT8 converter calibrates activations."""
    def gen():
        for i in range(X.shape[0]):
            yield [X[i : i + 1].astype(np.float32)]
    return gen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--sequences", type=Path, nargs="+", default=[here / "sequences"],
                        help="Dir(s) of .npz + index.json; several are trained on together")
    parser.add_argument("--labels", type=Path, default=here / "labels", help="labels/ dir (for head class counts)")
    parser.add_argument("--out", type=Path, default=here / "model", help="Output dir for model + tflite")
    parser.add_argument("--seq-len", type=int, default=64, help="Frames per clip (pad/truncate) (default: 64)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--mirror", action="store_true",
                        help="add left-right mirrored copies; only when no gesture's meaning depends on side")
    parser.add_argument("--val-sequences", type=Path, nargs="+",
                        help="validate on these dir(s) -- people never in --sequences -- instead of a random 15%%")
    args = parser.parse_args()

    np = _require("numpy", "numpy")
    tf = _require("tensorflow", "tensorflow")
    keras, layers = tf.keras, tf.keras.layers

    schema = json.loads((args.labels / "label_schema.json").read_text())
    heads = schema["heads"]

    X, Y, head_order = load_dataset(np, args.sequences, args.seq_len, args.labels)
    heads = {name: heads[name] for name in head_order}
    print(f"Dataset: {X.shape[0]} clips, seq_len={args.seq_len}, feature_dim={lf.FEATURE_DIM}")
    print(f"Heads: {[(n, heads[n]['num_classes']) for n in head_order]}")

    model = build_model(keras, layers, args.seq_len, heads)
    model.compile(
        optimizer="adam",
        loss={name: "sparse_categorical_crossentropy" for name in head_order},
        metrics={name: "accuracy" for name in head_order},
    )

    # Keras's validation_split takes the LAST 15% without shuffling, and shards
    # load sorted by gesture id -- unshuffled, whole gestures would be held out
    # and never trained on.
    perm = np.random.default_rng(0).permutation(len(X))
    X, Y = X[perm], Y[perm]
    fit_kwargs = {}
    if args.val_sequences:
        # Held-out PEOPLE, not a random slice: a random slice shares people
        # (and oversampled copies) with training and reads optimistic.
        Xv, Yv, _ = load_dataset(np, args.val_sequences, args.seq_len, args.labels)
        print(f"Validation: {Xv.shape[0]} clips from {[str(d) for d in args.val_sequences]}")
        fit_kwargs["validation_data"] = (Xv, {name: Yv[:, i] for i, name in enumerate(head_order)})
        if args.mirror:
            X, Y = np.concatenate([X, lf.mirror(X)]), np.concatenate([Y, Y])
    elif args.mirror:
        # After the shuffle, so the held-out tail gets mirrors of its own samples
        # only -- appending mirrors before it would leak training samples into it.
        cut = int(len(X) * 0.85)
        X = np.concatenate([X[:cut], lf.mirror(X[:cut]), X[cut:]])
        Y = np.concatenate([Y[:cut], Y[:cut], Y[cut:]])
        cut2 = 2 * cut
        tail = np.random.default_rng(1).permutation(cut2)
        X[:cut2], Y[:cut2] = X[tail], Y[tail]
        fit_kwargs["validation_split"] = (len(X) - cut2) / len(X)
    else:
        fit_kwargs["validation_split"] = 0.15
    y_split = {name: Y[:, i] for i, name in enumerate(head_order)}
    args.out.mkdir(parents=True, exist_ok=True)
    keras_path = args.out / "open_gesture.keras"
    if args.val_sequences:
        # Held-out-people accuracy swings by up to 30 points between epochs, so
        # the last epoch is a lottery: keep the best one, and halve the learning
        # rate when it stalls to damp the swings.
        watch = dict(monitor="val_gesture_id_accuracy", mode="max")
        fit_kwargs["callbacks"] = [
            keras.callbacks.ModelCheckpoint(str(keras_path), save_best_only=True, **watch),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, **watch),
        ]
    history = model.fit(X, y_split, epochs=args.epochs, batch_size=args.batch, **fit_kwargs)

    if args.val_sequences:
        model = keras.models.load_model(keras_path)
        best = max(history.history["val_gesture_id_accuracy"])
        print(f"Kept the best epoch: val_gesture_id_accuracy {best:.3f}")
    model.save(keras_path)
    print(f"Saved {keras_path}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(np, X)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_path = args.out / "open_gesture_int8.tflite"
    tflite_path.write_bytes(converter.convert())
    print(f"Saved {tflite_path}")
    print("\nFor Hailo-8L: feed this .tflite (or an ONNX export) to the Hailo")
    print("Dataflow Compiler to produce a .hef. Hailo does not run .tflite directly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
