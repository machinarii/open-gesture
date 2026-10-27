#!/usr/bin/env python3
"""Fast baseline: a hand-shape classifier trained on HaGRID v2's 2D landmarks.

HaGRID v2 (https://github.com/hukenovs/hagrid) ships MediaPipe hand landmarks
in its annotation zip (719 MB), so a hand-shape model can be trained without
downloading a single image. This is the "fast" half of the HaGRID plan; the
full half re-extracts Holistic landmarks from the images so samples fit the
258-D layout in landmark_format.py.

What it learns is hand SHAPE, not gesture: HaGRID has 2D landmarks only (no z,
no body pose, no motion), so e.g. `stop` is the shape behind both neg-05 and
greet-02. The model predicts HaGRID classes; HAGRID_TO_GESTURES says which of
the 99 gestures each class is a direct match for. Only direct matches are
listed -- partial ones (right shape, missing motion/arm/placement) are left out
on purpose, so the curated-image score is not inflated by them.

Evaluation runs on the 99 curated images via annotations/pose.json (the hand
landmarks the annotate pipeline already extracted), which are never trained on.

LICENCE: HaGRID uses its own reworking of CC BY-SA 4.0 ("not a Creative
Commons license", license/en_us.pdf). Commercial use is allowed; attribution
and share-alike apply to adapted material. Per the design spec HaGRID is NOT
bundled with this repo: keep the annotations outside it, and treat
model/hagrid_hands.pt (gitignored) as HaGRID-derived until a lawyer says
otherwise.

Requires numpy and torch (annotate/.venv has both).

Usage:
    python3 hagrid_hands.py train --annotations /path/to/annotations.zip
    python3 hagrid_hands.py eval
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# Direct matches only: same hand shape, static gesture. Checked against
# HaGRID's images/gestures.png and each gesture's physical_description.
HAGRID_TO_GESTURES = {
    "like": ["affirm-01"],
    "dislike": ["neg-01"],
    "ok": ["affirm-02"],
    "one": ["number-01"],
    "peace": ["number-02", "greet-06"],
    "three": ["number-03"],  # index+middle+ring, thumb holds pinky: as curated
    "four": ["number-04"],
    "palm": ["number-05"],
    "stop": ["neg-05", "greet-02"],
    "mute": ["sil-01"],
    "call": ["prac-02", "prac-07"],
    "rock": ["meme-02"],
    "middle_finger": ["rude-01"],
    "point": ["dir-01"],
    "holy": ["resp-02", "conv-05"],
    "timeout": ["sport-01", "team-02", "urg-08"],
    "take_picture": ["prac-03"],
    "xsign": ["neg-03"],
    "hand_heart": ["love-03"],
    # Touchless Control shapes added for HaGRID (manifest ctrl-01..05).
    # The descriptions are written from HaGRID's class names; check them
    # against images/gestures.png.
    "fist": ["ctrl-01"],
    "grabbing": ["ctrl-02"],
    "grip": ["ctrl-03"],
    "little_finger": ["ctrl-04"],
    "two_up": ["ctrl-05"],
}
NO_GESTURE = "no_gesture"
CLASSES = sorted(HAGRID_TO_GESTURES) + [NO_GESTURE]
TWO_HANDED = {"holy", "timeout", "take_picture", "xsign", "hand_heart"}

HAGRID_ASPECT = 0.75  # w/h of most HaGRID images (3:4 portrait, sampled); not in the annotations
FEATURE_DIM = 2 * 42 + 4
AUG_COPIES = 4  # augmented copies of the training set, drawn once


def features(hands: list[np.ndarray]) -> np.ndarray:
    """One or two (21, 2) hands in aspect-corrected image space -> FEATURE_DIM vector.

    Each hand is wrist-centred and scaled to unit size. Two hands are ordered
    left-to-right by wrist so slot order is stable, and their placement (offset
    in units of the first hand's size, log size ratio) is what separates e.g.
    `timeout` from two unrelated hands.
    """
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    hands = sorted(hands[:2], key=lambda h: h[0, 0])
    scales = []
    for i, h in enumerate(hands):
        rel = h - h[0]
        scales.append(float(np.linalg.norm(rel, axis=1).max()) or 1.0)
        vec[i * 42 : (i + 1) * 42] = (rel / scales[-1]).ravel()
    if len(hands) == 2:
        vec[84:86] = (hands[1][0] - hands[0][0]) / scales[0]
        vec[86] = np.log(scales[1] / scales[0])
    vec[87] = len(hands) - 1
    return vec


def augmented(hands: list[np.ndarray], rng) -> list[np.ndarray]:
    """Random mirror (left vs right hand) and aspect jitter (HaGRID image sizes vary)."""
    stretch = np.array([rng.uniform(0.8, 1.25), 1.0], dtype=np.float32)
    if rng.random() < 0.5:
        stretch[0] *= -1
    return [h * stretch for h in hands]


def load_hagrid(zip_path: Path, split: str, cap: int, rng) -> tuple[list, np.ndarray]:
    """Hand samples for CLASSES from one HaGRID split, at most `cap` per class.

    One-handed classes contribute each hand carrying the class label; two-handed
    classes contribute the pair; every hand labelled no_gesture (in any file
    read) is a negative. Hands without landmarks are skipped.
    """
    z = zipfile.ZipFile(zip_path)
    per_class: dict[str, list] = {c: [] for c in CLASSES}
    for cls in CLASSES:
        for rec in json.loads(z.read(f"annotations/{split}/{cls}.json")).values():
            hands = [
                (label, np.asarray(lm, dtype=np.float32) * [HAGRID_ASPECT, 1.0])
                for label, lm in zip(rec["labels"], rec.get("hand_landmarks") or [])
                if len(lm) == 21
            ]
            if cls in TWO_HANDED:
                pair = [h for label, h in hands if label == cls]
                if len(pair) == 2:
                    per_class[cls].append(pair)
            elif cls != NO_GESTURE:
                per_class[cls] += [[h] for label, h in hands if label == cls]
            per_class[NO_GESTURE] += [[h] for label, h in hands if label == NO_GESTURE]
    samples, y = [], []
    for ci, cls in enumerate(CLASSES):
        pool = per_class[cls]
        samples += [pool[i] for i in rng.permutation(len(pool))[:cap]]
        y += [ci] * min(cap, len(pool))
        print(f"  {split:5} {cls:14} {min(cap, len(pool)):6} of {len(pool)}")
    return samples, np.asarray(y)


def build_model(torch):
    nn = torch.nn
    return nn.Sequential(
        nn.Linear(FEATURE_DIM, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, len(CLASSES)),
    )


def predict(torch, model, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(X)), 1).numpy()


def train(args) -> int:
    import torch

    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    train_s, ytr = load_hagrid(args.annotations, "train", args.cap, rng)
    test_s, yte = load_hagrid(args.annotations, "test", args.cap // 4, rng)
    Xtr = torch.from_numpy(np.stack([features(augmented(s, rng)) for _ in range(AUG_COPIES) for s in train_s]))
    ytr = torch.from_numpy(np.tile(ytr, AUG_COPIES))
    Xte = np.stack([features(s) for s in test_s])

    model = build_model(torch)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 1024):
            idx = perm[i : i + 1024]
            loss = torch.nn.functional.cross_entropy(model(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        acc = (predict(torch, model, Xte).argmax(1) == yte).mean()
        print(f"epoch {epoch + 1:2}  loss {loss.item():.3f}  HaGRID test acc {acc:.3f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES}, args.out / "hagrid_hands.pt")
    print(f"Saved {args.out / 'hagrid_hands.pt'}")
    return 0


def classify_image(torch, model, hands: list[np.ndarray]) -> tuple[str, float]:
    """Best gesture call over each hand alone and the pair; no_gesture if it never wins."""
    candidates = [[h] for h in hands] + ([hands[:2]] if len(hands) >= 2 else [])
    if not candidates:
        return NO_GESTURE, 1.0
    probs = predict(torch, model, np.stack([features(c) for c in candidates]))
    gesture = probs[:, :-1]
    i, j = np.unravel_index(gesture.argmax(), gesture.shape)
    if gesture[i, j] >= probs[i, -1]:
        return CLASSES[j], float(gesture[i, j])
    return NO_GESTURE, float(probs[:, -1].max())


def evaluate(args) -> int:
    import torch

    ckpt = torch.load(args.out / "hagrid_hands.pt")
    if ckpt["classes"] != CLASSES:
        raise SystemExit("error: model was trained with a different class list; retrain")
    model = build_model(torch)
    model.load_state_dict(ckpt["state_dict"])
    pose = json.loads(args.pose.read_text())["records"]
    gestures = json.loads((REPO / "manifest.json").read_text())["gestures"]
    expected = {g: c for c, gs in HAGRID_TO_GESTURES.items() for g in gs}

    hits = covered = fires = uncovered = 0
    for g in gestures:
        rec = pose.get(g["id"], {})
        # Curated images are 1024x1024, so normalised coords are already aspect-correct.
        hands = [np.asarray(h["landmarks"], dtype=np.float32)[:, :2] for h in rec.get("hands", [])]
        pred, conf = classify_image(torch, model, hands)
        want = expected.get(g["id"])
        if want:
            covered += 1
            hits += pred == want
            mark = "ok  " if pred == want else "MISS"
            print(f"{mark} {g['id']:11} {g['name'][:32]:32} want {want:13} got {pred} ({conf:.2f})")
        else:
            uncovered += 1
            fires += pred != NO_GESTURE
    print(f"\nCovered gestures: {hits}/{covered} correct (chance {1 / len(CLASSES):.0%})")
    print(f"Uncovered gestures called as a HaGRID class: {fires}/{uncovered} "
          "(not errors as such -- many share a covered hand shape)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["train", "eval"])
    parser.add_argument("--annotations", type=Path, help="HaGRID v2 annotations.zip (with landmarks)")
    parser.add_argument("--pose", type=Path, default=REPO / "annotations" / "pose.json")
    parser.add_argument("--out", type=Path, default=HERE / "model")
    parser.add_argument("--cap", type=int, default=20000, help="max training samples per class")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.command == "train":
        if not args.annotations:
            parser.error("train needs --annotations")
        return train(args)
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
