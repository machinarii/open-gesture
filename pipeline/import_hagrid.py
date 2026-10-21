#!/usr/bin/env python3
"""Full HaGRID path: HaGRID v2 images -> MediaPipe Holistic -> sequences/ shards.

The fast path (hagrid_hands.py) trains on HaGRID's own 2D hand landmarks. This
one re-extracts every sampled image with MediaPipe Holistic through
landmark_format.frame_features -- the same 258-D layout and extraction path as
extract_landmarks.py -- so the shards feed train_classifier.py unchanged.

Images are read straight out of the 119 GB 512px zip with HTTP range requests,
one request per image, so only the sampled images are ever downloaded. The
zip's central directory is cached locally on first use (~1M entries, ~1 min).

A still image is a held pose, so its frame is tiled to --frames: a single
frame followed by zero padding would let the model tell HaGRID samples from
real clips by their length alone.

Labels: HaGRID classes map to gestures via hagrid_hands.HAGRID_TO_GESTURES.
Where one hand shape maps to several gestures (e.g. `stop` -> neg-05, greet-02),
the sample is labelled with the FIRST id: no still image can separate them, so
score predictions by group (eval_curated.py does). HaGRID's `no_gesture` images
become the `none` class (export_labels.NONE).

Resumable: each finished image is logged to <out>/index.partial.jsonl as it
completes, and a re-run skips what the log already holds -- so an interrupted
import picks up where it stopped, and raising --per-class only fetches the new
images (the per-class sample order is fixed by --seed). Fetch failures are not
logged, so they are retried. index.json is written at the end from the log.

--curated writes the 99 curated images through the same extraction instead,
as the held-out test set for eval_curated.py. Never mix those into training.

LICENCE: see hagrid_hands.py -- HaGRID-derived shards stay outside git
(sequences_hagrid_*/ is gitignored) and carry share-alike terms.

Requires mediapipe (with the legacy solutions API, e.g. 0.10.14), opencv-python,
numpy -- annotate/.venv has all three.

Usage:
    python3 import_hagrid.py --annotations annotations.zip --per-class 1000
    python3 import_hagrid.py --curated
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import struct
import sys
import urllib.request
import zipfile
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

import landmark_format as lf
from extract_landmarks import load_label_lookup
from hagrid_hands import HAGRID_TO_GESTURES

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
IMAGES_URL = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/hagridv2_512.zip"
ZIP_PREFIX = "HaGRIDv2_dataset_512"


def fetch_range(url: str, start: int, end: int, tries: int = 4) -> bytes:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
            return urllib.request.urlopen(req, timeout=120).read()
        except OSError:
            if attempt == tries - 1:
                raise
    raise AssertionError("unreachable")


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests; lets zipfile read the central directory."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
        self.size = int(head.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = [offset, self.pos + offset, self.size + offset][whence]
        return self.pos

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        data = fetch_range(self.url, self.pos, self.pos + n)
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)


def remote_index(cache: Path) -> dict[str, tuple[int, int]]:
    """{member name: (local header offset, compressed size)}, cached after the first read."""
    if cache.is_file():
        return pickle.loads(cache.read_bytes())
    print("Reading the remote zip's central directory (once, ~1 min)...")
    zf = zipfile.ZipFile(HttpFile(IMAGES_URL))
    index = {i.filename: (i.header_offset, i.compress_size) for i in zf.infolist()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(index))
    return index


def fetch_member(name: str, offset: int, size: int) -> bytes:
    """One zip member via a single range request: local header, then stored or deflated data.

    The local header's extra field can differ from the central directory's, so
    1 KB of slack is fetched and the real lengths are read from the header.
    """
    blob = fetch_range(IMAGES_URL, offset, offset + 30 + len(name) + 1024 + size)
    sig, _, _, method, _, _, _, _, _, name_len, extra_len = struct.unpack("<IHHHHHIIIHH", blob[:30])
    if sig != 0x04034B50:
        raise ValueError(f"bad local header for {name}")
    start = 30 + name_len + extra_len
    data = blob[start : start + size]
    if method == 8:
        return zlib.decompressobj(-15).decompress(data)
    if method == 0:
        return data
    raise ValueError(f"unsupported zip method {method} for {name}")


_holistic = None


def extract(image_bytes: bytes) -> tuple[list[float], float] | None:
    """(Holistic features, width / height) for one encoded image, or None if no hand was found."""
    global _holistic
    import cv2
    import mediapipe as mp

    if _holistic is None:  # one per worker process
        _holistic = mp.solutions.holistic.Holistic(
            static_image_mode=True, model_complexity=1, refine_face_landmarks=False
        )
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    results = _holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # Every sampled image shows a gesturing hand; without one the label would
    # attach to an undetected frame (cf. extract_landmarks --min-detections).
    if not (results.left_hand_landmarks or results.right_hand_landmarks):
        return None
    return lf.frame_features(results), image.shape[1] / image.shape[0]


# Training label per HaGRID class: the first gesture of each shape group, and
# `none` for HaGRID's no_gesture.
CLASS_TO_GESTURE = {cls: gestures[0] for cls, gestures in HAGRID_TO_GESTURES.items()}
CLASS_TO_GESTURE["no_gesture"] = "none"


def select(annotations: Path, split: str, per_class: int, rng) -> list[tuple[str, str]]:
    """(HaGRID class, image id) pairs: up to per_class images per mapped class.

    Each class's full permutation is drawn before slicing, so the first N
    picks are the same for any per_class >= N -- what makes raising
    --per-class a resume rather than a different sample.
    """
    z = zipfile.ZipFile(annotations)
    picked = []
    # no_gesture last: adding it must not shift the draws of the gesture classes.
    for cls in sorted(HAGRID_TO_GESTURES) + ["no_gesture"]:
        ids = sorted(json.loads(z.read(f"annotations/{split}/{cls}.json")))
        picked += [(cls, ids[i]) for i in rng.permutation(len(ids))[:per_class]]
    return picked


def write_shards(items, out: Path, lookup, head_order, frames: int, workers: int,
                 resume: bool = True, batch: int = 2000) -> None:
    """items: (gesture_id, shard stem, zero-arg callable returning image bytes).

    Works in batches so at most `batch` encoded images are in memory, and logs
    each outcome to index.partial.jsonl as it lands (see module docstring).
    """
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "index.partial.jsonl"
    if not resume:
        log_path.unlink(missing_ok=True)
    done = {}
    if log_path.is_file():
        for line in log_path.read_text().splitlines():
            rec = json.loads(line)
            done[rec["stem"]] = rec
    todo = [item for item in items if item[1] not in done]
    if done:
        print(f"Resuming: {len(items) - len(todo)} of {len(items)} already done")
    failed = 0
    with ThreadPoolExecutor(workers * 4) as fetchers, ProcessPoolExecutor(workers) as extractors, \
            log_path.open("a") as log:
        for start in range(0, len(todo), batch):
            downloads = [(gid, stem, fetchers.submit(load)) for gid, stem, load in todo[start : start + batch]]
            jobs = []
            for gid, stem, dl in downloads:
                try:
                    jobs.append((gid, stem, extractors.submit(extract, dl.result())))
                except Exception as exc:  # one bad download must not sink the run
                    print(f"  fetch failed {stem}: {exc}")
                    failed += 1
            for gid, stem, job in jobs:
                result = job.result()
                if result is None:
                    rec = {"stem": stem, "dropped": True}
                else:
                    feats, aspect = result
                    np.savez_compressed(
                        out / f"{stem}.npz",
                        sequence=np.tile(np.asarray(feats, dtype=np.float32), (frames, 1)),
                        labels=np.asarray(lookup[gid], dtype=np.int32),
                        gesture_id=gid,
                        aspect=np.float32(aspect),
                    )
                    rec = {"stem": stem, "file": f"{stem}.npz", "gesture_id": gid, "frames": frames}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                done[stem] = rec
            print(f"  {min(start + batch, len(todo))}/{len(todo)} processed", flush=True)

    # Only this run's selection, in selection order: a smaller --per-class
    # after a bigger one must not drag the extra shards back in.
    shards = [done[stem] for _, stem, _ in items if stem in done and not done[stem].get("dropped")]
    index = {"head_order": head_order, "feature_dim": lf.FEATURE_DIM,
             "shards": [{k: s[k] for k in ("file", "gesture_id", "frames")} for s in shards]}
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    dropped = sum(1 for _, stem, _ in items if done.get(stem, {}).get("dropped"))
    print(f"\nKept {len(shards)}, dropped {dropped} (no hand detected), fetch failures {failed} (retried on re-run).")
    print(f"Index: {out / 'index.json'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, help="HaGRID v2 annotations.zip")
    parser.add_argument("--curated", action="store_true", help="extract the 99 curated images instead")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--per-class", type=int, default=1000)
    parser.add_argument("--frames", type=int, default=64, help="tile each still to this many frames")
    parser.add_argument("--labels", type=Path, default=HERE / "labels")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cache", type=Path, default=HERE / ".cache" / "hagrid_512_index.pkl")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    head_order, lookup = load_label_lookup(args.labels)

    if args.curated:
        gestures = json.loads((REPO / "manifest.json").read_text())["gestures"]
        items = [(g["id"], g["id"], (REPO / g["file"]).read_bytes) for g in gestures]
        write_shards(items, args.out or HERE / "sequences_curated", lookup, head_order, args.frames, args.workers,
                     resume=False)  # 99 local images: always re-extract
        return 0

    if not args.annotations:
        parser.error("--annotations is required unless --curated")
    index = remote_index(args.cache)
    picked = select(args.annotations, args.split, args.per_class, np.random.default_rng(args.seed))
    items = []
    for cls, image_id in picked:
        name = f"{ZIP_PREFIX}/{cls}/{image_id}.jpg"
        if name not in index:
            print(f"  not in image zip: {name}")
            continue
        offset, size = index[name]
        gid = CLASS_TO_GESTURE[cls]
        items.append((gid, f"{gid}__hagrid_{cls}_{image_id}", lambda n=name, o=offset, s=size: fetch_member(n, o, s)))
    print(f"Fetching and extracting {len(items)} images ({args.split})...")
    write_shards(items, args.out or HERE / f"sequences_hagrid_{args.split}", lookup, head_order,
                 args.frames, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
