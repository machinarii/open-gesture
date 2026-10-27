#!/usr/bin/env python3
"""Cut IPN Hand videos into per-gesture clips for extract_landmarks.py.

IPN Hand (https://gibranbenitez.github.io/IPN_Hand/) is 13 touchless-control
gestures plus non-gesture segments, recorded by 50 people at 640x480, 30 fps.
LICENCE: CC BY 4.0 ("The data and annotations in the IPN Hand dataset are
licensed under a Creative Commons Attribution 4.0 License") -- commercial use
allowed with attribution. Still kept out of the repo: clips are large.

Each annotated segment becomes one clip named the way extract_landmarks.py
expects, so no new extraction code is needed:

    clips_ipn/ctrl-07__ipn_<video>_<t_start>.mp4

Then:
    python3 extract_landmarks.py --clips clips_ipn --out sequences_ipn --min-hand-detections 0.5
    python3 train_classifier.py --sequences sequences_ipn sequences_hagrid_train  # no --mirror:
    #   throw left/right swap meaning under a mirror

Annotation format assumed (IPN's Annot_*List.txt): a CSV with columns
video,label,id,t_start,t_end,frames, frames 1-indexed and inclusive. The
script checks the header and stops if it differs.

Requires: opencv-python

Usage:
    python3 import_ipn.py --videos DIR --annotations Annot_List.txt [--out clips_ipn]

IPN's official split is by subject: pass --videos-from Annot_TrainList.txt (or
Annot_TestList.txt) to keep only that split's videos, with a separate --out each.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from export_labels import NONE

HERE = Path(__file__).resolve().parent

# IPN label -> manifest gesture id. B0A (point with one finger) is the
# existing dir-01; D0X (non-gesture) feeds the none class.
IPN_TO_GESTURE = {
    "D0X": NONE,
    "B0A": "dir-01",
    "B0B": "ctrl-06",
    "G01": "ctrl-07",
    "G02": "ctrl-08",
    "G03": "ctrl-09",
    "G04": "ctrl-10",
    "G05": "ctrl-11",
    "G06": "ctrl-12",
    "G07": "ctrl-13",
    "G08": "ctrl-14",
    "G09": "ctrl-15",
    "G10": "ctrl-16",
    "G11": "ctrl-17",
}
COLUMNS = {"video", "label", "t_start", "t_end"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", type=Path, required=True, help="dir of IPN videos (any container)")
    parser.add_argument("--annotations", type=Path, required=True, help="IPN Annot_*List.txt")
    parser.add_argument("--out", type=Path, default=HERE / "clips_ipn")
    parser.add_argument("--videos-from", type=Path,
                        help="keep only videos named in the first column of this file (IPN's Annot_TrainList.txt / Annot_TestList.txt)")
    args = parser.parse_args()

    with args.annotations.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or not COLUMNS <= set(rows[0]):
        raise SystemExit(f"error: expected columns {sorted(COLUMNS)}, got {list(rows[0]) if rows else 'nothing'}")

    if args.videos_from:
        keep = {line.split(",")[0] for line in args.videos_from.read_text().splitlines() if line}
        rows = [r for r in rows if r["video"] in keep]
    videos = {p.stem: p for p in args.videos.iterdir() if p.is_file()}
    args.out.mkdir(parents=True, exist_ok=True)
    by_video: dict[str, list[dict]] = {}
    for r in rows:
        by_video.setdefault(r["video"], []).append(r)

    written = skipped = 0
    for video, segs in sorted(by_video.items()):
        if video not in videos:
            print(f"  missing video {video}; skipped {len(segs)} segments")
            skipped += len(segs)
            continue
        todo = []
        for r in segs:
            gid = IPN_TO_GESTURE.get(r["label"])
            if gid is None:
                raise SystemExit(f"error: unknown IPN label {r['label']!r}")
            path = args.out / f"{gid}__ipn_{video}_{r['t_start']}.mp4"
            if not path.exists():  # resumable
                todo.append((int(r["t_start"]), int(r["t_end"]), path))
        if not todo:
            continue

        cap = cv2.VideoCapture(str(videos[video]))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        for start, end, path in sorted(todo):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start - 1)  # 1-indexed
            tmp = args.out / ".partial" / path.name  # outside extract_landmarks.py's view until done
            tmp.parent.mkdir(exist_ok=True)
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            for _ in range(end - start + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
            writer.release()
            tmp.rename(path)
            written += 1
        cap.release()
        print(f"  {video}: {len(todo)} clips")

    print(f"\nWrote {written} clips to {args.out}; skipped {skipped} segments with no video.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
