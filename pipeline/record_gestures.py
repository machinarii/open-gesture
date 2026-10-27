#!/usr/bin/env python3
"""Record your own test clips of the 99 gestures with a webcam.

HaGRID covers 25 of the 99 gestures, and the curated test set has one still
image each, so it cannot tell models apart by more than noise. This walks the
manifest one gesture at a time: it shows the gesture's name, description and
reference image over a live preview, and records a short clip when you press
space. Clips are named the way extract_landmarks.py expects:

    recordings/affirm-01__rec_<who>_<take>.mp4

A single camera is deliberate: inference is one camera into MediaPipe, so the
test clips should be too. Existing clips are skipped, so re-running resumes.
Multi-person gestures (number_of_people != single) are skipped unless
--include-multi. A `none` clip (hands idle) is recorded last for the none class.

Keys: space = record, s = skip, q = quit.
The preview is mirrored for comfort; the saved video is not, so
direction-sensitive gestures keep their real side.

Then:
    python3 extract_landmarks.py --clips recordings --out sequences_recorded
    python3 eval_curated.py --sequences sequences_recorded --all

Requires: opencv-python, numpy
"""
from __future__ import annotations

import argparse
import json
import textwrap
import time
from pathlib import Path

import cv2
import numpy as np

from export_labels import NONE

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def overlay(frame, lines, ref):
    """Draw the prompt text (and the reference image, top right) on a frame."""
    h, w = frame.shape[:2]
    if ref is not None:
        rw = w // 4
        rh = min(h, int(ref.shape[0] * rw / ref.shape[1]))
        frame[:rh, w - rw:] = cv2.resize(ref, (rw, rh))
    y = 30
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 24
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=HERE / "recordings")
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifest.json")
    parser.add_argument("--who", default="me", help="performer tag in filenames (no '__')")
    parser.add_argument("--takes", type=int, default=1, help="clips per gesture")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--only", nargs="*", help="gesture ids to record (default: all)")
    parser.add_argument("--include-multi", action="store_true", help="also prompt for multi-person gestures")
    args = parser.parse_args()
    if "__" in args.who:
        parser.error("--who must not contain '__' (it separates the gesture id)")

    gestures = json.loads(args.manifest.read_text())["gestures"]
    if not args.include_multi:
        gestures = [g for g in gestures if g["number_of_people"] == "single"]
    gestures.append({"id": NONE, "name": "No gesture",
                     "physical_description": "Hands relaxed or idle: at your sides, resting, fidgeting"})
    if args.only:
        gestures = [g for g in gestures if g["id"] in args.only]

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"error: cannot open camera {args.camera}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.out.mkdir(parents=True, exist_ok=True)

    recorded = 0
    for n, g in enumerate(gestures, 1):
        todo = [args.out / f"{g['id']}__rec_{args.who}_{t}.mp4" for t in range(args.takes)]
        todo = [p for p in todo if not p.exists()]
        if not todo:
            continue
        ref = cv2.imread(str(ROOT / g["file"])) if g.get("file") else None
        desc = textwrap.wrap(g["physical_description"], 70)
        for path in todo:
            head = [f"[{n}/{len(gestures)}] {g['id']}  {g['name']}  ({path.stem.rsplit('_', 1)[1]})"]
            key = -1
            while key not in (ord(" "), ord("s"), ord("q")):
                ok, frame = cap.read()
                if not ok:
                    raise SystemExit("error: camera stopped")
                cv2.imshow("record", overlay(cv2.flip(frame, 1), head + desc + ["space=record  s=skip  q=quit"], ref))
                key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print(f"\nRecorded {recorded} clips; re-run to resume.")
                return 0
            if key == ord("s"):
                break  # skips this gesture's remaining takes

            end = time.time() + 2  # countdown so the hands are in place
            while time.time() < end:
                ok, frame = cap.read()
                if ok:
                    cv2.imshow("record", overlay(cv2.flip(frame, 1), head + [f"get ready... {end - time.time():.0f}"], ref))
                cv2.waitKey(1)

            # Write elsewhere, then move: a crash never leaves a half clip
            # where extract_landmarks.py (top-level files only) would read it.
            tmp = args.out / ".partial" / path.name
            tmp.parent.mkdir(exist_ok=True)
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            end = time.time() + args.seconds
            while time.time() < end:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)  # unmirrored
                shown = cv2.flip(frame, 1)
                cv2.circle(shown, (w - 20, h - 20), 10, (0, 0, 255), -1)
                cv2.imshow("record", overlay(shown, head, None))
                cv2.waitKey(1)
            writer.release()
            tmp.rename(path)
            recorded += 1
            print(f"saved {path.name}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nRecorded {recorded} clips into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
