#!/usr/bin/env python3
"""Export training labels from manifest.json.

Turns the gesture manifest into the per-head label schema the multi-head
TFLite classifier trains against. Produces two artifacts:

  label_schema.json   Class -> index maps for every prediction head. This is
                      the single source of truth shared by the trainer and the
                      on-device inference code; both must encode/decode labels
                      through the SAME indices, so the maps are written once
                      here and never recomputed downstream.

  labels.jsonl        One record per gesture: the integer-encoded label for
                      each head plus the raw metadata, ready to be attached to
                      the landmark sequences coming out of MediaPipe.

The heads mirror the README's multi-label design (gesture ID + emotional
state + arousal + gesture type + cultural context), with `category` added as
a coarse grouping head.

Note on PAD: the manifest carries valence (`emotional_state`) and `arousal`
but NOT dominance, so this exports PA, not full PAD. Add a `dominance` field
to the manifest if you want the third head.

Usage:
    python3 export_labels.py [--manifest PATH] [--out-dir DIR]

stdlib only -- runs anywhere Python 3 does.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Each head: (manifest field, output key). Order is cosmetic only; class
# indices come from the sorted vocabulary built per head below.
HEADS = [
    ("id", "gesture_id"),
    ("category", "category"),
    ("emotional_state", "emotional_state"),
    ("arousal", "arousal"),
    ("gesture_type", "gesture_type"),
    ("cultural_context", "cultural_context"),
]


def build_vocab(gestures: list[dict], field: str) -> dict[str, int]:
    """Deterministic class->index map for one head.

    Sorted so the mapping is stable across runs and machines -- the trainer on
    the Mac Studio and a re-export on RunPod must agree on indices, otherwise
    a model's head silently points at the wrong classes.
    """
    values = sorted({str(g[field]) for g in gestures})
    return {value: idx for idx, value in enumerate(values)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "manifest.json",
        help="Path to manifest.json (default: repo root)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "labels",
        help="Output directory (default: pipeline/labels/)",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    gestures = manifest["gestures"]

    vocabs = {out_key: build_vocab(gestures, field) for field, out_key in HEADS}

    args.out_dir.mkdir(parents=True, exist_ok=True)

    schema = {
        "source_manifest": str(args.manifest.name),
        "source_generated_at": manifest.get("generated_at"),
        "total_gestures": len(gestures),
        "heads": {
            out_key: {
                "source_field": field,
                "num_classes": len(vocabs[out_key]),
                "classes": vocabs[out_key],
            }
            for field, out_key in HEADS
        },
        "notes": {
            "pad": "valence + arousal only; no dominance in manifest (PA, not PAD)",
        },
    }
    schema_path = args.out_dir / "label_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2) + "\n")

    labels_path = args.out_dir / "labels.jsonl"
    with labels_path.open("w") as fh:
        for g in gestures:
            record = {
                "id": g["id"],
                "name": g["name"],
                "file": g["file"],
                "labels": {
                    out_key: vocabs[out_key][str(g[field])] for field, out_key in HEADS
                },
                "meta": {
                    "intent": g["intent"],
                    "emotional_adjectives": g["emotional_adjectives"],
                    "body_parts": g["body_parts"],
                    "number_of_people": g["number_of_people"],
                },
            }
            fh.write(json.dumps(record) + "\n")

    print(f"Wrote {schema_path}")
    print(f"Wrote {labels_path}  ({len(gestures)} records)")
    print("\nHeads:")
    for _, out_key in HEADS:
        print(f"  {out_key:18s} {len(vocabs[out_key]):3d} classes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
