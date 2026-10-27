#!/usr/bin/env python3
"""Render manifest.md (the human-readable catalogue) from manifest.json.

manifest.json is the hand-edited source; re-run this after editing it.
Sections follow the manifest's `categories` order, gestures by `index`.

Usage:
    python3 render_manifest_md.py [--manifest PATH] [--out PATH]

stdlib only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def render(m: dict) -> str:
    gestures = m["gestures"]
    out = [
        "# Open Gesture Library", "",
        f"**{len(gestures)} gestures** across **{len(m['categories'])} categories**", "",
        f"Generated: {m['generated_at']}", "",
        "## Metadata Schema", "",
        "| Field | Description |",
        "|-------|-------------|",
        *(f"| `{k}` | {v} |" for k, v in m["schema"].items()),
        "", "---", "",
    ]
    for cat in m["categories"]:
        out += [f"## {cat}", ""]
        for g in sorted((g for g in gestures if g["category"] == cat), key=lambda g: g["index"]):
            out += [
                f"### {g['index']}. {g['name']}", "",
                f"![]({g['file']})", "",
                f"- **ID:** `{g['id']}`",
                f"- **Physical Description:** {g['physical_description']}",
                f"- **Intent:** {g['intent']}",
                f"- **Emotional State:** {g['emotional_state']} | **Arousal:** {g['arousal']}",
                f"- **Emotions:** {', '.join(g['emotional_adjectives'])}",
                f"- **Gesture Type:** {g['gesture_type']}",
                f"- **Body Parts:** {', '.join(g['body_parts'])}",
                f"- **People:** {g['number_of_people']} | **Culture:** {g['cultural_context']}",
                "",
            ]
        out += ["---", ""]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifest.json")
    parser.add_argument("--out", type=Path, default=ROOT / "manifest.md")
    args = parser.parse_args()
    args.out.write_text(render(json.loads(args.manifest.read_text())))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
