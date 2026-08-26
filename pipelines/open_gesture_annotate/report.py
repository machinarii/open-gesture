"""Cross-check curated metadata against model predictions.

Reports disagreements. Asserts nothing about which side is wrong — a disagreement
is a prompt for human review, not an automated correction. This module never
writes to manifest.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from open_gesture_annotate.io import Gesture, load_manifest

VALENCE_CONFIDENCE = 0.15  # below this magnitude, valence is not evidence either way
PEOPLE_EXPECTED = {"single": 1, "2 person": 2, "3 or more": 3}


@dataclass
class Finding:
    gesture_id: str
    check: str
    severity: str  # "high" | "medium" | "low"
    curated: str
    predicted: str
    note: str


def _ok_records(path: Path) -> dict:
    """Load a sidecar's successful records, or {} if the sidecar is absent."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {gid: r for gid, r in data.get("records", {}).items() if r.get("status") == "ok"}


def check_people_count(gestures: list[Gesture], faces: dict) -> list[Finding]:
    findings = []
    for g in gestures:
        rec = faces.get(g.id)
        if not rec or rec.get("status") != "ok":
            continue
        curated = g.raw.get("number_of_people")
        expected = PEOPLE_EXPECTED.get(curated)
        if expected is None:
            continue
        detected = int(rec.get("face_count", 0))
        matches = detected >= 3 if curated == "3 or more" else detected == expected
        if not matches:
            findings.append(Finding(
                g.id, "number_of_people", "high", str(curated), str(detected),
                "detected face count disagrees with the curated participant bucket",
            ))
    return findings


def check_valence(gestures: list[Gesture], va: dict) -> list[Finding]:
    findings = []
    for g in gestures:
        rec = va.get(g.id)
        if not rec or rec.get("status") != "ok":
            continue
        curated = g.raw.get("emotional_state")
        if curated not in ("positive", "negative"):
            continue  # neutral makes no sign claim
        valence = float(rec.get("valence", 0.0))
        if abs(valence) < VALENCE_CONFIDENCE:
            continue  # not a confident contradiction
        predicted = "positive" if valence > 0 else "negative"
        if predicted != curated:
            findings.append(Finding(
                g.id, "emotional_state", "medium", curated, f"{predicted} ({valence:+.2f})",
                "predicted facial valence sign contradicts the curated label",
            ))
    return findings


def check_arousal(gestures: list[Gesture], va: dict) -> list[Finding]:
    """Compare the curated low/medium/high bucket against predicted terciles."""
    scored = [(g, float(va[g.id]["arousal"]))
              for g in gestures
              if va.get(g.id, {}).get("status") == "ok"
              and g.raw.get("arousal") in ("low", "medium", "high")]
    if len(scored) < 3:
        return []

    ordered = sorted(scored, key=lambda pair: pair[1])
    third = max(1, len(ordered) // 3)
    tercile = {}
    for rank, (g, _) in enumerate(ordered):
        tercile[g.id] = "low" if rank < third else ("high" if rank >= 2 * third else "medium")

    findings = []
    for g, value in scored:
        curated = g.raw["arousal"]
        predicted = tercile[g.id]
        if {curated, predicted} == {"low", "high"}:  # only flag opposite extremes
            findings.append(Finding(
                g.id, "arousal", "medium", curated, f"{predicted} ({value:+.2f})",
                "curated arousal bucket sits at the opposite extreme of the predicted tercile",
            ))
    return findings


def check_body_parts(gestures: list[Gesture], pose: dict) -> list[Finding]:
    hand_words = {"hand", "hands", "finger", "fingers", "thumb", "palm", "fist", "wrist"}
    findings = []
    for g in gestures:
        rec = pose.get(g.id)
        if not rec or rec.get("status") != "ok":
            continue
        parts = {str(p).lower() for p in g.raw.get("body_parts", [])}
        if parts & hand_words and int(rec.get("hand_count", 0)) == 0:
            findings.append(Finding(
                g.id, "body_parts", "high", ", ".join(sorted(parts)), "0 hands detected",
                "curated body_parts claims a hand but MediaPipe detected none",
            ))
    return findings


def check_similarity(gestures: list[Gesture], embeddings: dict) -> list[Finding]:
    """Flag the bottom decile of image-to-intent similarity."""
    scored = [(g, float(embeddings[g.id]["similarity_intent"]))
              for g in gestures
              if embeddings.get(g.id, {}).get("status") == "ok"]
    if len(scored) < 10:
        return []
    ordered = sorted(scored, key=lambda pair: pair[1])
    cutoff = max(1, len(ordered) // 10)
    return [
        Finding(g.id, "intent_similarity", "low", g.raw.get("intent", ""), f"{score:.3f}",
                "image-to-intent similarity is in the bottom decile; possible weak or "
                "mislabelled image")
        for g, score in ordered[:cutoff]
    ]


def build_findings(root: Path, out_dir: Path) -> list[Finding]:
    gestures = load_manifest(root)
    out_dir = Path(out_dir)
    faces = _ok_records(out_dir / "faces.json")
    va = _ok_records(out_dir / "valence_arousal.json")
    pose = _ok_records(out_dir / "pose.json")
    embeddings = _ok_records(out_dir / "embeddings.json")

    findings = (
        check_people_count(gestures, faces)
        + check_body_parts(gestures, pose)
        + check_valence(gestures, va)
        + check_arousal(gestures, va)
        + check_similarity(gestures, embeddings)
    )
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(findings, key=lambda f: (order[f.severity], f.check, f.gesture_id))


def write_report(root: Path, out_dir: Path) -> Path:
    gestures = {g.id: g for g in load_manifest(root)}
    findings = build_findings(root, out_dir)

    lines = [
        "# Annotation Quality Report",
        "",
        "Disagreements between the curated metadata in `manifest.json` and the model",
        "predictions under `annotations/`. A disagreement is a prompt for human review,",
        "not a verdict: either side may be wrong. Nothing here has been applied to",
        "`manifest.json`.",
        "",
        f"**{len(findings)} finding(s)** across {len(gestures)} gestures.",
        "",
    ]

    if not findings:
        lines.append("No disagreements found.")
    else:
        lines += [
            "| Severity | Check | Gesture | Curated | Predicted | Image |",
            "|---|---|---|---|---|---|",
        ]
        for f in findings:
            gesture = gestures.get(f.gesture_id)
            image = f"`{gesture.file}`" if gesture else ""
            curated = str(f.curated).replace("|", "\\|")[:60]
            lines.append(
                f"| {f.severity} | {f.check} | `{f.gesture_id}` | {curated} "
                f"| {f.predicted} | {image} |"
            )
        lines += ["", "## Notes", ""]
        for check in sorted({f.check for f in findings}):
            note = next(f.note for f in findings if f.check == check)
            lines.append(f"- **{check}** — {note}")

    path = Path(out_dir) / "quality_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
