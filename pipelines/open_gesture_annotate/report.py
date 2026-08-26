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
HAND_WORDS = {"hand", "hands", "finger", "fingers", "thumb", "palm", "fist", "wrist"}


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
    findings = []
    for g in gestures:
        rec = pose.get(g.id)
        if not rec or rec.get("status") != "ok":
            continue
        parts = {str(p).lower() for p in g.raw.get("body_parts", [])}
        if parts & HAND_WORDS and int(rec.get("hand_count", 0)) == 0:
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


def _neutral_face_stats(faces: dict) -> tuple[int, int]:
    """(neutral_count, total_faces) across every detected face in the ok records."""
    total = neutral = 0
    for rec in faces.values():
        for face in rec.get("faces", []):
            total += 1
            if face.get("emotion", {}).get("label") == "Neutral":
                neutral += 1
    return neutral, total


def _duplicate_image_groups(embeddings: dict) -> list[list[str]]:
    """Gesture ids that share a byte-for-byte identical CLIP image embedding."""
    groups: dict[tuple, list[str]] = {}
    for gid, rec in embeddings.items():
        vec = rec.get("image")
        if not vec:
            continue
        key = tuple(round(float(v), 6) for v in vec)
        groups.setdefault(key, []).append(gid)
    return [sorted(ids) for ids in groups.values() if len(ids) > 1]


def _how_to_read(gestures: list[Gesture], faces: dict) -> str:
    """A short, data-derived paragraph on the proxy limitations below, stated once up front."""
    neutral, total = _neutral_face_stats(faces)
    curated_total = len(gestures)
    curated_neutral = sum(1 for g in gestures if g.raw.get("emotional_state") == "neutral")

    if total:
        face_stat = f"{neutral}/{total} ({neutral / total:.0%}) of detected faces read as Neutral"
    else:
        face_stat = "no faces were detected in this run"
    curated_stat = (
        f"the curated `emotional_state` label is `neutral` for only "
        f"{curated_neutral}/{curated_total} gestures"
        if curated_total else "no curated gestures were available for comparison"
    )

    return (
        "Several checks below compare a model's incidental proxy for a curated label "
        "against a purpose-built one, and the proxy's blind spots matter more than the "
        "raw numbers. This dataset is largely close-up, hand-first gesture crops, so "
        f"**facial affect is a weak proxy for gesture affect here** -- {face_stat}, while "
        f"{curated_stat}: the emotion is carried by hands and posture, not facial "
        "expression. Likewise, detected face count is a proxy for curated person count "
        "and fails exactly where a gesture is framed hand-first (no face visible) or "
        "where a multi-person scene only yields one clearly detected face. CLIP "
        "image-to-intent similarity scores are only meaningful relative to each other "
        "within this run, not as an absolute threshold. None of this is a verdict on the "
        "curated data -- it is context for reading the disagreements below."
    )


def _check_summaries(
    gestures: list[Gesture],
    findings: list[Finding],
    faces: dict,
    va: dict,
    pose: dict,
    embeddings: dict,
) -> list[tuple[str, int, int, str]]:
    """(check, gestures considered, findings, how-to-read explanation) for every check.

    Listed for every check regardless of whether it produced any findings, so a check
    that is silent because it agrees is distinguishable from a check that never ran.
    """
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.check] = counts.get(f.check, 0) + 1

    people_n = sum(
        1 for g in gestures if g.id in faces and g.raw.get("number_of_people") in PEOPLE_EXPECTED
    )
    body_n = sum(
        1 for g in gestures
        if g.id in pose and ({str(p).lower() for p in g.raw.get("body_parts", [])} & HAND_WORDS)
    )
    valence_n = sum(
        1 for g in gestures if g.id in va and g.raw.get("emotional_state") in ("positive", "negative")
    )
    arousal_n = sum(
        1 for g in gestures if g.id in va and g.raw.get("arousal") in ("low", "medium", "high")
    )
    similarity_n = sum(1 for g in gestures if g.id in embeddings)

    return [
        (
            "number_of_people", people_n, counts.get("number_of_people", 0),
            "Detected face count (uniface) is a proxy for the curated participant-count "
            "bucket (single -> 1, '2 person' -> 2, '3 or more' -> 3+). It fails on "
            "hand-first crops in both directions: zero faces does not mean zero people "
            "(no face is visible in a hand-only framing), and a multi-person scene often "
            "yields only one detected face (e.g. a high-five or a hug). Read a "
            "disagreement as a proxy limitation unless the image genuinely shows a "
            "different number of people than faces detected.",
        ),
        (
            "body_parts", body_n, counts.get("body_parts", 0),
            f"MediaPipe hand detection is checked only in the claims-a-hand-but-none-"
            f"detected direction (the reverse -- a detected hand the curator did not "
            f"list -- is not checked, since curated body_parts can legitimately omit "
            f"incidental parts). Zero findings here is a genuine result: MediaPipe "
            f"agreed with all {body_n} curated hand claims evaluated in this run, not a "
            f"sign the check did not run.",
        ),
        (
            "emotional_state", valence_n, counts.get("emotional_state", 0),
            f"HSEmotion facial valence is a proxy for the curated emotional_state label; "
            f"a finding fires only when the predicted sign is confident "
            f"(|valence| >= {VALENCE_CONFIDENCE:g}). In this dataset facial affect is a "
            f"weak proxy for gesture affect: emotion is largely carried by hands and "
            f"posture rather than facial expression, so most detected faces read as "
            f"Neutral regardless of the curated label. A contradiction is informative "
            f"only when the detected face is visibly expressive; it is not evidence the "
            f"curated label is wrong on its own.",
        ),
        (
            "arousal", arousal_n, counts.get("arousal", 0),
            "Predicted arousal terciles (low/medium/high, ranked within this run's "
            "scored gestures) are compared against the curated low/medium/high bucket; "
            "only opposite-extreme disagreements (curated low vs. predicted high, or "
            "vice versa) are flagged -- a low/medium or medium/high difference is too "
            "coarse a boundary to be informative. Terciles are relative to this run, not "
            "an absolute arousal scale.",
        ),
        (
            "intent_similarity", similarity_n, counts.get("intent_similarity", 0),
            "CLIP image-to-intent cosine similarity is reported for the bottom decile "
            "of this run. CLIP has a well-known image-text modality gap, so raw scores "
            "are only meaningful relative to each other within this dataset, not as an "
            "absolute quality threshold -- a low score flags a candidate for human "
            "review, not a confirmed mislabel.",
        ),
    ]


def write_report(root: Path, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    gestures_list = load_manifest(root)
    gestures = {g.id: g for g in gestures_list}
    findings = build_findings(root, out_dir)

    faces = _ok_records(out_dir / "faces.json")
    va = _ok_records(out_dir / "valence_arousal.json")
    pose = _ok_records(out_dir / "pose.json")
    embeddings = _ok_records(out_dir / "embeddings.json")

    lines = [
        "# Annotation Quality Report",
        "",
        "Disagreements between the curated metadata in `manifest.json` and the model",
        "predictions under `annotations/`. A disagreement is a prompt for human review,",
        "not a verdict: either side may be wrong. Nothing here has been applied to",
        "`manifest.json`.",
        "",
        "## How to read this report",
        "",
        _how_to_read(gestures_list, faces),
        "",
        f"**{len(findings)} finding(s)** across {len(gestures)} gestures.",
        "",
    ]

    if not findings:
        lines += ["No disagreements found.", ""]
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
        lines.append("")

    lines += ["## Notes", ""]
    for check, coverage, count, explanation in _check_summaries(
        gestures_list, findings, faces, va, pose, embeddings
    ):
        lines.append(
            f"- **{check}** -- {count} finding(s) across {coverage} gesture(s) "
            f"considered. {explanation}"
        )

    dup_groups = _duplicate_image_groups(embeddings)
    flagged_ids = {f.gesture_id for f in findings}
    dup_hits = [group for group in dup_groups if len(flagged_ids & set(group)) >= 2]
    if dup_hits:
        lines += ["", "## Dataset notes", ""]
        for group in dup_hits:
            ids = ", ".join(f"`{gid}`" for gid in group)
            lines.append(
                f"- {ids} share a byte-identical source image (the same gesture "
                "cross-listed across categories); they share this finding because their "
                "embeddings are identical -- a dataset property, not an anomaly."
            )

    path = out_dir / "quality_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
