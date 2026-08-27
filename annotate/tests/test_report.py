import json

import pytest

from open_gesture_annotate.io import Gesture
from open_gesture_annotate.report import (
    check_arousal,
    check_body_parts,
    check_people_count,
    check_similarity,
    check_valence,
    write_report,
)


def _gesture(gid, **raw):
    return Gesture(id=gid, index=1, name="G", category="C", file=f"{gid}.png", raw=raw)


# --- number_of_people ---


def test_single_matching_one_face_is_not_flagged():
    g = [_gesture("a", number_of_people="single")]
    assert check_people_count(g, {"a": {"status": "ok", "face_count": 1}}) == []


def test_single_with_two_faces_is_flagged():
    g = [_gesture("a", number_of_people="single")]
    findings = check_people_count(g, {"a": {"status": "ok", "face_count": 2}})
    assert len(findings) == 1
    assert findings[0].check == "number_of_people"
    assert findings[0].predicted == "2"


def test_three_or_more_with_three_faces_is_not_flagged():
    g = [_gesture("a", number_of_people="3 or more")]
    assert check_people_count(g, {"a": {"status": "ok", "face_count": 4}}) == []


def test_an_error_record_is_skipped_not_flagged():
    g = [_gesture("a", number_of_people="single")]
    assert check_people_count(g, {"a": {"status": "error", "error": "x"}}) == []


def test_a_missing_record_is_skipped():
    assert check_people_count([_gesture("a", number_of_people="single")], {}) == []


# --- valence ---


def test_positive_label_with_positive_valence_is_not_flagged():
    g = [_gesture("a", emotional_state="positive")]
    assert check_valence(g, {"a": {"status": "ok", "valence": 0.6}}) == []


def test_positive_label_with_negative_valence_is_flagged():
    g = [_gesture("a", emotional_state="positive")]
    findings = check_valence(g, {"a": {"status": "ok", "valence": -0.7}})
    assert len(findings) == 1
    assert findings[0].check == "emotional_state"


def test_neutral_label_is_never_flagged_on_sign():
    g = [_gesture("a", emotional_state="neutral")]
    assert check_valence(g, {"a": {"status": "ok", "valence": -0.9}}) == []


def test_a_valence_near_zero_is_not_flagged():
    """Only a confident contradiction counts; near-zero valence is not evidence."""
    g = [_gesture("a", emotional_state="positive")]
    assert check_valence(g, {"a": {"status": "ok", "valence": -0.05}}) == []


# --- arousal ---


# Terciles need at least 3 scored gestures, so these use 6:
# ranks 0-1 -> low, 2-3 -> medium, 4-5 -> high.
BUCKETS = ["low", "low", "medium", "medium", "high", "high"]


def _arousal_case(values):
    gestures = [_gesture(chr(ord("a") + i), arousal=BUCKETS[i]) for i in range(6)]
    va = {g.id: {"status": "ok", "arousal": v} for g, v in zip(gestures, values)}
    return gestures, va


def test_arousal_agreeing_with_the_predicted_terciles_is_not_flagged():
    gestures, va = _arousal_case([0.1, 0.2, 0.4, 0.5, 0.8, 0.9])
    assert check_arousal(gestures, va) == []


def test_arousal_at_the_opposite_extreme_is_flagged():
    """Curated order fully reversed: the two 'low' and two 'high' gestures swap terciles."""
    gestures, va = _arousal_case([0.9, 0.8, 0.5, 0.4, 0.2, 0.1])
    findings = check_arousal(gestures, va)
    assert {f.gesture_id for f in findings} == {"a", "b", "e", "f"}


def test_middle_buckets_are_never_flagged():
    """Only opposite extremes count; low-vs-medium is not a contradiction worth reporting."""
    gestures, va = _arousal_case([0.1, 0.2, 0.4, 0.5, 0.8, 0.9])
    assert all(f.gesture_id not in ("c", "d") for f in check_arousal(gestures, va))


def test_too_few_scored_gestures_yields_no_terciles():
    g = [_gesture("a", arousal="low"), _gesture("b", arousal="high")]
    va = {"a": {"status": "ok", "arousal": 0.9}, "b": {"status": "ok", "arousal": 0.1}}
    assert check_arousal(g, va) == []


# --- body_parts ---


def test_hand_claim_with_a_detected_hand_is_not_flagged():
    g = [_gesture("a", body_parts=["hand", "thumb"])]
    assert check_body_parts(g, {"a": {"status": "ok", "hand_count": 1, "body_detected": True}}) == []


def test_hand_claim_with_no_detected_hand_is_flagged():
    g = [_gesture("a", body_parts=["hand", "thumb"])]
    findings = check_body_parts(g, {"a": {"status": "ok", "hand_count": 0, "body_detected": True}})
    assert len(findings) == 1
    assert findings[0].check == "body_parts"


def test_a_gesture_claiming_no_hand_is_not_flagged():
    g = [_gesture("a", body_parts=["head"])]
    assert check_body_parts(g, {"a": {"status": "ok", "hand_count": 0, "body_detected": True}}) == []


# --- similarity ---


def test_the_bottom_decile_of_similarity_is_flagged():
    gestures = [_gesture(f"g{i}") for i in range(10)]
    emb = {f"g{i}": {"status": "ok", "similarity_intent": i / 10} for i in range(10)}
    findings = check_similarity(gestures, emb)
    assert [f.gesture_id for f in findings] == ["g0"]


# --- report ---


def test_write_report_creates_a_markdown_file(tmp_path, monkeypatch):
    monkeypatch.setattr("open_gesture_annotate.report.load_manifest", lambda root: [])
    path = write_report(tmp_path, tmp_path)
    assert path.name == "quality_report.md"
    assert "# Annotation Quality Report" in path.read_text(encoding="utf-8")


def test_write_report_never_touches_the_manifest(tmp_path, monkeypatch):
    from open_gesture_annotate.io import repo_root

    before = (repo_root() / "manifest.json").read_bytes()
    monkeypatch.setattr("open_gesture_annotate.report.load_manifest", lambda root: [])
    write_report(tmp_path, tmp_path)
    assert (repo_root() / "manifest.json").read_bytes() == before


# --- interpretation notes (fix round 1) ---


def test_write_report_reports_zero_finding_check_coverage(tmp_path, monkeypatch):
    """A check that agrees on everything must say so, with its coverage -- not go silent."""
    gestures = [
        _gesture("a", body_parts=["hand"]),
        _gesture("b", body_parts=["hand", "thumb"]),
    ]
    monkeypatch.setattr("open_gesture_annotate.report.load_manifest", lambda root: gestures)
    pose = {
        "records": {
            "a": {"status": "ok", "hand_count": 1, "body_detected": True},
            "b": {"status": "ok", "hand_count": 2, "body_detected": True},
        }
    }
    (tmp_path / "pose.json").write_text(json.dumps(pose), encoding="utf-8")

    text = write_report(tmp_path, tmp_path).read_text(encoding="utf-8")
    assert "0 finding(s) across 2 gesture(s) considered" in text
    assert "body_parts" in text


def test_write_report_explains_the_facial_affect_proxy_limitation(tmp_path, monkeypatch):
    """The framing must read 'weak proxy', never imply the curated label is wrong."""
    gestures = [_gesture("a", emotional_state="neutral")]
    monkeypatch.setattr("open_gesture_annotate.report.load_manifest", lambda root: gestures)
    faces = {
        "records": {
            "a": {"status": "ok", "face_count": 1, "faces": [{"emotion": {"label": "Neutral"}}]},
        }
    }
    (tmp_path / "faces.json").write_text(json.dumps(faces), encoding="utf-8")

    text = write_report(tmp_path, tmp_path).read_text(encoding="utf-8")
    assert "How to read this report" in text
    assert "weak proxy for gesture affect" in text
    assert "1/1 (100%) of detected faces read as Neutral" in text
    # The report may discuss the concept, but must always explicitly disclaim it --
    # never assert as fact that the curated label is the one that's wrong.
    assert "not evidence the curated label is wrong" in text
    assert "the curated data is wrong" not in text.lower()
    assert "the curated labels are wrong" not in text.lower()
    assert "the manifest is wrong" not in text.lower()


def test_write_report_notes_a_surfaced_duplicate_image_pair(tmp_path, monkeypatch):
    """Two gestures sharing a byte-identical image must be explained, not left as a mystery."""
    n = 20
    gestures = [_gesture(f"g{i}") for i in range(n)]
    monkeypatch.setattr("open_gesture_annotate.report.load_manifest", lambda root: gestures)

    records = {}
    for i in range(n):
        if i < 2:
            vec = [1.0, 2.0]  # g0 and g1 share an identical embedding (duplicate image)
        else:
            vec = [float(i), float(i)]
        records[f"g{i}"] = {
            "status": "ok",
            "similarity_intent": i / n,
            "image": vec,
        }
    embeddings = {"records": records}
    (tmp_path / "embeddings.json").write_text(json.dumps(embeddings), encoding="utf-8")

    text = write_report(tmp_path, tmp_path).read_text(encoding="utf-8")
    assert "## Dataset notes" in text
    assert "`g0`" in text and "`g1`" in text
    assert "dataset property, not an anomaly" in text


def test_write_report_omits_dataset_notes_when_no_duplicate_is_flagged(tmp_path, monkeypatch):
    gestures = [_gesture("a", emotional_state="neutral")]
    monkeypatch.setattr("open_gesture_annotate.report.load_manifest", lambda root: gestures)
    text = write_report(tmp_path, tmp_path).read_text(encoding="utf-8")
    assert "## Dataset notes" not in text
