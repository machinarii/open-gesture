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
