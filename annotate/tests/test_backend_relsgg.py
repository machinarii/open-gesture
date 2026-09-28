"""Region derivation and contact logic, tested without relsgg installed.

relations_relsgg imports relsgg lazily (inside _model()), so everything this
file exercises runs on a plain checkout — which is the point: the region
geometry is this backend's own logic, not the model's.
"""

import numpy as np

from open_gesture_annotate.backends.relations_relsgg import (
    BOX_PAD,
    GESTURE_PREDICATES,
    RelSGGBackend,
    has_self_contact,
    top_contact_score,
    regions_from_pose,
)

W, H = 200, 100


def _body(visible=True, vis=0.9):
    """33 BlazePose landmarks spread across the frame, all sighted or all not."""
    v = vis if visible else 0.0
    return {"landmarks": [[i / 33, i / 66, 0.0, v] for i in range(33)]}


def _hand(side="Left"):
    return {"handedness": side, "score": 0.99,
            "landmarks": [[0.5 + i / 200, 0.5 + i / 400, 0.0] for i in range(21)]}


def test_backend_identity():
    b = RelSGGBackend()
    assert b.name == "relsgg-relations"
    assert b.sidecar == "relations.json"


def test_regions_cover_hands_and_body_parts():
    rec = {"status": "ok", "hands": [_hand("Left")], "body": _body()}
    labels = [r["label"] for r in regions_from_pose(rec, W, H)]
    assert labels == ["left hand", "head", "torso", "left arm", "right arm"]


def test_boxes_are_pixels_inside_the_frame():
    rec = {"status": "ok", "hands": [_hand()], "body": _body()}
    for region in regions_from_pose(rec, W, H):
        x1, y1, x2, y2 = region["box"]
        assert 0 <= x1 < x2 <= W
        assert 0 <= y1 < y2 <= H


def test_box_is_padded_beyond_the_raw_landmark_extent():
    rec = {"status": "ok", "hands": [], "body": _body()}
    torso = next(r for r in regions_from_pose(rec, W, H) if r["label"] == "torso")
    raw_left = min(i / 33 for i in (11, 12, 23, 24)) * W
    assert torso["box"][0] == raw_left - BOX_PAD * W


def test_invisible_body_landmarks_yield_no_body_regions():
    rec = {"status": "ok", "hands": [_hand()], "body": _body(visible=False)}
    assert [r["kind"] for r in regions_from_pose(rec, W, H)] == ["hand"]


def test_a_degenerate_region_is_dropped_not_emitted_as_a_zero_area_box():
    """Every landmark at the same point: padding still gives area, so force the
    collapse with a zero-size frame -- the guard must drop it, not emit x2 <= x1.
    """
    rec = {"status": "ok", "hands": [_hand()], "body": _body()}
    assert regions_from_pose(rec, 0, 0) == []


def test_missing_or_failed_pose_record_yields_no_regions():
    assert regions_from_pose(None, W, H) == []
    assert regions_from_pose({"status": "error", "error": "boom"}, W, H) == []


def test_hand_touching_head_is_self_contact_in_either_direction():
    forward = [{"subject_kind": "hand", "object_kind": "body",
                "predicate": "touching", "score": 0.9}]
    reverse = [{"subject_kind": "body", "object_kind": "hand",
                "predicate": "covering", "score": 0.9}]
    assert has_self_contact(forward)
    assert has_self_contact(reverse)


def test_adjacency_is_not_contact():
    """The model ranking `beside` over `touching` for a pair is a statement that
    this is not contact -- one predicate per pair is all predict() returns.
    """
    adjacent = [{"subject_kind": "hand", "object_kind": "body",
                 "predicate": "beside", "score": 0.99},
                {"subject_kind": "hand", "object_kind": "body",
                 "predicate": "above", "score": 0.99}]
    assert not has_self_contact(adjacent)


def test_a_weak_score_still_counts_as_contact():
    """No absolute gate: over the 99 gestures the median triplet score is 0.027,
    so a probability-shaped threshold would discard real contact. The score is
    reported separately for judging strength.
    """
    weak = [{"subject_kind": "hand", "object_kind": "body",
             "predicate": "touching", "score": 0.05}]
    assert has_self_contact(weak)
    assert top_contact_score(weak) == 0.05


def test_top_contact_score_ignores_non_contact_and_non_hand_body():
    mixed = [{"subject_kind": "hand", "object_kind": "body",
              "predicate": "beside", "score": 0.99},
             {"subject_kind": "hand", "object_kind": "hand",
              "predicate": "touching", "score": 0.98},
             {"subject_kind": "hand", "object_kind": "body",
              "predicate": "touching", "score": 0.30}]
    assert top_contact_score(mixed) == 0.30


def test_top_contact_score_is_zero_without_contact():
    assert top_contact_score([{"subject_kind": "hand", "object_kind": "body",
                               "predicate": "beside", "score": 0.9}]) == 0.0


def test_hand_to_hand_contact_is_not_body_contact():
    both_hands = [{"subject_kind": "hand", "object_kind": "hand",
                   "predicate": "touching", "score": 0.99}]
    assert not has_self_contact(both_hands)


def test_vocabulary_has_no_duplicates():
    """Synonyms compete for rank in relsgg; exact duplicates would split scores
    for nothing.
    """
    assert len(GESTURE_PREDICATES) == len(set(GESTURE_PREDICATES))


def test_unavailable_without_pose_sidecar(tmp_path):
    b = RelSGGBackend()
    b.set_output_dir(tmp_path)
    available, reason = b.available()
    assert available is False
    assert "pose.json" in reason or "relsgg not installed" in reason


def test_annotate_with_fewer_than_two_regions_records_ok_not_error(tmp_path):
    """A hand-first crop with no visible body is normal framing in this dataset."""
    b = RelSGGBackend()
    b.set_output_dir(tmp_path)
    (tmp_path / "pose.json").write_text(
        '{"_backend":{"name":"x","version":"0","run_at":"t"},'
        '"records":{"g-01":{"status":"ok","hands":[],"body":null}}}'
    )
    from open_gesture_annotate.io import Gesture

    rec = b.annotate(np.zeros((H, W, 3), np.uint8),
                     Gesture(id="g-01", index=1, name="G", category="C", file="g.png", raw={}))
    assert rec["status"] == "ok"
    assert rec["triplets"] == []
    assert rec["self_contact"] is False



def test_no_predicate_is_both_contact_and_alternative():
    """Every contact predicate must be in the vocabulary, and the vocabulary's
    non-contact strings must be genuine alternatives to contact rather than
    things that are true at the same time as it -- see GESTURE_PREDICATES on why
    argmax-per-pair makes that distinction load-bearing.
    """
    from open_gesture_annotate.backends.relations_relsgg import CONTACT_PREDICATES

    assert CONTACT_PREDICATES <= set(GESTURE_PREDICATES)
    masking = {"near", "in front of", "close to", "next to"}
    assert not (set(GESTURE_PREDICATES) & masking)
