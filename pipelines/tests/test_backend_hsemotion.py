import json

import numpy as np
import pytest

pytest.importorskip("hsemotion")

import open_gesture_annotate.backends.affect_hsemotion as hsemotion_backend
from open_gesture_annotate.backends.affect_hsemotion import (
    EXPECTED_SCORE_LEN,
    HSEmotionBackend,
    split_scores,
)
from open_gesture_annotate.io import Gesture, load_image, load_manifest, repo_root
from open_gesture_annotate.schema import validate_record

CLASSES = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]


def test_backend_identity():
    b = HSEmotionBackend()
    assert b.name == "hsemotion-va"
    assert b.sidecar == "valence_arousal.json"


def test_split_scores_separates_emotions_from_valence_arousal():
    scores = np.arange(10, dtype=np.float32)
    emotions, valence, arousal = split_scores(scores, CLASSES)
    assert len(emotions) == 8
    assert emotions["Anger"] == 0.0
    assert (valence, arousal) == (8.0, 9.0)


def test_split_scores_rejects_an_unexpected_vector_length():
    with pytest.raises(ValueError, match="expected 10"):
        split_scores(np.arange(8, dtype=np.float32), CLASSES)


def test_expected_length_is_eight_emotions_plus_valence_arousal():
    assert EXPECTED_SCORE_LEN == 10


def test_upstream_class_count_still_matches():
    """Guard on the undocumented layout this backend depends on.

    Built via _build_recognizer() rather than calling HSEmotionRecognizer
    directly: this installation's torch (2.13.0) defaults torch.load's
    weights_only to True (torch >=2.6), which cannot unpickle the va_mtl
    checkpoint's full pickled model instance. _build_recognizer() is the
    documented, scoped workaround this backend already needs for annotate()
    to run at all -- see the module docstring.
    """
    fer = hsemotion_backend._build_recognizer()
    assert len(fer.idx_to_class) == 8


def test_available_reports_true_when_installed():
    assert HSEmotionBackend().available()[0] is True


def test_crop_falls_back_to_full_image_when_faces_json_absent(tmp_path, monkeypatch):
    """CONTROLLER RULING addition: no annotations/faces.json at all must never
    raise, and must fall back to running on the whole image.
    """
    monkeypatch.setattr(hsemotion_backend, "repo_root", lambda: tmp_path)
    b = HSEmotionBackend()
    image = np.zeros((50, 60, 3), dtype=np.uint8)
    gesture = Gesture("x", 1, "x", "c", "x.png", {})

    crop, face_source = b._crop_for_recognition(image, gesture)

    assert face_source == "full-image"
    assert crop is image


def test_crop_uses_uniface_bbox_when_face_record_exists(tmp_path, monkeypatch):
    """CONTROLLER RULING addition: when Task 7's faces.json has a face for this
    gesture, crop to its (expanded, clamped) bbox and set face_source='uniface'.
    Also exercises the read-once cache: faces.json is only parsed on first use.
    """
    monkeypatch.setattr(hsemotion_backend, "repo_root", lambda: tmp_path)
    (tmp_path / "annotations").mkdir()
    faces_sidecar = {
        "_backend": {"name": "uniface-face", "version": "0", "run_at": "now"},
        "records": {
            "x": {
                "status": "ok",
                "face_count": 1,
                "faces": [{"bbox": [10, 10, 30, 30]}],
            }
        },
    }
    (tmp_path / "annotations" / "faces.json").write_text(json.dumps(faces_sidecar))

    b = HSEmotionBackend()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    gesture = Gesture("x", 1, "x", "c", "x.png", {})

    crop, face_source = b._crop_for_recognition(image, gesture)
    assert face_source == "uniface"
    assert crop.shape[0] < image.shape[0]
    assert crop.shape[1] < image.shape[1]

    # Cached: a second call must not re-read the file. Break the file, then
    # confirm behaviour is unchanged because the parsed index is cached.
    (tmp_path / "annotations" / "faces.json").write_text("not json")
    crop2, face_source2 = b._crop_for_recognition(image, gesture)
    assert face_source2 == "uniface"
    assert crop2.shape == crop.shape


@pytest.mark.slow
def test_annotates_a_real_image():
    b = HSEmotionBackend()
    gestures = {g.id: g for g in load_manifest(repo_root())}
    g = gestures["affirm-01"]
    rec = b.annotate(load_image(repo_root(), g), g)
    validate_record(rec)
    assert isinstance(rec["valence"], float)
    assert isinstance(rec["arousal"], float)
    assert len(rec["emotion_scores"]) == 8


@pytest.mark.slow
def test_annotates_using_the_real_faces_json_and_reports_uniface_source():
    """The dataset's real annotations/faces.json has a face_count 1 record for
    affirm-01, so the real (non-monkeypatched) run must report face_source
    'uniface', not the full-image fallback.
    """
    b = HSEmotionBackend()
    gestures = {g.id: g for g in load_manifest(repo_root())}
    g = gestures["affirm-01"]
    rec = b.annotate(load_image(repo_root(), g), g)
    validate_record(rec)
    assert rec["face_source"] == "uniface"


@pytest.mark.slow
def test_faceless_gesture_falls_back_to_full_image_and_stays_ok():
    """bicycle-01 has face_count 0 in the real faces.json (genuinely faceless
    hand-only crop). Must still be status 'ok' via the full-image fallback.
    """
    b = HSEmotionBackend()
    gestures = {g.id: g for g in load_manifest(repo_root())}
    g = gestures["bicycle-01"]
    rec = b.annotate(load_image(repo_root(), g), g)
    validate_record(rec)
    assert rec["status"] == "ok"
    assert rec["face_source"] == "full-image"
