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


def test_crop_falls_back_to_full_image_when_repo_root_default_has_no_faces_json(tmp_path, monkeypatch):
    """CONTROLLER RULING addition: no annotations/faces.json at all must never
    raise, and must fall back to running on the whole image. Existing fallback
    coverage for the never-called-the-hook path (repo_root()/annotations
    default), preserved alongside the set_output_dir-based test below.
    """
    monkeypatch.setattr(hsemotion_backend, "repo_root", lambda: tmp_path)
    b = HSEmotionBackend()
    image = np.zeros((50, 60, 3), dtype=np.uint8)
    gesture = Gesture("x", 1, "x", "c", "x.png", {})

    crop, face_source = b._crop_for_recognition(image, gesture)

    assert face_source == "full-image"
    assert crop is image


def test_crop_falls_back_to_full_image_when_set_output_dir_has_no_faces_json(tmp_path):
    """Same fallback guarantee, but exercised via the set_output_dir hook
    (Fix 3) instead of monkeypatching repo_root -- this is the code path the
    runner actually uses for `og-annotate run --out ...`.
    """
    b = HSEmotionBackend()
    b.set_output_dir(tmp_path)
    image = np.zeros((50, 60, 3), dtype=np.uint8)
    gesture = Gesture("x", 1, "x", "c", "x.png", {})

    crop, face_source = b._crop_for_recognition(image, gesture)

    assert face_source == "full-image"
    assert crop is image


def test_set_output_dir_reads_faces_json_from_out_dir_not_the_repo_one(tmp_path):
    """Regression for Fix 3: with a temp --out directory containing its own
    faces.json, `va` must read THAT file, not annotations/faces.json in the
    repo. Uses gesture id 'affirm-01', which the real repo faces.json records
    as face_count 1 (uniface source) -- this temp faces.json instead records
    it as face_count 0, so a full-image fallback here can only happen if the
    temp file, not the repo one, was actually read.
    """
    faces_sidecar = {
        "_backend": {"name": "uniface-face", "version": "0", "run_at": "now"},
        "records": {"affirm-01": {"status": "ok", "face_count": 0, "faces": []}},
    }
    (tmp_path / "faces.json").write_text(json.dumps(faces_sidecar))

    b = HSEmotionBackend()
    b.set_output_dir(tmp_path)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    gesture = Gesture("affirm-01", 1, "x", "c", "x.png", {})

    crop, face_source = b._crop_for_recognition(image, gesture)

    assert face_source == "full-image"
    assert crop is image


def test_available_is_false_when_faces_json_is_wholly_missing(tmp_path):
    """Fix 4: a wholly missing faces.json (e.g. `--backends va` alone, or
    `--backends va,face` -- the face->va ordering is otherwise an alphabetical
    accident) must make the backend unavailable with a reason naming the fix,
    rather than silently degrading every record to full-image inference.
    """
    b = HSEmotionBackend()
    b.set_output_dir(tmp_path)  # tmp_path has no faces.json at all

    available, reason = b.available()

    assert available is False
    assert "faces.json" in reason
    assert "face" in reason  # names the fix: run the face backend first


def test_available_is_true_and_falls_back_per_image_when_faces_json_has_zero_faces(tmp_path):
    """Fix 4: a *present* faces.json with face_count 0 for a gesture (e.g. the
    six gestures that genuinely have no face: bicycle-01, bicycle-03, dir-02,
    meme-01, prac-03, urg-02) must NOT make the backend unavailable, and must
    still fall back to full-image inference for that one image.
    """
    faces_sidecar = {
        "_backend": {"name": "uniface-face", "version": "0", "run_at": "now"},
        "records": {"x": {"status": "ok", "face_count": 0, "faces": []}},
    }
    (tmp_path / "faces.json").write_text(json.dumps(faces_sidecar))

    b = HSEmotionBackend()
    b.set_output_dir(tmp_path)

    available, _ = b.available()
    assert available is True

    image = np.zeros((50, 60, 3), dtype=np.uint8)
    gesture = Gesture("x", 1, "x", "c", "x.png", {})
    crop, face_source = b._crop_for_recognition(image, gesture)
    assert face_source == "full-image"
    assert crop is image


def test_run_backend_reports_unavailable_for_an_out_dir_with_no_faces_json(gestures, image_root, tmp_path):
    """Ordering regression: runner.run_backend must call set_output_dir BEFORE
    available(), not after. If it called available() first, `va`'s check would
    consult repo_root()/annotations/faces.json (which exists in this real
    repo checkout) instead of `tmp_path` (this run's --out, which has no
    faces.json at all), report itself available, and then silently degrade
    every one of the 99 records to full-image inference during annotate() --
    exactly the failure Fix 4 exists to prevent. With the hook called first,
    `available()` sees the real --out directory and refuses up front.
    """
    from open_gesture_annotate.runner import run_backend

    backend = HSEmotionBackend()
    summary = run_backend(backend, gestures, image_root, tmp_path)

    assert summary.unavailable is not None
    assert "faces.json" in summary.unavailable
    assert "face" in summary.unavailable
    assert (summary.ok, summary.errors, summary.skipped) == (0, 0, 0)


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
