import numpy as np
import pytest

pytest.importorskip("mediapipe")

from open_gesture_annotate.backends.pose_mediapipe import MODEL_URLS, MediaPipePoseBackend
from open_gesture_annotate.io import Gesture, load_image, load_manifest, repo_root
from open_gesture_annotate.schema import validate_record


def test_backend_identity():
    b = MediaPipePoseBackend()
    assert b.name == "mediapipe-pose"
    assert b.sidecar == "pose.json"


def test_declares_both_model_bundles():
    assert set(MODEL_URLS) == {"hand_landmarker", "pose_landmarker"}


def test_provenance_names_mediapipe_as_apache():
    assert MediaPipePoseBackend().provenance()["library"]["license"] == "Apache-2.0"


def test_unavailable_when_model_files_are_missing(tmp_path):
    b = MediaPipePoseBackend(cache_dir=tmp_path)
    available, reason = b.available()
    assert available is False
    assert "hand_landmarker" in reason


@pytest.mark.slow
def test_detects_a_hand_in_the_thumbs_up_image():
    b = MediaPipePoseBackend()
    gestures = {g.id: g for g in load_manifest(repo_root())}
    g = gestures["affirm-01"]
    rec = b.annotate(load_image(repo_root(), g), g)
    validate_record(rec)
    assert rec["hand_count"] >= 1
    assert len(rec["hands"][0]["landmarks"]) == 21
    assert rec["hands"][0]["handedness"] in ("Left", "Right")


@pytest.mark.slow
def test_detects_a_body_in_the_thumbs_up_image():
    b = MediaPipePoseBackend()
    gestures = {g.id: g for g in load_manifest(repo_root())}
    g = gestures["affirm-01"]
    rec = b.annotate(load_image(repo_root(), g), g)
    assert rec["body_detected"] is True
    assert len(rec["body"]["landmarks"]) == 33


@pytest.mark.slow
def test_blank_image_yields_no_detections_without_raising():
    rec = MediaPipePoseBackend().annotate(np.zeros((256, 256, 3), np.uint8),
                                          Gesture("x", 1, "x", "c", "x.png", {}))
    validate_record(rec)
    assert rec["hand_count"] == 0
    assert rec["body_detected"] is False
