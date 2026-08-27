import pytest

pytest.importorskip("uniface")

from open_gesture_annotate.backends.face_uniface import UniFaceBackend
from open_gesture_annotate.io import load_image, load_manifest, repo_root
from open_gesture_annotate.schema import validate_record


def test_backend_identity():
    b = UniFaceBackend()
    assert b.name == "uniface-face"
    assert b.sidecar == "faces.json"


def test_available_reports_true_when_installed():
    assert UniFaceBackend().available()[0] is True


def test_provenance_lists_models_with_licences():
    prov = UniFaceBackend().provenance()
    assert prov["models"]
    assert all("license" in m for m in prov["models"])


@pytest.mark.slow
def test_annotates_a_real_image():
    b = UniFaceBackend()
    gestures = {g.id: g for g in load_manifest(repo_root())}
    g = gestures["affirm-01"]
    rec = b.annotate(load_image(repo_root(), g), g)
    validate_record(rec)
    assert rec["face_count"] == 1
    face = rec["faces"][0]
    assert len(face["bbox"]) == 4
    assert 0.0 <= face["confidence"] <= 1.0


@pytest.mark.slow
def test_records_zero_faces_without_raising():
    """A gesture image with no visible face must produce face_count 0, not an error."""
    import numpy as np

    from open_gesture_annotate.io import Gesture

    blank = np.zeros((256, 256, 3), np.uint8)
    rec = UniFaceBackend().annotate(blank, Gesture("x", 1, "x", "c", "x.png", {}))
    validate_record(rec)
    assert rec["face_count"] == 0
