import pytest

pytest.importorskip("feat")

from open_gesture_annotate.backends.affect_pyfeat import FEAT_AUS, PyFeatBackend
from open_gesture_annotate.io import load_image, load_manifest, repo_root
from open_gesture_annotate.schema import validate_record


def test_backend_identity():
    b = PyFeatBackend()
    assert b.name == "pyfeat-au"
    assert b.sidecar == "action_units.json"


def test_declares_the_twenty_feat_action_units():
    assert len(FEAT_AUS) == 20
    assert FEAT_AUS[0] == "AU01"
    assert FEAT_AUS[-1] == "AU43"


def test_au_list_matches_upstream():
    """Guard: if py-feat changes its AU set, fail loudly rather than write partial data."""
    from feat.pretrained import AU_LANDMARK_MAP

    assert list(AU_LANDMARK_MAP["Feat"]) == FEAT_AUS


def test_available_reports_true_when_installed():
    assert PyFeatBackend().available()[0] is True


def test_provenance_names_py_feat_as_mit():
    assert PyFeatBackend().provenance()["library"]["license"] == "MIT"


@pytest.mark.slow
def test_annotates_a_real_image():
    b = PyFeatBackend()
    gestures = {g.id: g for g in load_manifest(repo_root())}
    g = gestures["affirm-01"]
    rec = b.annotate(load_image(repo_root(), g), g)
    validate_record(rec)
    assert rec["face_count"] == 1
    assert set(rec["action_units"]) == set(FEAT_AUS)
    assert all(isinstance(v, float) for v in rec["action_units"].values())
    assert set(rec["emotions"]) == {"anger", "disgust", "fear", "happiness",
                                    "sadness", "surprise", "neutral"}


@pytest.mark.slow
def test_no_face_yields_zero_count_and_empty_aus():
    import numpy as np

    from open_gesture_annotate.io import Gesture

    rec = PyFeatBackend().annotate(np.zeros((256, 256, 3), np.uint8),
                                   Gesture("x", 1, "x", "c", "x.png", {}))
    validate_record(rec)
    assert rec["face_count"] == 0
    assert rec["action_units"] == {}
