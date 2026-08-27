import pytest

from open_gesture_annotate.backends.wholebody_rtmw import KEYPOINT_GROUPS, RTMWBackend


def test_backend_identity():
    b = RTMWBackend()
    assert b.name == "rtmw-wholebody"
    assert b.sidecar == "wholebody.json"


def test_coco_wholebody_groups_cover_133_keypoints():
    assert KEYPOINT_GROUPS["body"] == (0, 17)
    assert KEYPOINT_GROUPS["feet"] == (17, 23)
    assert KEYPOINT_GROUPS["face"] == (23, 91)
    assert KEYPOINT_GROUPS["hands"] == (91, 133)
    assert max(end for _, end in KEYPOINT_GROUPS.values()) == 133


def test_available_never_raises_even_without_mmpose():
    available, reason = RTMWBackend().available()
    assert isinstance(available, bool)
    assert isinstance(reason, str) and reason


def test_provenance_names_mmpose_as_apache():
    assert RTMWBackend().provenance()["library"]["license"] == "Apache-2.0"


@pytest.mark.slow
def test_annotates_a_real_image():
    b = RTMWBackend()
    if not b.available()[0]:
        pytest.skip("mmpose not installed")

    from open_gesture_annotate.io import load_image, load_manifest, repo_root
    from open_gesture_annotate.schema import validate_record

    gestures = {g.id: g for g in load_manifest(repo_root())}
    g = gestures["affirm-01"]
    rec = b.annotate(load_image(repo_root(), g), g)
    validate_record(rec)
    assert rec["person_count"] >= 1
    assert len(rec["people"][0]["keypoints"]) == 133
