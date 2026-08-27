from pathlib import Path

import pytest

from open_gesture_annotate.io import Gesture, load_manifest, repo_root


def test_repo_root_contains_manifest():
    assert (repo_root() / "manifest.json").is_file()


@pytest.fixture(scope="module")
def gestures():
    return load_manifest(repo_root())


def test_loads_all_99_gestures(gestures):
    assert len(gestures) == 99


def test_gesture_ids_are_unique(gestures):
    ids = [g.id for g in gestures]
    assert len(set(ids)) == len(ids)


def test_first_gesture_fields(gestures):
    g = gestures[0]
    assert g.id == "affirm-01"
    assert g.index == 1
    assert g.name == "Thumbs Up"
    assert g.category == "Affirmative & Positive"
    assert g.file == "gesture_images/affirmative-and-positive/affirm-01-thumbs-up.png"


def test_raw_preserves_curated_fields(gestures):
    g = gestures[0]
    assert g.raw["emotional_state"] == "positive"
    assert g.raw["arousal"] == "low"
    assert g.raw["number_of_people"] == "single"
    assert g.raw["body_parts"] == ["hand", "thumb"]


def test_every_image_file_exists(gestures):
    missing = [g.file for g in gestures if not (repo_root() / g.file).is_file()]
    assert missing == []
