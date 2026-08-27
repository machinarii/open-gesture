import json

import numpy as np
import pytest

from open_gesture_annotate.io import (
    completed_ids,
    load_image,
    load_manifest,
    read_sidecar,
    repo_root,
    sidecar_path,
    write_sidecar,
)
from open_gesture_annotate.schema import error, new_sidecar, ok


class DummyBackend:
    name = "dummy"
    version = "1.2.3"
    sidecar = "dummy.json"


def test_sidecar_path_joins_out_dir_and_backend_filename(tmp_path):
    assert sidecar_path(tmp_path, DummyBackend()) == tmp_path / "dummy.json"


def test_read_sidecar_returns_a_fresh_one_when_absent(tmp_path):
    data = read_sidecar(tmp_path / "dummy.json", DummyBackend())
    assert data["records"] == {}
    assert data["_backend"]["name"] == "dummy"


def test_write_then_read_roundtrips(tmp_path):
    path = tmp_path / "dummy.json"
    data = new_sidecar(DummyBackend())
    data["records"]["affirm-01"] = ok(faces=[{"bbox": [1, 2, 3, 4]}])
    write_sidecar(path, data)
    assert read_sidecar(path, DummyBackend())["records"]["affirm-01"]["faces"][0]["bbox"] == [1, 2, 3, 4]


def test_write_sidecar_is_atomic_leaving_no_temp_files(tmp_path):
    path = tmp_path / "dummy.json"
    write_sidecar(path, new_sidecar(DummyBackend()))
    assert [p.name for p in tmp_path.iterdir()] == ["dummy.json"]


def test_write_sidecar_rejects_an_invalid_payload(tmp_path):
    from open_gesture_annotate.schema import SchemaError

    data = new_sidecar(DummyBackend())
    data["records"]["affirm-01"] = {"no_status": True}
    with pytest.raises(SchemaError):
        write_sidecar(tmp_path / "dummy.json", data)


def test_write_sidecar_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "nested" / "deeper" / "dummy.json"
    write_sidecar(path, new_sidecar(DummyBackend()))
    assert path.is_file()


def test_completed_ids_counts_only_ok_records():
    data = new_sidecar(DummyBackend())
    data["records"]["a"] = ok(x=1)
    data["records"]["b"] = error("boom")
    assert completed_ids(data) == {"a"}


def test_written_json_is_human_readable(tmp_path):
    path = tmp_path / "dummy.json"
    write_sidecar(path, new_sidecar(DummyBackend()))
    assert path.read_text(encoding="utf-8").startswith("{\n")


def test_load_image_returns_a_bgr_array():
    gesture = load_manifest(repo_root())[0]
    img = load_image(repo_root(), gesture)
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3 and img.shape[2] == 3


def test_load_image_raises_a_clear_error_for_a_missing_file(tmp_path):
    from open_gesture_annotate.io import Gesture

    gesture = Gesture(id="x", index=1, name="x", category="c", file="nope.png", raw={})
    with pytest.raises(FileNotFoundError, match="nope.png"):
        load_image(tmp_path, gesture)
