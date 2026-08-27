import pytest

from open_gesture_annotate.schema import (
    SchemaError,
    error,
    new_sidecar,
    ok,
    validate_record,
    validate_sidecar,
)


class DummyBackend:
    name = "dummy"
    version = "1.2.3"
    sidecar = "dummy.json"


def test_ok_record_has_status_ok():
    assert ok(faces=[])["status"] == "ok"


def test_ok_record_carries_payload():
    assert ok(faces=[{"bbox": [1, 2, 3, 4]}])["faces"] == [{"bbox": [1, 2, 3, 4]}]


def test_error_record_has_status_and_message():
    rec = error("no face detected")
    assert rec["status"] == "error"
    assert rec["error"] == "no face detected"


def test_validate_record_rejects_missing_status():
    with pytest.raises(SchemaError, match="status"):
        validate_record({"faces": []})


def test_validate_record_rejects_unknown_status():
    with pytest.raises(SchemaError, match="status"):
        validate_record({"status": "maybe"})


def test_validate_record_rejects_error_without_message():
    with pytest.raises(SchemaError, match="error"):
        validate_record({"status": "error"})


def test_validate_record_rejects_non_json_serialisable():
    with pytest.raises(SchemaError, match="serialis"):
        validate_record({"status": "ok", "blob": object()})


def test_new_sidecar_records_backend_identity():
    data = new_sidecar(DummyBackend())
    assert data["_backend"]["name"] == "dummy"
    assert data["_backend"]["version"] == "1.2.3"
    assert data["_backend"]["run_at"].endswith("+00:00")
    assert data["records"] == {}


def test_validate_sidecar_accepts_a_fresh_one():
    validate_sidecar(new_sidecar(DummyBackend()))


def test_validate_sidecar_rejects_missing_records():
    with pytest.raises(SchemaError, match="records"):
        validate_sidecar({"_backend": {"name": "d", "version": "1", "run_at": "x"}})


def test_validate_sidecar_validates_each_record():
    data = new_sidecar(DummyBackend())
    data["records"]["affirm-01"] = {"faces": []}
    with pytest.raises(SchemaError, match="affirm-01"):
        validate_sidecar(data)
