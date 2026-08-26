import pytest

from open_gesture_annotate import registry
from open_gesture_annotate.io import read_sidecar, sidecar_path
from open_gesture_annotate.runner import run_backend
from tests.conftest import ExplodingBackend, FakeBackend


def test_run_annotates_every_gesture(gestures, image_root, tmp_path):
    backend = FakeBackend()
    summary = run_backend(backend, gestures, image_root, tmp_path)
    assert summary.ok == 5
    assert summary.errors == 0


def test_run_writes_a_record_per_gesture(gestures, image_root, tmp_path):
    run_backend(FakeBackend(), gestures, image_root, tmp_path)
    data = read_sidecar(sidecar_path(tmp_path, FakeBackend()), FakeBackend())
    assert set(data["records"]) == {g.id for g in gestures}


def test_a_failing_image_is_recorded_not_fatal(gestures, image_root, tmp_path):
    backend = FakeBackend(fail_on={"g-03"})
    summary = run_backend(backend, gestures, image_root, tmp_path)
    assert (summary.ok, summary.errors) == (4, 1)
    data = read_sidecar(sidecar_path(tmp_path, backend), backend)
    assert data["records"]["g-03"]["status"] == "error"
    assert "synthetic failure" in data["records"]["g-03"]["error"]


def test_the_run_continues_past_a_failure(gestures, image_root, tmp_path):
    backend = FakeBackend(fail_on={"g-01"})
    run_backend(backend, gestures, image_root, tmp_path)
    assert backend.calls == [g.id for g in gestures]


def test_an_unserialisable_result_is_recorded_as_an_error(gestures, image_root, tmp_path):
    backend = ExplodingBackend()
    summary = run_backend(backend, gestures, image_root, tmp_path)
    assert summary.errors == 5


def test_a_missing_image_is_recorded_as_an_error(gestures, tmp_path):
    summary = run_backend(FakeBackend(), gestures, tmp_path, tmp_path)
    assert summary.errors == 5


def test_an_unavailable_backend_is_skipped_not_fatal(gestures, image_root, tmp_path):
    backend = FakeBackend(available=False, reason="mediapipe not installed")
    summary = run_backend(backend, gestures, image_root, tmp_path)
    assert summary.unavailable == "mediapipe not installed"
    assert (summary.ok, summary.errors, summary.skipped) == (0, 0, 0)
    assert backend.calls == []


def test_rerunning_skips_completed_records(gestures, image_root, tmp_path):
    run_backend(FakeBackend(), gestures, image_root, tmp_path)
    second = FakeBackend()
    summary = run_backend(second, gestures, image_root, tmp_path)
    assert summary.skipped == 5
    assert second.calls == []


def test_rerunning_retries_previous_errors(gestures, image_root, tmp_path):
    run_backend(FakeBackend(fail_on={"g-02"}), gestures, image_root, tmp_path)
    second = FakeBackend()
    summary = run_backend(second, gestures, image_root, tmp_path)
    assert second.calls == ["g-02"]
    assert summary.ok == 1 and summary.skipped == 4


def test_force_reannotates_everything(gestures, image_root, tmp_path):
    run_backend(FakeBackend(), gestures, image_root, tmp_path)
    second = FakeBackend()
    summary = run_backend(second, gestures, image_root, tmp_path, force=True)
    assert second.calls == [g.id for g in gestures]
    assert summary.skipped == 0


def test_limit_caps_the_number_annotated(gestures, image_root, tmp_path):
    backend = FakeBackend()
    summary = run_backend(backend, gestures, image_root, tmp_path, limit=2)
    assert summary.ok == 2
    assert backend.calls == ["g-01", "g-02"]


# --- registry ---


def test_registry_get_returns_the_registered_backend():
    registry.register("t-fake", FakeBackend)
    assert registry.get("t-fake").name == "fake"


def test_registry_lists_registered_keys():
    registry.register("t-fake", FakeBackend)
    assert "t-fake" in registry.all_keys()


def test_registry_raises_a_helpful_error_for_an_unknown_key():
    with pytest.raises(KeyError, match="nope"):
        registry.get("nope")


def test_availability_reports_reason_without_raising():
    registry.register("t-down", lambda: FakeBackend(available=False, reason="no weights"))
    entry = {k: (a, r) for k, a, r in registry.availability()}["t-down"]
    assert entry == (False, "no weights")


def test_importing_the_registry_pulls_in_no_model_library():
    import subprocess
    import sys

    heavy = ["uniface", "feat", "hsemotion", "mediapipe", "torch", "mmpose"]
    code = (
        "import sys; import open_gesture_annotate.registry as r; r.all_keys();"
        f"print([m for m in {heavy!r} if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
