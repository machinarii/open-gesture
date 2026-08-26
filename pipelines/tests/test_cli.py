import pytest

from open_gesture_annotate import registry
from open_gesture_annotate.cli import main
from tests.conftest import FakeBackend


@pytest.fixture(autouse=True)
def registered_fake():
    registry.register("cli-fake", FakeBackend)
    yield


def test_list_prints_every_backend_key(capsys):
    assert main(["list"]) == 0
    assert "cli-fake" in capsys.readouterr().out


def test_list_prints_availability_reason(capsys):
    registry.register("cli-down", lambda: FakeBackend(available=False, reason="not installed"))
    main(["list"])
    assert "not installed" in capsys.readouterr().out


def test_run_reports_a_summary(capsys, image_root, tmp_path, gestures, monkeypatch):
    monkeypatch.setattr("open_gesture_annotate.cli.load_manifest", lambda root: gestures)
    monkeypatch.setattr("open_gesture_annotate.cli.repo_root", lambda: image_root)
    assert main(["run", "--backends", "cli-fake", "--out", str(tmp_path)]) == 0
    assert "5 ok" in capsys.readouterr().out


def test_run_rejects_an_unknown_backend(capsys, tmp_path):
    assert main(["run", "--backends", "nope", "--out", str(tmp_path)]) == 2
    assert "unknown backend" in capsys.readouterr().err


def test_run_returns_zero_when_a_backend_is_merely_unavailable(capsys, image_root, tmp_path, gestures, monkeypatch):
    registry.register("cli-down", lambda: FakeBackend(available=False, reason="not installed"))
    monkeypatch.setattr("open_gesture_annotate.cli.load_manifest", lambda root: gestures)
    monkeypatch.setattr("open_gesture_annotate.cli.repo_root", lambda: image_root)
    assert main(["run", "--backends", "cli-down", "--out", str(tmp_path)]) == 0
    assert "unavailable" in capsys.readouterr().out


def test_limit_is_passed_through(capsys, image_root, tmp_path, gestures, monkeypatch):
    monkeypatch.setattr("open_gesture_annotate.cli.load_manifest", lambda root: gestures)
    monkeypatch.setattr("open_gesture_annotate.cli.repo_root", lambda: image_root)
    main(["run", "--backends", "cli-fake", "--out", str(tmp_path), "--limit", "2"])
    assert "2 ok" in capsys.readouterr().out
