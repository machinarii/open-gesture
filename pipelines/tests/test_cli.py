import subprocess
import sys

import pytest

from open_gesture_annotate import registry
from open_gesture_annotate.cli import console_main, main
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


def test_main_returns_an_int_rather_than_exiting():
    # main() must stay a pure, testable function: no os._exit, no SystemExit.
    # console_main() is the only place that terminates the process, and it must
    # never be called in-process from a test (see test_console_main_exits_zero,
    # which runs it in a subprocess instead).
    result = main(["list"])
    assert isinstance(result, int)
    assert not isinstance(result, bool)


def test_console_main_exists_and_is_callable():
    assert callable(console_main)


def test_console_main_exits_zero_in_a_subprocess():
    # console_main() calls os._exit(), which would kill the pytest process if
    # invoked in-process -- always exercise it out-of-process.
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.argv = ['og-annotate', 'list']; "
         "from open_gesture_annotate.cli import console_main; console_main()"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0
    assert "list" not in proc.stderr  # sanity: no traceback dumped to stderr


def test_console_main_exits_two_for_an_unknown_backend_in_a_subprocess():
    # Regression: changing os._exit(code) to os._exit(0) in console_main would
    # silently swallow the 2 that `main()` returns for an unknown backend, and
    # test_console_main_exits_zero_in_a_subprocess alone cannot catch that --
    # it only ever asserts the zero case. Always exercised out-of-process,
    # same reason as the zero-case test above.
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.argv = ['og-annotate', 'run', '--backends', 'nope']; "
         "from open_gesture_annotate.cli import console_main; console_main()"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 2
    assert "unknown backend" in proc.stderr
