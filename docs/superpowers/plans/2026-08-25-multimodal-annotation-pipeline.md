# Multimodal Annotation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pluggable inference pipeline that annotates open-gesture's 113 images with face, affect, hand/body, and semantic signals, then cross-checks those predictions against the hand-curated `manifest.json`.

**Architecture:** A tiny `Backend` protocol (`available` / `provenance` / `annotate`) with one implementation per model family. A registry holds *lazy factories* so importing the package never imports a model library. A runner loops images × backends, isolating every failure into a per-record `status` field, and writes one JSON sidecar per backend under `annotations/`. `report.py` joins sidecars against curated metadata and emits `quality_report.md`.

**Tech Stack:** Python 3.10–3.14, uniface 4.x, py-feat, hsemotion, mediapipe, open_clip_torch, optional mmpose/RTMW, pytest, numpy, opencv-python.

**Spec:** `docs/superpowers/specs/2026-08-25-multimodal-annotation-pipeline-design.md`

## Global Constraints

- Python `>=3.10,<3.15` (uniface's constraint, tightest of the set). CPU-only by default.
- **Never modify** `manifest.json`, `manifest.md`, or anything under `gesture_images/`. The pipeline is read-only with respect to curated data.
- **One sidecar per backend, single writer.** No two backends write the same file.
- **Every record has a `status` field**, either `"ok"` or `"error"`. A failed image is a recorded outcome, never a missing key.
- **Lazy imports.** Importing `open_gesture_annotate.registry` must not import uniface, py-feat, hsemotion, mediapipe, torch, or mmpose. Model libraries are imported inside `available()` and `annotate()` only.
- **Licenses:** only MIT / Apache-2.0 / BSD dependencies. Never add OpenPose, Ultralytics, Sapiens, LibreFace, or OpenFace.
- All paths in sidecars are repo-relative POSIX strings, matching `manifest.json`'s `file` field.
- Commit after every task.

### Deviation from spec (deliberate)

The spec assigns `clip-embed` the sidecar `embeddings.npz`. Making one backend write a
binary format would force a special case through the whole I/O layer. Instead
`clip-embed` writes `embeddings.json` like every other backend (canonical, resumable,
diff-able), and Task 11 additionally emits `embeddings.npz` as a *derived* artifact for
numeric consumers. The protocol stays uniform; consumers still get their array file.

---

### Task 1: Environment and package scaffold

The machine's pyenv 3.13.2 has a broken `hashlib` — `blake2b`/`blake2s` raise
`unsupported hash type`, which breaks `pip`. Provisioning a sound interpreter is a
prerequisite, not an incidental fix.

**Files:**
- Create: `pipelines/pyproject.toml`
- Create: `pipelines/open_gesture_annotate/__init__.py`
- Create: `pipelines/tests/__init__.py`
- Create: `pipelines/tests/test_smoke.py`
- Create: `.gitignore`

- [ ] **Step 1: Find a working interpreter**

```bash
cd /Users/jin/open-gesture
for py in python3.12 python3.11 python3.10 /opt/homebrew/bin/python3.12 /usr/bin/python3; do
  command -v $py >/dev/null 2>&1 || continue
  $py -c "import hashlib; hashlib.blake2b(b'x'); import sys; print('OK', sys.version.split()[0], sys.executable)" 2>/dev/null
done
```

Expected: at least one line beginning `OK` with a version in [3.10, 3.15).
If none pass, install one: `brew install python@3.12`, then re-run.

- [ ] **Step 2: Create the venv**

```bash
cd /Users/jin/open-gesture
<WORKING_PYTHON> -m venv pipelines/.venv
pipelines/.venv/bin/python -c "import hashlib; hashlib.blake2b(b'x'); print('hashlib ok')"
pipelines/.venv/bin/python -m pip install -q --upgrade pip
```

Expected: `hashlib ok`, then pip upgrades without a `blake2b` traceback.

- [ ] **Step 3: Write `pipelines/pyproject.toml`**

```toml
[project]
name = "open-gesture-annotate"
version = "0.1.0"
description = "Multimodal annotation pipeline for the Open Gesture dataset"
requires-python = ">=3.10,<3.15"
dependencies = ["numpy>=1.24", "opencv-python>=4.8"]

[project.optional-dependencies]
face      = ["uniface[cpu]>=4.0.0"]
aus       = ["py-feat>=0.6"]
va        = ["hsemotion>=0.3"]
pose      = ["mediapipe>=0.10"]
embed     = ["open_clip_torch>=2.24", "torch>=2.0"]
wholebody = ["mmpose>=1.3", "mmcv>=2.1", "mmengine>=0.10", "mmdet>=3.2"]
dev       = ["pytest>=8.0", "pytest-cov"]

[project.scripts]
og-annotate = "open_gesture_annotate.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["open_gesture_annotate*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: requires real model weights; deselect with -m 'not slow'"]
addopts = "-m 'not slow'"
```

- [ ] **Step 4: Write `.gitignore` at repo root**

```gitignore
.DS_Store
pipelines/.venv/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/

# Machine-generated annotations: bulky, regenerable.
# _meta.json and quality_report.md are the reviewable artifacts and ARE committed.
annotations/*.json
annotations/*.npz
!annotations/_meta.json
```

- [ ] **Step 5: Write the smoke test**

`pipelines/tests/test_smoke.py`:

```python
import sys


def test_python_version_in_supported_range():
    assert (3, 10) <= sys.version_info[:2] < (3, 15)


def test_hashlib_is_not_broken():
    import hashlib

    hashlib.blake2b(b"open-gesture")


def test_package_imports():
    import open_gesture_annotate

    assert open_gesture_annotate.__version__ == "0.1.0"
```

- [ ] **Step 6: Run it and watch it fail**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_smoke.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'open_gesture_annotate'` (pytest itself is not installed yet either; install it in Step 7 and re-run).

- [ ] **Step 7: Create the package and install**

`pipelines/open_gesture_annotate/__init__.py`:

```python
"""Multimodal annotation pipeline for the Open Gesture dataset."""

__version__ = "0.1.0"
```

`pipelines/tests/__init__.py`: empty file.

```bash
cd /Users/jin/open-gesture/pipelines
.venv/bin/python -m pip install -q -e ".[dev]"
```

- [ ] **Step 8: Run tests to verify they pass**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest -v
```

Expected: 3 passed.

- [ ] **Step 9: Commit**

```bash
cd /Users/jin/open-gesture
git add .gitignore pipelines/pyproject.toml pipelines/open_gesture_annotate/__init__.py pipelines/tests/
git commit -m "feat: scaffold annotation pipeline package and venv"
```

---

### Task 2: Load the curated manifest

**Files:**
- Create: `pipelines/open_gesture_annotate/io.py`
- Create: `pipelines/tests/test_io_manifest.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Gesture` dataclass (`id: str`, `index: int`, `name: str`, `category: str`, `file: str`, `raw: dict`); `load_manifest(repo_root: Path) -> list[Gesture]`; `repo_root() -> Path`.

- [ ] **Step 1: Write the failing test**

`pipelines/tests/test_io_manifest.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_io_manifest.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'open_gesture_annotate.io'`

- [ ] **Step 3: Write the implementation**

`pipelines/open_gesture_annotate/io.py`:

```python
"""Reading curated data and writing machine-generated sidecars."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Gesture:
    """One curated gesture record from manifest.json."""

    id: str
    index: int
    name: str
    category: str
    file: str  # repo-relative POSIX path
    raw: dict = field(repr=False)


def repo_root() -> Path:
    """The open-gesture checkout root, found by walking up to manifest.json."""
    for candidate in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (candidate / "manifest.json").is_file():
            return candidate
    raise FileNotFoundError("could not locate manifest.json above this package")


def load_manifest(root: Path) -> list[Gesture]:
    """Load manifest.json into Gesture records, ordered by `index`."""
    data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    gestures = [
        Gesture(
            id=rec["id"],
            index=rec["index"],
            name=rec["name"],
            category=rec["category"],
            file=rec["file"],
            raw=rec,
        )
        for rec in data["gestures"]
    ]
    return sorted(gestures, key=lambda g: g.index)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_io_manifest.py -v
```

Expected: 6 passed. If `test_every_image_file_exists` fails, the manifest references a
missing image — record the list in the task notes and report it; do not "fix" it by
editing the manifest.

- [ ] **Step 5: Commit**

```bash
cd /Users/jin/open-gesture
git add pipelines/open_gesture_annotate/io.py pipelines/tests/test_io_manifest.py
git commit -m "feat: load curated gesture manifest"
```

---

### Task 3: Backend protocol and record schema

**Files:**
- Create: `pipelines/open_gesture_annotate/base.py`
- Create: `pipelines/open_gesture_annotate/schema.py`
- Create: `pipelines/tests/test_schema.py`

**Interfaces:**
- Consumes: `Gesture` from Task 2.
- Produces: `Backend` protocol (`name: str`, `version: str`, `sidecar: str`, `available() -> tuple[bool, str]`, `provenance() -> dict`, `annotate(image, gesture) -> dict`); `SchemaError`; `ok(**fields) -> dict`; `error(message: str) -> dict`; `validate_record(rec) -> None`; `validate_sidecar(data) -> None`; `new_sidecar(backend) -> dict`.

- [ ] **Step 1: Write the failing test**

`pipelines/tests/test_schema.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_schema.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'open_gesture_annotate.schema'`

- [ ] **Step 3: Write `base.py`**

```python
"""The Backend protocol every model family implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from open_gesture_annotate.io import Gesture


class BackendUnavailable(Exception):
    """Raised when a backend's library or weights cannot be resolved."""


@runtime_checkable
class Backend(Protocol):
    """One model family. Owns exactly one sidecar file and is its only writer."""

    name: str  # stable key, e.g. "uniface-face"
    version: str  # library version, recorded in _meta.json
    sidecar: str  # output filename, e.g. "faces.json"

    def available(self) -> tuple[bool, str]:
        """(importable and weights resolvable?, human-readable reason)."""

    def provenance(self) -> dict:
        """Model names, weight files, SHA256 and license of each weight."""

    def annotate(self, image: np.ndarray, gesture: Gesture) -> dict:
        """Annotate one BGR image. Raise on failure; the runner isolates it."""
```

- [ ] **Step 4: Write `schema.py`**

```python
"""Annotation record shape and validation."""

from __future__ import annotations

import json
from datetime import datetime, timezone

VALID_STATUSES = ("ok", "error")


class SchemaError(ValueError):
    """A record or sidecar does not match the annotation schema."""


def ok(**fields) -> dict:
    """Build a successful record. Payload keys are backend-specific."""
    return {"status": "ok", **fields}


def error(message: str) -> dict:
    """Build a failed record. Never raises; a failure is a recorded outcome."""
    return {"status": "error", "error": str(message)}


def validate_record(rec: dict) -> None:
    if not isinstance(rec, dict):
        raise SchemaError(f"record must be a dict, got {type(rec).__name__}")
    status = rec.get("status")
    if status not in VALID_STATUSES:
        raise SchemaError(f"record 'status' must be one of {VALID_STATUSES}, got {status!r}")
    if status == "error" and not rec.get("error"):
        raise SchemaError("record with status 'error' must carry a non-empty 'error' message")
    try:
        json.dumps(rec)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"record is not JSON-serialisable: {exc}") from exc


def new_sidecar(backend) -> dict:
    return {
        "_backend": {
            "name": backend.name,
            "version": backend.version,
            "sidecar": backend.sidecar,
            "run_at": datetime.now(timezone.utc).isoformat(),
        },
        "records": {},
    }


def validate_sidecar(data: dict) -> None:
    meta = data.get("_backend")
    if not isinstance(meta, dict):
        raise SchemaError("sidecar must carry a '_backend' object")
    for key in ("name", "version", "run_at"):
        if not meta.get(key):
            raise SchemaError(f"sidecar '_backend' is missing '{key}'")
    records = data.get("records")
    if not isinstance(records, dict):
        raise SchemaError("sidecar must carry a 'records' object")
    for gesture_id, rec in records.items():
        try:
            validate_record(rec)
        except SchemaError as exc:
            raise SchemaError(f"record {gesture_id!r}: {exc}") from exc
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_schema.py -v
```

Expected: 11 passed.

- [ ] **Step 6: Commit**

```bash
cd /Users/jin/open-gesture
git add pipelines/open_gesture_annotate/base.py pipelines/open_gesture_annotate/schema.py pipelines/tests/test_schema.py
git commit -m "feat: add backend protocol and annotation record schema"
```

---

### Task 4: Sidecar persistence and resume

**Files:**
- Modify: `pipelines/open_gesture_annotate/io.py` (append)
- Create: `pipelines/tests/test_io_sidecar.py`

**Interfaces:**
- Consumes: `Gesture`, `repo_root` (Task 2); `new_sidecar`, `validate_sidecar` (Task 3).
- Produces: `sidecar_path(out_dir, backend) -> Path`; `read_sidecar(path, backend) -> dict`; `write_sidecar(path, data) -> None`; `completed_ids(data) -> set[str]`; `load_image(root, gesture) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

`pipelines/tests/test_io_sidecar.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_io_sidecar.py -v
```

Expected: FAIL — `ImportError: cannot import name 'sidecar_path'`

- [ ] **Step 3: Append the implementation to `io.py`**

Add these imports to the top of `io.py`:

```python
import os
import tempfile

import cv2
import numpy as np

from open_gesture_annotate.schema import new_sidecar, validate_sidecar
```

Then append:

```python
def sidecar_path(out_dir: Path, backend) -> Path:
    return Path(out_dir) / backend.sidecar


def read_sidecar(path: Path, backend) -> dict:
    """Load an existing sidecar, or a fresh empty one if it does not exist."""
    path = Path(path)
    if not path.is_file():
        return new_sidecar(backend)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_sidecar(data)
    return data


def write_sidecar(path: Path, data: dict) -> None:
    """Validate then atomically write a sidecar, so a crash cannot truncate it."""
    validate_sidecar(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def completed_ids(data: dict) -> set[str]:
    """Gesture ids already annotated successfully — the basis for resume."""
    return {gid for gid, rec in data["records"].items() if rec.get("status") == "ok"}


def load_image(root: Path, gesture: Gesture) -> np.ndarray:
    """Read a gesture image as a BGR uint8 array."""
    path = Path(root) / gesture.file
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {gesture.file}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode image: {gesture.file}")
    return img
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_io_sidecar.py tests/test_io_manifest.py -v
```

Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/jin/open-gesture
git add pipelines/open_gesture_annotate/io.py pipelines/tests/test_io_sidecar.py
git commit -m "feat: add atomic sidecar persistence and resume support"
```

---

### Task 5: Registry and runner with failure isolation

This is the heart of the pipeline. It is tested entirely with fake backends — no
model weights, no network — so it runs in CI in under a second.

**Files:**
- Create: `pipelines/open_gesture_annotate/registry.py`
- Create: `pipelines/open_gesture_annotate/runner.py`
- Create: `pipelines/tests/conftest.py`
- Create: `pipelines/tests/test_runner.py`

**Interfaces:**
- Consumes: everything from Tasks 2–4.
- Produces: `register(key, factory)`; `get(key) -> Backend`; `all_keys() -> list[str]`; `availability() -> list[tuple[str, bool, str]]`; `RunSummary` dataclass (`backend: str`, `ok: int`, `errors: int`, `skipped: int`, `unavailable: str | None`); `run_backend(backend, gestures, root, out_dir, force=False, limit=None) -> RunSummary`.

- [ ] **Step 1: Write the fakes**

`pipelines/tests/conftest.py`:

```python
import numpy as np
import pytest

from open_gesture_annotate.io import Gesture
from open_gesture_annotate.schema import ok


class FakeBackend:
    """A backend that succeeds, with no model dependency."""

    name = "fake"
    version = "0.0.1"
    sidecar = "fake.json"

    def __init__(self, available=True, reason="ready", fail_on=()):
        self._available = available
        self._reason = reason
        self._fail_on = set(fail_on)
        self.calls = []

    def available(self):
        return self._available, self._reason

    def provenance(self):
        return {"models": [{"name": "fake-model", "license": "MIT"}]}

    def annotate(self, image, gesture):
        self.calls.append(gesture.id)
        if gesture.id in self._fail_on:
            raise RuntimeError(f"synthetic failure for {gesture.id}")
        return ok(width=int(image.shape[1]), height=int(image.shape[0]))


class ExplodingBackend(FakeBackend):
    """A backend whose annotate() returns something unserialisable."""

    name = "exploding"
    sidecar = "exploding.json"

    def annotate(self, image, gesture):
        return {"status": "ok", "blob": object()}


@pytest.fixture
def gestures():
    return [
        Gesture(id=f"g-{i:02d}", index=i, name=f"G{i}", category="C",
                file=f"images/g-{i:02d}.png", raw={})
        for i in range(1, 6)
    ]


@pytest.fixture
def image_root(tmp_path, gestures):
    """A throwaway tree of tiny real PNGs matching `gestures`."""
    import cv2

    (tmp_path / "images").mkdir()
    for i, g in enumerate(gestures, start=1):
        cv2.imwrite(str(tmp_path / g.file), np.full((8, 8 + i, 3), 128, np.uint8))
    return tmp_path
```

- [ ] **Step 2: Write the failing test**

`pipelines/tests/test_runner.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_runner.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'open_gesture_annotate.registry'`

- [ ] **Step 4: Write `registry.py`**

Factories are lazy — a value is a zero-arg callable, and the model library is imported
only inside the backend module, which is itself imported only when the factory runs.

```python
"""Lazy backend registry. Importing this module imports no model library."""

from __future__ import annotations

from typing import Callable

_FACTORIES: dict[str, Callable[[], object]] = {}


def register(key: str, factory: Callable[[], object]) -> None:
    _FACTORIES[key] = factory


def all_keys() -> list[str]:
    return sorted(_FACTORIES)


def get(key: str):
    if key not in _FACTORIES:
        raise KeyError(f"unknown backend {key!r}; known backends: {', '.join(all_keys())}")
    return _FACTORIES[key]()


def availability() -> list[tuple[str, bool, str]]:
    """(key, available, reason) for every backend. Never raises."""
    rows = []
    for key in all_keys():
        try:
            available, reason = get(key).available()
        except Exception as exc:  # an import error is an unavailability, not a crash
            available, reason = False, f"{type(exc).__name__}: {exc}"
        rows.append((key, available, reason))
    return rows


def _register_builtin_backends() -> None:
    """Register lazily: each lambda imports its module only when called."""

    def _lazy(module: str, cls: str):
        def factory():
            import importlib

            return getattr(importlib.import_module(module), cls)()

        return factory

    pkg = "open_gesture_annotate.backends"
    register("face", _lazy(f"{pkg}.face_uniface", "UniFaceBackend"))
    register("aus", _lazy(f"{pkg}.affect_pyfeat", "PyFeatBackend"))
    register("va", _lazy(f"{pkg}.affect_hsemotion", "HSEmotionBackend"))
    register("pose", _lazy(f"{pkg}.pose_mediapipe", "MediaPipePoseBackend"))
    register("embed", _lazy(f"{pkg}.embed_clip", "ClipEmbedBackend"))
    register("wholebody", _lazy(f"{pkg}.wholebody_rtmw", "RTMWBackend"))


_register_builtin_backends()
```

- [ ] **Step 5: Write `runner.py`**

```python
"""Run one backend across the gesture set, isolating every failure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from open_gesture_annotate.io import (
    Gesture,
    completed_ids,
    load_image,
    read_sidecar,
    sidecar_path,
    write_sidecar,
)
from open_gesture_annotate.schema import error, validate_record


@dataclass
class RunSummary:
    backend: str
    ok: int = 0
    errors: int = 0
    skipped: int = 0
    unavailable: str | None = None

    @property
    def attempted(self) -> int:
        return self.ok + self.errors


def run_backend(
    backend,
    gestures: list[Gesture],
    root: Path,
    out_dir: Path,
    force: bool = False,
    limit: int | None = None,
) -> RunSummary:
    """Annotate `gestures` with `backend`, writing after every image.

    Writing after each image is deliberate: a crash mid-run loses at most one
    result, and the next invocation resumes from the sidecar.
    """
    summary = RunSummary(backend=backend.name)

    is_available, reason = backend.available()
    if not is_available:
        summary.unavailable = reason
        return summary

    path = sidecar_path(out_dir, backend)
    data = read_sidecar(path, backend)
    done = set() if force else completed_ids(data)

    attempted = 0
    for gesture in gestures:
        if gesture.id in done:
            summary.skipped += 1
            continue
        if limit is not None and attempted >= limit:
            break
        attempted += 1

        try:
            image = load_image(root, gesture)
            record = backend.annotate(image, gesture)
            validate_record(record)
        except Exception as exc:
            record = error(f"{type(exc).__name__}: {exc}")
            summary.errors += 1
        else:
            summary.ok += 1

        data["records"][gesture.id] = record
        write_sidecar(path, data)

    return summary
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_runner.py -v
```

Expected: 16 passed. Note `test_importing_the_registry_pulls_in_no_model_library` is the
guard on the lazy-import constraint — if it fails, a backend module is being imported at
registry import time.

- [ ] **Step 7: Commit**

```bash
cd /Users/jin/open-gesture
git add pipelines/open_gesture_annotate/registry.py pipelines/open_gesture_annotate/runner.py pipelines/tests/conftest.py pipelines/tests/test_runner.py
git commit -m "feat: add lazy backend registry and isolating runner"
```

---

### Task 6: Command-line interface

**Files:**
- Create: `pipelines/open_gesture_annotate/cli.py`
- Create: `pipelines/tests/test_cli.py`

**Interfaces:**
- Consumes: `registry.availability`, `registry.get`, `registry.all_keys`, `run_backend`, `load_manifest`, `repo_root`.
- Produces: `main(argv: list[str] | None = None) -> int`; `build_parser() -> argparse.ArgumentParser`.

- [ ] **Step 1: Write the failing test**

`pipelines/tests/test_cli.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_cli.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'open_gesture_annotate.cli'`

- [ ] **Step 3: Write `cli.py`**

```python
"""og-annotate command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from open_gesture_annotate import registry
from open_gesture_annotate.io import load_manifest, repo_root
from open_gesture_annotate.runner import run_backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="og-annotate", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="show backends and their availability")

    run = sub.add_parser("run", help="annotate the gesture set")
    run.add_argument("--backends", default="all",
                     help="comma-separated backend keys, or 'all' (default)")
    run.add_argument("--out", default=None, help="output directory (default: <repo>/annotations)")
    run.add_argument("--force", action="store_true", help="re-annotate already-completed records")
    run.add_argument("--limit", type=int, default=None, help="annotate at most N images per backend")

    report = sub.add_parser("report", help="regenerate annotations/quality_report.md")
    report.add_argument("--out", default=None, help="annotations directory")

    return parser


def _resolve_out(out: str | None) -> Path:
    return Path(out) if out else repo_root() / "annotations"


def _cmd_list() -> int:
    for key, available, reason in registry.availability():
        mark = "available" if available else "UNAVAILABLE"
        print(f"  {key:<12} {mark:<12} {reason}")
    return 0


def _cmd_run(args) -> int:
    keys = registry.all_keys() if args.backends == "all" else \
        [k.strip() for k in args.backends.split(",") if k.strip()]

    unknown = [k for k in keys if k not in registry.all_keys()]
    if unknown:
        print(f"unknown backend(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"known: {', '.join(registry.all_keys())}", file=sys.stderr)
        return 2

    out_dir = _resolve_out(args.out)
    gestures = load_manifest(repo_root())

    for key in keys:
        summary = run_backend(registry.get(key), gestures, repo_root(), out_dir,
                              force=args.force, limit=args.limit)
        if summary.unavailable:
            print(f"  {key:<12} unavailable: {summary.unavailable}")
        else:
            print(f"  {key:<12} {summary.ok} ok, {summary.errors} errors, "
                  f"{summary.skipped} skipped")
    return 0


def _cmd_report(args) -> int:
    from open_gesture_annotate.report import write_report

    path = write_report(repo_root(), _resolve_out(args.out))
    print(f"wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        return _cmd_list()
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "report":
        return _cmd_report(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_cli.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Verify the console script is wired**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/og-annotate list
```

Expected: six rows (`aus`, `embed`, `face`, `pose`, `va`, `wholebody`), all reporting
UNAVAILABLE with an import-error reason — no backend module exists yet. This confirms the
registry degrades gracefully rather than crashing.

- [ ] **Step 6: Commit**

```bash
cd /Users/jin/open-gesture
git add pipelines/open_gesture_annotate/cli.py pipelines/tests/test_cli.py
git commit -m "feat: add og-annotate CLI with list and run commands"
```

---

### Task 7: uniface backend — face geometry, head pose, gaze, emotion, demographics

uniface's predictor class names are not fully documented publicly. **Step 1 discovers the
real API before any code is written against it.** Do not skip it and do not guess names.

**Files:**
- Create: `pipelines/open_gesture_annotate/backends/__init__.py`
- Create: `pipelines/open_gesture_annotate/backends/face_uniface.py`
- Create: `pipelines/tests/test_backend_uniface.py`

**Interfaces:**
- Consumes: `Backend` protocol, `ok`/`error`, `Gesture`.
- Produces: `UniFaceBackend` with `name="uniface-face"`, `sidecar="faces.json"`. Record payload: `{"status": "ok", "faces": [{"bbox": [x1,y1,x2,y2], "confidence": float, "landmarks": [[x,y],...], "head_pose": {"pitch","yaw","roll"} | None, "gaze": {"pitch","yaw"} | None, "emotion": {"label": str, "scores": {}} | None, "demographics": {"age_group","sex"} | None}], "face_count": int}`.

- [ ] **Step 1: Install uniface and discover its API**

```bash
cd /Users/jin/open-gesture/pipelines
.venv/bin/python -m pip install -e ".[face]"
.venv/bin/python - <<'PY'
import uniface, inspect
print("version:", getattr(uniface, "__version__", "?"))
print("exports:", sorted(n for n in dir(uniface) if n[0].isupper()))
from uniface import FaceAnalyzer
print("FaceAnalyzer:", inspect.signature(FaceAnalyzer.__init__))
print("analyze:", inspect.signature(FaceAnalyzer.analyze))
PY
```

Record the printed predictor class names. Map them to roles: detection, landmarks, head
pose, gaze, emotion (AffectNet), demographics (FairFace). Write the mapping into the
module docstring of `face_uniface.py` so the next reader does not have to rediscover it.

- [ ] **Step 2: Inspect the Face object's real attributes**

```bash
cd /Users/jin/open-gesture/pipelines
.venv/bin/python - <<'PY'
import cv2
from uniface import FaceAnalyzer
a = FaceAnalyzer()
faces = list(a.analyze(cv2.imread("../gesture_images/affirmative-and-positive/affirm-01-thumbs-up.png")))
print("n faces:", len(faces))
f = faces[0]
print("attrs:", [k for k in dir(f) if not k.startswith("_")])
print("bbox:", f.bbox, "conf:", f.confidence)
PY
```

Expected: `n faces: 1`. Record the attribute names — the implementation reads them via
`getattr(face, attr, None)` so a missing attribute yields `None` rather than raising.

- [ ] **Step 3: Write the failing test**

`pipelines/tests/test_backend_uniface.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it fails**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_backend_uniface.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'open_gesture_annotate.backends'`

- [ ] **Step 5: Write the implementation**

`pipelines/open_gesture_annotate/backends/__init__.py`: empty file.

`pipelines/open_gesture_annotate/backends/face_uniface.py` — replace the predictor
construction in `_analyzer()` with the real class names discovered in Step 1:

```python
"""uniface backend: face geometry, head pose, gaze, emotion, demographics.

Predictor class names discovered on 2026-08-25 (see Task 7 Step 1):
    detection    -> <fill in from Step 1>
    landmarks    -> <fill in from Step 1>
    head pose    -> <fill in from Step 1>
    gaze         -> <fill in from Step 1>
    emotion      -> <fill in from Step 1>
    demographics -> FairFace
"""

from __future__ import annotations

import numpy as np

from open_gesture_annotate.io import Gesture
from open_gesture_annotate.schema import ok


def _as_list(value):
    """Convert numpy scalars/arrays to plain JSON-serialisable Python."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_as_list(v) for v in value]
    if isinstance(value, dict):
        return {k: _as_list(v) for k, v in value.items()}
    return value


class UniFaceBackend:
    name = "uniface-face"
    sidecar = "faces.json"

    def __init__(self) -> None:
        self._impl = None

    @property
    def version(self) -> str:
        try:
            import uniface

            return getattr(uniface, "__version__", "unknown")
        except ImportError:
            return "unavailable"

    def available(self) -> tuple[bool, str]:
        try:
            import uniface  # noqa: F401
        except ImportError as exc:
            return False, f"uniface not installed ({exc}); pip install -e '.[face]'"
        try:
            self._analyzer()
        except Exception as exc:
            return False, f"uniface installed but weights failed to load: {exc}"
        return True, f"uniface {self.version}"

    def _analyzer(self):
        """Build the FaceAnalyzer once. Predictor names come from Step 1."""
        if self._impl is None:
            from uniface import FaceAnalyzer, FairFace

            # Add the head-pose, gaze and emotion predictors discovered in Step 1.
            self._impl = FaceAnalyzer(predictors=[FairFace()])
        return self._impl

    def provenance(self) -> dict:
        return {
            "library": {"name": "uniface", "version": self.version, "license": "MIT"},
            "models": [
                {"name": "detector", "license": "check upstream weight licence"},
                {"name": "FairFace", "license": "check upstream weight licence"},
            ],
            "warning": (
                "uniface's README states some pretrained weights are not MIT. "
                "Confirm each weight's licence before commercial redistribution."
            ),
        }

    def annotate(self, image: np.ndarray, gesture: Gesture) -> dict:
        faces = []
        for face in self._analyzer().analyze(image):
            faces.append(
                {
                    "bbox": _as_list(getattr(face, "bbox", None)),
                    "confidence": _as_list(getattr(face, "confidence", None)),
                    "landmarks": _as_list(getattr(face, "landmarks", None)),
                    "head_pose": _as_list(getattr(face, "head_pose", None)),
                    "gaze": _as_list(getattr(face, "gaze", None)),
                    "emotion": _as_list(getattr(face, "emotion", None)),
                    "demographics": {
                        "age_group": _as_list(getattr(face, "age_group", None)),
                        "sex": _as_list(getattr(face, "sex", None)),
                    },
                }
            )
        return ok(faces=faces, face_count=len(faces))
```

- [ ] **Step 6: Run the fast tests**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_backend_uniface.py -v
```

Expected: 3 passed, 2 deselected (the `slow` ones).

- [ ] **Step 7: Run the slow tests against real weights**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_backend_uniface.py -v -m slow
```

Expected: 2 passed. First run downloads weights.

- [ ] **Step 8: Annotate the real dataset**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/og-annotate run --backends face
```

Expected: `face   99 ok, 0 errors, 0 skipped` (or a small number of errors on images with
no visible face — those are recorded, not fatal). Confirm `annotations/faces.json` exists.

- [ ] **Step 9: Commit**

```bash
cd /Users/jin/open-gesture
git add pipelines/open_gesture_annotate/backends/ pipelines/tests/test_backend_uniface.py
git commit -m "feat: add uniface face annotation backend"
```

---

### Task 8: py-feat backend — FACS Action Units

**Files:**
- Create: `pipelines/open_gesture_annotate/backends/affect_pyfeat.py`
- Create: `pipelines/tests/test_backend_pyfeat.py`

**Interfaces:**
- Consumes: `Backend` protocol, `ok`, `Gesture`.
- Produces: `PyFeatBackend` with `name="pyfeat-au"`, `sidecar="action_units.json"`. Record payload: `{"status": "ok", "face_count": int, "action_units": {"AU01": float, ...}, "emotions": {"anger": float, ...}, "head_pose": {"Pitch","Roll","Yaw"}}`.

The 20 `Feat` AUs, verified from `feat/pretrained.py` on 2026-08-25: AU01 AU02 AU04 AU05
AU06 AU07 AU09 AU10 AU11 AU12 AU14 AU15 AU17 AU20 AU23 AU24 AU25 AU26 AU28 AU43.
The 7 emotions, from `feat/utils/__init__.py`: anger disgust fear happiness sadness
surprise neutral.

- [ ] **Step 1: Install and confirm the import path**

```bash
cd /Users/jin/open-gesture/pipelines
.venv/bin/python -m pip install -e ".[aus]"
.venv/bin/python - <<'PY'
from feat.detector import Detectorv1
from feat.pretrained import AU_LANDMARK_MAP
from feat.utils import FEAT_EMOTION_COLUMNS, FEAT_FACEPOSE_COLUMNS_3D
print("AUs:", AU_LANDMARK_MAP["Feat"])
print("emotions:", FEAT_EMOTION_COLUMNS)
print("pose:", FEAT_FACEPOSE_COLUMNS_3D)
PY
```

Expected: the 20 AUs, 7 emotions, and `['Pitch', 'Roll', 'Yaw']` above. If `Detectorv1`
is not importable from `feat.detector`, try `from feat import Detectorv1` and record which
worked.

- [ ] **Step 2: Write the failing test**

`pipelines/tests/test_backend_pyfeat.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_backend_pyfeat.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'open_gesture_annotate.backends.affect_pyfeat'`

- [ ] **Step 4: Write the implementation**

`pipelines/open_gesture_annotate/backends/affect_pyfeat.py`:

```python
"""py-feat backend: FACS Action Units, categorical emotion, 3D head pose.

FACS is Ekman & Friesen's coding system, which the project README cites. Action
Units give the affect metadata the same anatomical grounding that BAP gives
`body_parts`.

py-feat's detector works on file paths, not arrays, so `annotate` writes the
image to a temporary file. `identity_model=None` deliberately: face identity
embeddings are not needed and ArcFace weights carry a separate licence.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from open_gesture_annotate.io import Gesture
from open_gesture_annotate.schema import ok

FEAT_AUS = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
    "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24",
    "AU25", "AU26", "AU28", "AU43",
]
FEAT_EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
FEAT_POSE = ["Pitch", "Roll", "Yaw"]


class PyFeatBackend:
    name = "pyfeat-au"
    sidecar = "action_units.json"

    def __init__(self) -> None:
        self._impl = None

    @property
    def version(self) -> str:
        try:
            from importlib.metadata import version

            return version("py-feat")
        except Exception:
            return "unknown"

    def available(self) -> tuple[bool, str]:
        try:
            from feat.detector import Detectorv1  # noqa: F401
        except ImportError as exc:
            return False, f"py-feat not installed ({exc}); pip install -e '.[aus]'"
        return True, f"py-feat {self.version}"

    def _detector(self):
        if self._impl is None:
            from feat.detector import Detectorv1

            self._impl = Detectorv1(
                face_model="retinaface",
                landmark_model="mobilefacenet",
                au_model="xgb",
                emotion_model="resmasknet",
                identity_model=None,
                device="cpu",
            )
        return self._impl

    def provenance(self) -> dict:
        return {
            "library": {"name": "py-feat", "version": self.version, "license": "MIT"},
            "models": [
                {"name": "retinaface", "role": "detection", "license": "MIT"},
                {"name": "mobilefacenet", "role": "landmarks", "license": "MIT"},
                {"name": "xgb", "role": "action units", "license": "MIT"},
                {"name": "resmasknet", "role": "emotion", "license": "MIT"},
            ],
        }

    def annotate(self, image: np.ndarray, gesture: Gesture) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.png"
            cv2.imwrite(str(path), image)
            fex = self._detector().detect([str(path)], data_type="image", progress_bar=False)

        if fex is None or len(fex) == 0:
            return ok(face_count=0, action_units={}, emotions={}, head_pose={})

        row = fex.iloc[0]

        def _grab(columns):
            return {c: float(row[c]) for c in columns if c in fex.columns and row[c] == row[c]}

        return ok(
            face_count=int(len(fex)),
            action_units=_grab(FEAT_AUS),
            emotions=_grab(FEAT_EMOTIONS),
            head_pose=_grab(FEAT_POSE),
        )
```

Note `row[c] == row[c]` is a NaN check — py-feat emits NaN for undetected columns, and
NaN is not valid JSON, so those keys are dropped rather than written.

- [ ] **Step 5: Run the fast tests**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_backend_pyfeat.py -v
```

Expected: 5 passed, 2 deselected.

- [ ] **Step 6: Run the slow tests**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_backend_pyfeat.py -v -m slow
```

Expected: 2 passed.

- [ ] **Step 7: Annotate the dataset and commit**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/og-annotate run --backends aus
cd /Users/jin/open-gesture
git add pipelines/open_gesture_annotate/backends/affect_pyfeat.py pipelines/tests/test_backend_pyfeat.py
git commit -m "feat: add py-feat action unit backend"
```

---

### Task 9: HSEmotion backend — continuous valence and arousal

**The valence/arousal output is undocumented.** HSEmotion's README claims only discrete
emotions. Source inspection of `hsemotion/facial_emotions.py` (2026-08-25) shows
`predict_emotions` computes its softmax over `scores[:-2]`, meaning the `enet_b0_8_va_mtl`
multi-task model appends **valence and arousal as the final two elements** of `scores`.
This backend depends on that layout, so it asserts the vector length and fails loudly if
upstream ever changes it, rather than silently writing garbage.

**Files:**
- Create: `pipelines/open_gesture_annotate/backends/affect_hsemotion.py`
- Create: `pipelines/tests/test_backend_hsemotion.py`

**Interfaces:**
- Consumes: `Backend` protocol, `ok`, `Gesture`; face bounding boxes from `annotations/faces.json` (Task 7).
- Produces: `HSEmotionBackend` with `name="hsemotion-va"`, `sidecar="valence_arousal.json"`; `EXPECTED_SCORE_LEN = 10`; `split_scores(scores) -> tuple[dict, float, float]`. Record payload: `{"status": "ok", "label": str, "valence": float, "arousal": float, "emotion_scores": {...}, "face_source": "uniface" | "full-image"}`.

- [ ] **Step 1: Install and confirm the score layout**

```bash
cd /Users/jin/open-gesture/pipelines
.venv/bin/python -m pip install -e ".[va]"
.venv/bin/python - <<'PY'
import cv2, numpy as np
from hsemotion.facial_emotions import HSEmotionRecognizer
fer = HSEmotionRecognizer(model_name="enet_b0_8_va_mtl", device="cpu")
img = cv2.cvtColor(cv2.imread("../gesture_images/affirmative-and-positive/affirm-01-thumbs-up.png"), cv2.COLOR_BGR2RGB)
label, scores = fer.predict_emotions(img, logits=False)
print("classes:", fer.idx_to_class)
print("label:", label, "len(scores):", len(scores))
print("emotions:", scores[:-2], "valence/arousal:", scores[-2:])
PY
```

Expected: 8 classes, `len(scores): 10`. If the length is not 10, **stop** — the layout
assumption is broken and the backend must be redesigned, not patched.

- [ ] **Step 2: Write the failing test**

`pipelines/tests/test_backend_hsemotion.py`:

```python
import numpy as np
import pytest

pytest.importorskip("hsemotion")

from open_gesture_annotate.backends.affect_hsemotion import (
    EXPECTED_SCORE_LEN,
    HSEmotionBackend,
    split_scores,
)
from open_gesture_annotate.io import load_image, load_manifest, repo_root
from open_gesture_annotate.schema import validate_record

CLASSES = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]


def test_backend_identity():
    b = HSEmotionBackend()
    assert b.name == "hsemotion-va"
    assert b.sidecar == "valence_arousal.json"


def test_split_scores_separates_emotions_from_valence_arousal():
    scores = np.arange(10, dtype=np.float32)
    emotions, valence, arousal = split_scores(scores, CLASSES)
    assert len(emotions) == 8
    assert emotions["Anger"] == 0.0
    assert (valence, arousal) == (8.0, 9.0)


def test_split_scores_rejects_an_unexpected_vector_length():
    with pytest.raises(ValueError, match="expected 10"):
        split_scores(np.arange(8, dtype=np.float32), CLASSES)


def test_expected_length_is_eight_emotions_plus_valence_arousal():
    assert EXPECTED_SCORE_LEN == 10


def test_upstream_class_count_still_matches():
    """Guard on the undocumented layout this backend depends on."""
    from hsemotion.facial_emotions import HSEmotionRecognizer

    fer = HSEmotionRecognizer(model_name="enet_b0_8_va_mtl", device="cpu")
    assert len(fer.idx_to_class) == 8


def test_available_reports_true_when_installed():
    assert HSEmotionBackend().available()[0] is True


@pytest.mark.slow
def test_annotates_a_real_image():
    b = HSEmotionBackend()
    gestures = {g.id: g for g in load_manifest(repo_root())}
    g = gestures["affirm-01"]
    rec = b.annotate(load_image(repo_root(), g), g)
    validate_record(rec)
    assert isinstance(rec["valence"], float)
    assert isinstance(rec["arousal"], float)
    assert len(rec["emotion_scores"]) == 8
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_backend_hsemotion.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'open_gesture_annotate.backends.affect_hsemotion'`

- [ ] **Step 4: Write the implementation**

`pipelines/open_gesture_annotate/backends/affect_hsemotion.py`:

```python
"""HSEmotion backend: continuous valence and arousal.

The valence/arousal output is UNDOCUMENTED. HSEmotion's README describes only
discrete emotions. Source inspection of hsemotion/facial_emotions.py on
2026-08-25 shows predict_emotions() computes its softmax over scores[:-2],
which means the enet_b0_8_va_mtl multi-task head appends valence and arousal as
the final two elements. split_scores() asserts that layout so an upstream change
fails loudly instead of writing garbage.

Valence and arousal are what the README's PAD framing (Russell & Mehrabian 1977)
actually calls for, and are the signal `report.py` checks `emotional_state` and
`arousal` against.
"""

from __future__ import annotations

import cv2
import numpy as np

from open_gesture_annotate.io import Gesture
from open_gesture_annotate.schema import ok

MODEL_NAME = "enet_b0_8_va_mtl"
N_EMOTIONS = 8
EXPECTED_SCORE_LEN = N_EMOTIONS + 2  # 8 emotion logits + valence + arousal


def split_scores(scores, classes: list[str]) -> tuple[dict, float, float]:
    """Split the va_mtl output into emotion scores, valence and arousal."""
    scores = np.asarray(scores).ravel()
    if scores.size != EXPECTED_SCORE_LEN:
        raise ValueError(
            f"HSEmotion {MODEL_NAME} returned {scores.size} scores, expected "
            f"{EXPECTED_SCORE_LEN} ({N_EMOTIONS} emotions + valence + arousal). "
            "The undocumented output layout this backend relies on has changed."
        )
    emotions = {name: float(scores[i]) for i, name in enumerate(classes[:N_EMOTIONS])}
    return emotions, float(scores[-2]), float(scores[-1])


class HSEmotionBackend:
    name = "hsemotion-va"
    sidecar = "valence_arousal.json"

    def __init__(self) -> None:
        self._impl = None

    @property
    def version(self) -> str:
        try:
            from importlib.metadata import version

            return version("hsemotion")
        except Exception:
            return "unknown"

    def available(self) -> tuple[bool, str]:
        try:
            from hsemotion.facial_emotions import HSEmotionRecognizer  # noqa: F401
        except ImportError as exc:
            return False, f"hsemotion not installed ({exc}); pip install -e '.[va]'"
        return True, f"hsemotion {self.version} ({MODEL_NAME})"

    def _recognizer(self):
        if self._impl is None:
            from hsemotion.facial_emotions import HSEmotionRecognizer

            self._impl = HSEmotionRecognizer(model_name=MODEL_NAME, device="cpu")
        return self._impl

    def provenance(self) -> dict:
        return {
            "library": {"name": "hsemotion", "version": self.version, "license": "Apache-2.0"},
            "models": [{"name": MODEL_NAME, "role": "emotion + valence/arousal",
                        "license": "Apache-2.0"}],
            "note": "valence/arousal read from scores[-2:]; layout is undocumented "
                    "and asserted at runtime by split_scores().",
        }

    def annotate(self, image: np.ndarray, gesture: Gesture) -> dict:
        fer = self._recognizer()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label, scores = fer.predict_emotions(rgb, logits=False)
        classes = [fer.idx_to_class[i] for i in sorted(fer.idx_to_class)]
        emotions, valence, arousal = split_scores(scores, classes)
        return ok(
            label=str(label),
            valence=valence,
            arousal=arousal,
            emotion_scores=emotions,
            face_source="full-image",
        )
```

- [ ] **Step 5: Run the fast tests**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_backend_hsemotion.py -v
```

Expected: 6 passed, 1 deselected.

- [ ] **Step 6: Run the slow test**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_backend_hsemotion.py -v -m slow
```

Expected: 1 passed.

- [ ] **Step 7: Annotate the dataset and commit**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/og-annotate run --backends va
cd /Users/jin/open-gesture
git add pipelines/open_gesture_annotate/backends/affect_hsemotion.py pipelines/tests/test_backend_hsemotion.py
git commit -m "feat: add HSEmotion valence/arousal backend"
```

---

### Task 10: MediaPipe backend — hands and body

This closes the largest gap in the pipeline. `body_parts` is overwhelmingly hand- and
finger-valued, and no face model can see it.

**Files:**
- Create: `pipelines/open_gesture_annotate/backends/pose_mediapipe.py`
- Create: `pipelines/tests/test_backend_mediapipe.py`

**Interfaces:**
- Consumes: `Backend` protocol, `ok`, `Gesture`.
- Produces: `MediaPipePoseBackend` with `name="mediapipe-pose"`, `sidecar="pose.json"`; `MODEL_URLS: dict[str, str]`; `ensure_models(cache_dir) -> dict[str, Path]`. Record payload: `{"status": "ok", "hands": [{"handedness": "Left"|"Right", "score": float, "landmarks": [[x,y,z],...]}], "hand_count": int, "body": {"landmarks": [[x,y,z,visibility],...]} | None, "body_detected": bool}`.

- [ ] **Step 1: Install and download the task bundles**

```bash
cd /Users/jin/open-gesture/pipelines
.venv/bin/python -m pip install -e ".[pose]"
mkdir -p .models
curl -sSLo .models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
curl -sSLo .models/pose_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker.task
ls -la .models/
```

Expected: two `.task` files, each at least 1 MB. If a URL 404s, find the current one in
the MediaPipe model index and record the working URL in `MODEL_URLS`.
Add `pipelines/.models/` to `.gitignore`.

- [ ] **Step 2: Write the failing test**

`pipelines/tests/test_backend_mediapipe.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_backend_mediapipe.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'open_gesture_annotate.backends.pose_mediapipe'`

- [ ] **Step 4: Write the implementation**

`pipelines/open_gesture_annotate/backends/pose_mediapipe.py`:

```python
"""MediaPipe backend: 21x2 hand landmarks and 33 body pose landmarks.

The `body_parts` field in manifest.json is overwhelmingly hand- and finger-valued.
No face model can see it, so this is the backend that makes `body_parts`
checkable at all.

Landmark coordinates are normalised to [0, 1] in image space; z is relative depth.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from open_gesture_annotate.io import Gesture
from open_gesture_annotate.schema import ok

MODEL_URLS = {
    "hand_landmarker": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "pose_landmarker": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker.task"
    ),
}

DEFAULT_CACHE = Path(__file__).resolve().parents[2] / ".models"


class MediaPipePoseBackend:
    name = "mediapipe-pose"
    sidecar = "pose.json"

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE
        self._hands = None
        self._body = None

    @property
    def version(self) -> str:
        try:
            from importlib.metadata import version

            return version("mediapipe")
        except Exception:
            return "unknown"

    def _model_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.task"

    def available(self) -> tuple[bool, str]:
        try:
            import mediapipe  # noqa: F401
        except ImportError as exc:
            return False, f"mediapipe not installed ({exc}); pip install -e '.[pose]'"
        missing = [k for k in MODEL_URLS if not self._model_path(k).is_file()]
        if missing:
            return False, (
                f"missing model bundle(s): {', '.join(missing)} — download into "
                f"{self.cache_dir} (see MODEL_URLS)"
            )
        return True, f"mediapipe {self.version}"

    def _detectors(self):
        if self._hands is None:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            self._hands = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(
                        model_asset_path=str(self._model_path("hand_landmarker"))
                    ),
                    num_hands=2,
                )
            )
            self._body = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=mp_python.BaseOptions(
                        model_asset_path=str(self._model_path("pose_landmarker"))
                    ),
                    num_poses=1,
                )
            )
        return self._hands, self._body

    def provenance(self) -> dict:
        return {
            "library": {"name": "mediapipe", "version": self.version, "license": "Apache-2.0"},
            "models": [
                {"name": k, "url": url, "license": "Apache-2.0"} for k, url in MODEL_URLS.items()
            ],
        }

    def annotate(self, image: np.ndarray, gesture: Gesture) -> dict:
        import mediapipe as mp

        hands_det, body_det = self._detectors()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        hand_result = hands_det.detect(mp_image)
        hands = []
        for i, landmarks in enumerate(hand_result.hand_landmarks):
            category = hand_result.handedness[i][0]
            hands.append(
                {
                    "handedness": str(category.category_name),
                    "score": float(category.score),
                    "landmarks": [[float(p.x), float(p.y), float(p.z)] for p in landmarks],
                }
            )

        body_result = body_det.detect(mp_image)
        body = None
        if body_result.pose_landmarks:
            body = {
                "landmarks": [
                    [float(p.x), float(p.y), float(p.z), float(getattr(p, "visibility", 0.0))]
                    for p in body_result.pose_landmarks[0]
                ]
            }

        return ok(
            hands=hands,
            hand_count=len(hands),
            body=body,
            body_detected=body is not None,
        )
```

- [ ] **Step 5: Run the fast tests**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_backend_mediapipe.py -v
```

Expected: 4 passed, 3 deselected.

- [ ] **Step 6: Run the slow tests**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_backend_mediapipe.py -v -m slow
```

Expected: 3 passed.

- [ ] **Step 7: Annotate the dataset and commit**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/og-annotate run --backends pose
cd /Users/jin/open-gesture
git add .gitignore pipelines/open_gesture_annotate/backends/pose_mediapipe.py pipelines/tests/test_backend_mediapipe.py
git commit -m "feat: add MediaPipe hand and body pose backend"
```

---

### Task 11: CLIP embedding backend

Implements the embedding space from README §4 and yields an image↔text similarity that
surfaces mislabeled or weak images.

**Files:**
- Create: `pipelines/open_gesture_annotate/backends/embed_clip.py`
- Create: `pipelines/tests/test_backend_clip.py`
- Modify: `pipelines/open_gesture_annotate/cli.py` (add `export-npz` subcommand)

**Interfaces:**
- Consumes: `Backend` protocol, `ok`, `Gesture`, `read_sidecar`, `sidecar_path`.
- Produces: `ClipEmbedBackend` with `name="clip-embed"`, `sidecar="embeddings.json"`; `MODEL_NAME`, `PRETRAINED`; `export_npz(out_dir) -> Path`. Record payload: `{"status": "ok", "image": [float, ...], "intent": [...], "description": [...], "similarity_intent": float, "similarity_description": float, "dim": int}`. All vectors are L2-normalised, so similarity is a plain dot product.

- [ ] **Step 1: Install and confirm the model loads**

```bash
cd /Users/jin/open-gesture/pipelines
.venv/bin/python -m pip install -e ".[embed]"
.venv/bin/python - <<'PY'
import open_clip
m, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tok = open_clip.get_tokenizer("ViT-B-32")
print("ok, embed dim:", m.visual.output_dim)
PY
```

Expected: `ok, embed dim: 512`.

- [ ] **Step 2: Write the failing test**

`pipelines/tests/test_backend_clip.py`:

```python
import numpy as np
import pytest

pytest.importorskip("open_clip")

from open_gesture_annotate.backends.embed_clip import ClipEmbedBackend, export_npz
from open_gesture_annotate.io import load_image, load_manifest, repo_root
from open_gesture_annotate.schema import new_sidecar, ok, validate_record
from open_gesture_annotate.io import write_sidecar


def test_backend_identity():
    b = ClipEmbedBackend()
    assert b.name == "clip-embed"
    assert b.sidecar == "embeddings.json"


def test_provenance_names_an_apache_licence():
    assert "Apache" in ClipEmbedBackend().provenance()["library"]["license"]


def test_export_npz_stacks_records_into_arrays(tmp_path):
    b = ClipEmbedBackend()
    data = new_sidecar(b)
    for gid in ("a", "b"):
        data["records"][gid] = ok(image=[1.0, 0.0], intent=[0.0, 1.0],
                                  description=[1.0, 0.0],
                                  similarity_intent=0.0,
                                  similarity_description=1.0, dim=2)
    write_sidecar(tmp_path / "embeddings.json", data)

    path = export_npz(tmp_path)
    loaded = np.load(path, allow_pickle=False)
    assert list(loaded["ids"]) == ["a", "b"]
    assert loaded["image"].shape == (2, 2)
    assert loaded["intent"].shape == (2, 2)


def test_export_npz_skips_error_records(tmp_path):
    from open_gesture_annotate.schema import error

    b = ClipEmbedBackend()
    data = new_sidecar(b)
    data["records"]["a"] = ok(image=[1.0, 0.0], intent=[0.0, 1.0], description=[1.0, 0.0],
                              similarity_intent=0.0, similarity_description=1.0, dim=2)
    data["records"]["b"] = error("boom")
    write_sidecar(tmp_path / "embeddings.json", data)

    loaded = np.load(export_npz(tmp_path), allow_pickle=False)
    assert list(loaded["ids"]) == ["a"]


@pytest.mark.slow
def test_embeds_a_real_image():
    b = ClipEmbedBackend()
    gestures = {g.id: g for g in load_manifest(repo_root())}
    g = gestures["affirm-01"]
    rec = b.annotate(load_image(repo_root(), g), g)
    validate_record(rec)
    assert rec["dim"] == 512
    assert len(rec["image"]) == 512
    assert abs(float(np.linalg.norm(rec["image"])) - 1.0) < 1e-3
    assert -1.0 <= rec["similarity_intent"] <= 1.0
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_backend_clip.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'open_gesture_annotate.backends.embed_clip'`

- [ ] **Step 4: Write the implementation**

`pipelines/open_gesture_annotate/backends/embed_clip.py`:

```python
"""CLIP backend: image and text embeddings for the README §4 embedding space.

Embeds each image alongside its curated `intent` and `physical_description`, and
records the image-text cosine similarity. Low similarity flags an image whose
visual content disagrees with its label.

All vectors are L2-normalised on write, so cosine similarity is a dot product and
consumers need no renormalisation.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from open_gesture_annotate.io import Gesture
from open_gesture_annotate.schema import ok

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


def _normalise(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


class ClipEmbedBackend:
    name = "clip-embed"
    sidecar = "embeddings.json"

    def __init__(self) -> None:
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    @property
    def version(self) -> str:
        try:
            from importlib.metadata import version

            return version("open_clip_torch")
        except Exception:
            return "unknown"

    def available(self) -> tuple[bool, str]:
        try:
            import open_clip  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            return False, f"open_clip/torch not installed ({exc}); pip install -e '.[embed]'"
        return True, f"open_clip {self.version} ({MODEL_NAME}/{PRETRAINED})"

    def _load(self):
        if self._model is None:
            import open_clip
            import torch

            model, _, preprocess = open_clip.create_model_and_transforms(
                MODEL_NAME, pretrained=PRETRAINED
            )
            model.eval()
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(MODEL_NAME)
            self._torch = torch
        return self._model, self._preprocess, self._tokenizer

    def provenance(self) -> dict:
        return {
            "library": {"name": "open_clip_torch", "version": self.version,
                        "license": "Apache-2.0 / MIT"},
            "models": [{"name": MODEL_NAME, "pretrained": PRETRAINED, "license": "Apache-2.0"}],
        }

    def annotate(self, image: np.ndarray, gesture: Gesture) -> dict:
        from PIL import Image

        model, preprocess, tokenizer = self._load()
        torch = self._torch

        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        intent = gesture.raw.get("intent", "") or ""
        description = gesture.raw.get("physical_description", "") or ""

        with torch.no_grad():
            img_vec = model.encode_image(preprocess(pil).unsqueeze(0))[0].cpu().numpy()
            txt = model.encode_text(tokenizer([intent, description])).cpu().numpy()

        img_vec = _normalise(img_vec.astype(np.float32))
        intent_vec = _normalise(txt[0].astype(np.float32))
        desc_vec = _normalise(txt[1].astype(np.float32))

        return ok(
            image=[round(float(v), 6) for v in img_vec],
            intent=[round(float(v), 6) for v in intent_vec],
            description=[round(float(v), 6) for v in desc_vec],
            similarity_intent=round(float(img_vec @ intent_vec), 6),
            similarity_description=round(float(img_vec @ desc_vec), 6),
            dim=int(img_vec.size),
        )


def export_npz(out_dir: Path) -> Path:
    """Derive annotations/embeddings.npz from embeddings.json for numeric consumers."""
    out_dir = Path(out_dir)
    data = json.loads((out_dir / "embeddings.json").read_text(encoding="utf-8"))
    records = {gid: r for gid, r in data["records"].items() if r.get("status") == "ok"}
    ids = sorted(records)
    path = out_dir / "embeddings.npz"
    np.savez_compressed(
        path,
        ids=np.array(ids, dtype=object).astype(str),
        image=np.array([records[i]["image"] for i in ids], dtype=np.float32),
        intent=np.array([records[i]["intent"] for i in ids], dtype=np.float32),
        description=np.array([records[i]["description"] for i in ids], dtype=np.float32),
    )
    return path
```

- [ ] **Step 5: Add the `export-npz` subcommand to `cli.py`**

In `build_parser()`, after the `report` subparser:

```python
    export = sub.add_parser("export-npz", help="derive annotations/embeddings.npz")
    export.add_argument("--out", default=None, help="annotations directory")
```

Add the handler:

```python
def _cmd_export_npz(args) -> int:
    from open_gesture_annotate.backends.embed_clip import export_npz

    print(f"wrote {export_npz(_resolve_out(args.out))}")
    return 0
```

And in `main()`, before the final `return 2`:

```python
    if args.command == "export-npz":
        return _cmd_export_npz(args)
```

- [ ] **Step 6: Run the fast tests**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_backend_clip.py tests/test_cli.py -v
```

Expected: 4 passed in the CLIP file (1 deselected), 6 passed in the CLI file.

- [ ] **Step 7: Run the slow test**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_backend_clip.py -v -m slow
```

Expected: 1 passed.

- [ ] **Step 8: Annotate, export, and commit**

```bash
cd /Users/jin/open-gesture/pipelines
.venv/bin/og-annotate run --backends embed
.venv/bin/og-annotate export-npz
cd /Users/jin/open-gesture
git add pipelines/open_gesture_annotate/backends/embed_clip.py pipelines/open_gesture_annotate/cli.py pipelines/tests/test_backend_clip.py
git commit -m "feat: add CLIP embedding backend and npz export"
```

---

### Task 12: RTMW whole-body backend (optional extra)

The mmcv/mmpose stack pins hard and installs badly. This task is **allowed to end in a
recorded failure**: if the install does not succeed in reasonable time, commit the backend
with `available()` reporting the reason and move on. Nothing else depends on it.

**Files:**
- Create: `pipelines/open_gesture_annotate/backends/wholebody_rtmw.py`
- Create: `pipelines/tests/test_backend_rtmw.py`

**Interfaces:**
- Consumes: `Backend` protocol, `ok`, `Gesture`.
- Produces: `RTMWBackend` with `name="rtmw-wholebody"`, `sidecar="wholebody.json"`; `KEYPOINT_GROUPS: dict[str, tuple[int, int]]`. Record payload: `{"status": "ok", "person_count": int, "people": [{"keypoints": [[x,y],...133], "scores": [float,...133]}], "groups": {"body": [0,17], "feet": [17,23], "face": [23,91], "hands": [91,133]}}`.

- [ ] **Step 1: Write the failing test**

`pipelines/tests/test_backend_rtmw.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_backend_rtmw.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'open_gesture_annotate.backends.wholebody_rtmw'`

- [ ] **Step 3: Write the implementation**

`pipelines/open_gesture_annotate/backends/wholebody_rtmw.py`:

```python
"""RTMW backend: COCO-WholeBody 133 keypoints (optional extra).

Higher-fidelity hand keypoints than MediaPipe, at the cost of the mmcv/mmpose
dependency stack, which pins hard and installs badly. Isolated behind the
[wholebody] extra: available() reports the reason and the runner skips it, so a
failed install of this backend cannot affect any other.
"""

from __future__ import annotations

import numpy as np

from open_gesture_annotate.io import Gesture
from open_gesture_annotate.schema import ok

# COCO-WholeBody index ranges, [start, end).
KEYPOINT_GROUPS = {
    "body": (0, 17),
    "feet": (17, 23),
    "face": (23, 91),
    "hands": (91, 133),
}


class RTMWBackend:
    name = "rtmw-wholebody"
    sidecar = "wholebody.json"

    def __init__(self) -> None:
        self._impl = None

    @property
    def version(self) -> str:
        try:
            from importlib.metadata import version

            return version("mmpose")
        except Exception:
            return "unknown"

    def available(self) -> tuple[bool, str]:
        try:
            from mmpose.apis import MMPoseInferencer  # noqa: F401
        except Exception as exc:
            return False, (
                f"mmpose not usable ({type(exc).__name__}: {exc}); "
                "pip install -e '.[wholebody]' — this stack pins hard and may not install"
            )
        return True, f"mmpose {self.version}"

    def _inferencer(self):
        if self._impl is None:
            from mmpose.apis import MMPoseInferencer

            self._impl = MMPoseInferencer("wholebody")
        return self._impl

    def provenance(self) -> dict:
        return {
            "library": {"name": "mmpose", "version": self.version, "license": "Apache-2.0"},
            "models": [{"name": "RTMW wholebody", "keypoints": 133, "license": "Apache-2.0"}],
        }

    def annotate(self, image: np.ndarray, gesture: Gesture) -> dict:
        people = []
        for result in self._inferencer()(image):
            for pred in result["predictions"][0]:
                people.append(
                    {
                        "keypoints": [[float(x), float(y)] for x, y in pred["keypoints"]],
                        "scores": [float(s) for s in pred["keypoint_scores"]],
                    }
                )
        return ok(
            person_count=len(people),
            people=people,
            groups={k: list(v) for k, v in KEYPOINT_GROUPS.items()},
        )
```

- [ ] **Step 4: Run the fast tests**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_backend_rtmw.py -v
```

Expected: 4 passed, 1 deselected. These pass **whether or not mmpose is installed** —
that is the point of the isolation.

- [ ] **Step 5: Attempt the install (time-boxed)**

```bash
cd /Users/jin/open-gesture/pipelines
.venv/bin/python -m pip install -e ".[wholebody]" 2>&1 | tail -20
.venv/bin/og-annotate list
```

If the install fails, record the error in the commit message and continue. `og-annotate
list` must still show every other backend as available — verify that before moving on.

- [ ] **Step 6: Commit**

```bash
cd /Users/jin/open-gesture
git add pipelines/open_gesture_annotate/backends/wholebody_rtmw.py pipelines/tests/test_backend_rtmw.py
git commit -m "feat: add optional RTMW whole-body backend"
```

---

### Task 13: Provenance manifest

**Files:**
- Create: `pipelines/open_gesture_annotate/provenance.py`
- Create: `pipelines/tests/test_provenance.py`
- Modify: `pipelines/open_gesture_annotate/cli.py` (write `_meta.json` at the end of `run`)

**Interfaces:**
- Consumes: `registry.availability`, `registry.get`.
- Produces: `PERMISSIVE = {"MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause"}`; `is_permissive(license_str) -> bool`; `collect(keys) -> dict`; `write_meta(out_dir, keys) -> Path`; `licence_warnings(meta) -> list[str]`.

- [ ] **Step 1: Write the failing test**

`pipelines/tests/test_provenance.py`:

```python
import json

from open_gesture_annotate import registry
from open_gesture_annotate.provenance import (
    collect,
    is_permissive,
    licence_warnings,
    write_meta,
)
from tests.conftest import FakeBackend


def test_recognises_permissive_licences():
    assert is_permissive("MIT")
    assert is_permissive("Apache-2.0")
    assert is_permissive("BSD-3-Clause")


def test_rejects_non_permissive_licences():
    assert not is_permissive("CC-BY-NC-4.0")
    assert not is_permissive("AGPL-3.0")
    assert not is_permissive("check upstream weight licence")


def test_collect_records_each_backend(tmp_path):
    registry.register("prov-fake", FakeBackend)
    meta = collect(["prov-fake"])
    assert meta["backends"]["prov-fake"]["available"] is True
    assert meta["backends"]["prov-fake"]["provenance"]["models"][0]["license"] == "MIT"


def test_collect_records_unavailable_backends_without_raising():
    registry.register("prov-down", lambda: FakeBackend(available=False, reason="nope"))
    meta = collect(["prov-down"])
    assert meta["backends"]["prov-down"]["available"] is False
    assert meta["backends"]["prov-down"]["reason"] == "nope"


def test_no_warnings_for_an_all_permissive_run():
    registry.register("prov-fake", FakeBackend)
    assert licence_warnings(collect(["prov-fake"])) == []


def test_warns_about_a_non_permissive_weight():
    class Restricted(FakeBackend):
        name = "restricted"

        def provenance(self):
            return {"models": [{"name": "w", "license": "CC-BY-NC-4.0"}]}

    registry.register("prov-nc", Restricted)
    warnings = licence_warnings(collect(["prov-nc"]))
    assert len(warnings) == 1
    assert "CC-BY-NC-4.0" in warnings[0]


def test_write_meta_produces_readable_json(tmp_path):
    registry.register("prov-fake", FakeBackend)
    path = write_meta(tmp_path, ["prov-fake"])
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "_meta.json"
    assert "generated_at" in meta
    assert "prov-fake" in meta["backends"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_provenance.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'open_gesture_annotate.provenance'`

- [ ] **Step 3: Write `provenance.py`**

```python
"""Record what produced the annotations, and under which licence.

The repo promises MIT code and CC-BY-4.0 images with commercial use permitted.
Pretrained weights are licensed separately from the libraries that load them, so
a run that silently pulls a non-commercial weight would quietly falsify that
promise. This module makes every weight's licence explicit and warns on any that
is not permissive.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from open_gesture_annotate import registry

PERMISSIVE = {"MIT", "Apache-2.0", "Apache 2.0", "BSD-3-Clause", "BSD-2-Clause",
              "Apache-2.0 / MIT"}


def is_permissive(license_str: str | None) -> bool:
    return bool(license_str) and license_str.strip() in PERMISSIVE


def collect(keys: list[str]) -> dict:
    backends = {}
    for key in keys:
        entry: dict = {}
        try:
            backend = registry.get(key)
            available, reason = backend.available()
            entry = {
                "name": backend.name,
                "version": backend.version,
                "sidecar": backend.sidecar,
                "available": bool(available),
                "reason": reason,
                "provenance": backend.provenance(),
            }
        except Exception as exc:
            entry = {"available": False, "reason": f"{type(exc).__name__}: {exc}",
                     "provenance": {}}
        backends[key] = entry
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backends": backends,
    }


def licence_warnings(meta: dict) -> list[str]:
    """One warning per model whose licence is not known-permissive."""
    warnings = []
    for key, entry in meta["backends"].items():
        for model in entry.get("provenance", {}).get("models", []):
            lic = model.get("license")
            if not is_permissive(lic):
                warnings.append(
                    f"{key}: model {model.get('name', '?')!r} has licence {lic!r} — "
                    "verify before commercial redistribution"
                )
    return warnings


def write_meta(out_dir: Path, keys: list[str]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = collect(keys)
    meta["licence_warnings"] = licence_warnings(meta)
    path = out_dir / "_meta.json"
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
```

- [ ] **Step 4: Wire it into `cli.py`**

At the end of `_cmd_run`, before `return 0`:

```python
    from open_gesture_annotate.provenance import licence_warnings, write_meta

    meta_path = write_meta(out_dir, keys)
    print(f"wrote {meta_path}")
    for warning in licence_warnings(json.loads(meta_path.read_text(encoding="utf-8"))):
        print(f"  LICENCE: {warning}", file=sys.stderr)
    return 0
```

Add `import json` to the top of `cli.py`.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_provenance.py tests/test_cli.py -v
```

Expected: 7 passed in provenance, 6 passed in CLI.

- [ ] **Step 6: Commit**

```bash
cd /Users/jin/open-gesture
git add pipelines/open_gesture_annotate/provenance.py pipelines/open_gesture_annotate/cli.py pipelines/tests/test_provenance.py
git commit -m "feat: record model provenance and warn on non-permissive weights"
```

---

### Task 14: Quality report — the cross-check

This is the deliverable that earns the pipeline. It reports disagreements and asserts
nothing about which side is wrong. **It never rewrites `manifest.json`.**

**Files:**
- Create: `pipelines/open_gesture_annotate/report.py`
- Create: `pipelines/tests/test_report.py`

**Interfaces:**
- Consumes: `load_manifest`, `repo_root`, sidecars from Tasks 7–11.
- Produces: `Finding` dataclass (`gesture_id: str`, `check: str`, `severity: str`, `curated: str`, `predicted: str`, `note: str`); `check_people_count(gestures, faces) -> list[Finding]`; `check_valence(gestures, va) -> list[Finding]`; `check_arousal(gestures, va) -> list[Finding]`; `check_body_parts(gestures, pose) -> list[Finding]`; `check_similarity(gestures, embeddings) -> list[Finding]`; `build_findings(root, out_dir) -> list[Finding]`; `write_report(root, out_dir) -> Path`.

Expected-count mapping for `number_of_people`: `single` → 1, `2 person` → 2,
`3 or more` → 3+.

- [ ] **Step 1: Write the failing test**

`pipelines/tests/test_report.py`:

```python
import pytest

from open_gesture_annotate.io import Gesture
from open_gesture_annotate.report import (
    check_arousal,
    check_body_parts,
    check_people_count,
    check_similarity,
    check_valence,
    write_report,
)


def _gesture(gid, **raw):
    return Gesture(id=gid, index=1, name="G", category="C", file=f"{gid}.png", raw=raw)


# --- number_of_people ---


def test_single_matching_one_face_is_not_flagged():
    g = [_gesture("a", number_of_people="single")]
    assert check_people_count(g, {"a": {"status": "ok", "face_count": 1}}) == []


def test_single_with_two_faces_is_flagged():
    g = [_gesture("a", number_of_people="single")]
    findings = check_people_count(g, {"a": {"status": "ok", "face_count": 2}})
    assert len(findings) == 1
    assert findings[0].check == "number_of_people"
    assert findings[0].predicted == "2"


def test_three_or_more_with_three_faces_is_not_flagged():
    g = [_gesture("a", number_of_people="3 or more")]
    assert check_people_count(g, {"a": {"status": "ok", "face_count": 4}}) == []


def test_an_error_record_is_skipped_not_flagged():
    g = [_gesture("a", number_of_people="single")]
    assert check_people_count(g, {"a": {"status": "error", "error": "x"}}) == []


def test_a_missing_record_is_skipped():
    assert check_people_count([_gesture("a", number_of_people="single")], {}) == []


# --- valence ---


def test_positive_label_with_positive_valence_is_not_flagged():
    g = [_gesture("a", emotional_state="positive")]
    assert check_valence(g, {"a": {"status": "ok", "valence": 0.6}}) == []


def test_positive_label_with_negative_valence_is_flagged():
    g = [_gesture("a", emotional_state="positive")]
    findings = check_valence(g, {"a": {"status": "ok", "valence": -0.7}})
    assert len(findings) == 1
    assert findings[0].check == "emotional_state"


def test_neutral_label_is_never_flagged_on_sign():
    g = [_gesture("a", emotional_state="neutral")]
    assert check_valence(g, {"a": {"status": "ok", "valence": -0.9}}) == []


def test_a_valence_near_zero_is_not_flagged():
    """Only a confident contradiction counts; near-zero valence is not evidence."""
    g = [_gesture("a", emotional_state="positive")]
    assert check_valence(g, {"a": {"status": "ok", "valence": -0.05}}) == []


# --- arousal ---


# Terciles need at least 3 scored gestures, so these use 6:
# ranks 0-1 -> low, 2-3 -> medium, 4-5 -> high.
BUCKETS = ["low", "low", "medium", "medium", "high", "high"]


def _arousal_case(values):
    gestures = [_gesture(chr(ord("a") + i), arousal=BUCKETS[i]) for i in range(6)]
    va = {g.id: {"status": "ok", "arousal": v} for g, v in zip(gestures, values)}
    return gestures, va


def test_arousal_agreeing_with_the_predicted_terciles_is_not_flagged():
    gestures, va = _arousal_case([0.1, 0.2, 0.4, 0.5, 0.8, 0.9])
    assert check_arousal(gestures, va) == []


def test_arousal_at_the_opposite_extreme_is_flagged():
    """Curated order fully reversed: the two 'low' and two 'high' gestures swap terciles."""
    gestures, va = _arousal_case([0.9, 0.8, 0.5, 0.4, 0.2, 0.1])
    findings = check_arousal(gestures, va)
    assert {f.gesture_id for f in findings} == {"a", "b", "e", "f"}


def test_middle_buckets_are_never_flagged():
    """Only opposite extremes count; low-vs-medium is not a contradiction worth reporting."""
    gestures, va = _arousal_case([0.1, 0.2, 0.4, 0.5, 0.8, 0.9])
    assert all(f.gesture_id not in ("c", "d") for f in check_arousal(gestures, va))


def test_too_few_scored_gestures_yields_no_terciles():
    g = [_gesture("a", arousal="low"), _gesture("b", arousal="high")]
    va = {"a": {"status": "ok", "arousal": 0.9}, "b": {"status": "ok", "arousal": 0.1}}
    assert check_arousal(g, va) == []


# --- body_parts ---


def test_hand_claim_with_a_detected_hand_is_not_flagged():
    g = [_gesture("a", body_parts=["hand", "thumb"])]
    assert check_body_parts(g, {"a": {"status": "ok", "hand_count": 1, "body_detected": True}}) == []


def test_hand_claim_with_no_detected_hand_is_flagged():
    g = [_gesture("a", body_parts=["hand", "thumb"])]
    findings = check_body_parts(g, {"a": {"status": "ok", "hand_count": 0, "body_detected": True}})
    assert len(findings) == 1
    assert findings[0].check == "body_parts"


def test_a_gesture_claiming_no_hand_is_not_flagged():
    g = [_gesture("a", body_parts=["head"])]
    assert check_body_parts(g, {"a": {"status": "ok", "hand_count": 0, "body_detected": True}}) == []


# --- similarity ---


def test_the_bottom_decile_of_similarity_is_flagged():
    gestures = [_gesture(f"g{i}") for i in range(10)]
    emb = {f"g{i}": {"status": "ok", "similarity_intent": i / 10} for i in range(10)}
    findings = check_similarity(gestures, emb)
    assert [f.gesture_id for f in findings] == ["g0"]


# --- report ---


def test_write_report_creates_a_markdown_file(tmp_path, monkeypatch):
    monkeypatch.setattr("open_gesture_annotate.report.load_manifest", lambda root: [])
    path = write_report(tmp_path, tmp_path)
    assert path.name == "quality_report.md"
    assert "# Annotation Quality Report" in path.read_text(encoding="utf-8")


def test_write_report_never_touches_the_manifest(tmp_path, monkeypatch):
    from open_gesture_annotate.io import repo_root

    before = (repo_root() / "manifest.json").read_bytes()
    monkeypatch.setattr("open_gesture_annotate.report.load_manifest", lambda root: [])
    write_report(tmp_path, tmp_path)
    assert (repo_root() / "manifest.json").read_bytes() == before
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_report.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'open_gesture_annotate.report'`

- [ ] **Step 3: Write `report.py`**

```python
"""Cross-check curated metadata against model predictions.

Reports disagreements. Asserts nothing about which side is wrong — a disagreement
is a prompt for human review, not an automated correction. This module never
writes to manifest.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from open_gesture_annotate.io import Gesture, load_manifest

VALENCE_CONFIDENCE = 0.15  # below this magnitude, valence is not evidence either way
PEOPLE_EXPECTED = {"single": 1, "2 person": 2, "3 or more": 3}


@dataclass
class Finding:
    gesture_id: str
    check: str
    severity: str  # "high" | "medium" | "low"
    curated: str
    predicted: str
    note: str


def _ok_records(path: Path) -> dict:
    """Load a sidecar's successful records, or {} if the sidecar is absent."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {gid: r for gid, r in data.get("records", {}).items() if r.get("status") == "ok"}


def check_people_count(gestures: list[Gesture], faces: dict) -> list[Finding]:
    findings = []
    for g in gestures:
        rec = faces.get(g.id)
        if not rec or rec.get("status") != "ok":
            continue
        curated = g.raw.get("number_of_people")
        expected = PEOPLE_EXPECTED.get(curated)
        if expected is None:
            continue
        detected = int(rec.get("face_count", 0))
        matches = detected >= 3 if curated == "3 or more" else detected == expected
        if not matches:
            findings.append(Finding(
                g.id, "number_of_people", "high", str(curated), str(detected),
                "detected face count disagrees with the curated participant bucket",
            ))
    return findings


def check_valence(gestures: list[Gesture], va: dict) -> list[Finding]:
    findings = []
    for g in gestures:
        rec = va.get(g.id)
        if not rec or rec.get("status") != "ok":
            continue
        curated = g.raw.get("emotional_state")
        if curated not in ("positive", "negative"):
            continue  # neutral makes no sign claim
        valence = float(rec.get("valence", 0.0))
        if abs(valence) < VALENCE_CONFIDENCE:
            continue  # not a confident contradiction
        predicted = "positive" if valence > 0 else "negative"
        if predicted != curated:
            findings.append(Finding(
                g.id, "emotional_state", "medium", curated, f"{predicted} ({valence:+.2f})",
                "predicted facial valence sign contradicts the curated label",
            ))
    return findings


def check_arousal(gestures: list[Gesture], va: dict) -> list[Finding]:
    """Compare the curated low/medium/high bucket against predicted terciles."""
    scored = [(g, float(va[g.id]["arousal"]))
              for g in gestures
              if va.get(g.id, {}).get("status") == "ok"
              and g.raw.get("arousal") in ("low", "medium", "high")]
    if len(scored) < 3:
        return []

    ordered = sorted(scored, key=lambda pair: pair[1])
    third = max(1, len(ordered) // 3)
    tercile = {}
    for rank, (g, _) in enumerate(ordered):
        tercile[g.id] = "low" if rank < third else ("high" if rank >= 2 * third else "medium")

    findings = []
    for g, value in scored:
        curated = g.raw["arousal"]
        predicted = tercile[g.id]
        if {curated, predicted} == {"low", "high"}:  # only flag opposite extremes
            findings.append(Finding(
                g.id, "arousal", "medium", curated, f"{predicted} ({value:+.2f})",
                "curated arousal bucket sits at the opposite extreme of the predicted tercile",
            ))
    return findings


def check_body_parts(gestures: list[Gesture], pose: dict) -> list[Finding]:
    hand_words = {"hand", "hands", "finger", "fingers", "thumb", "palm", "fist", "wrist"}
    findings = []
    for g in gestures:
        rec = pose.get(g.id)
        if not rec or rec.get("status") != "ok":
            continue
        parts = {str(p).lower() for p in g.raw.get("body_parts", [])}
        if parts & hand_words and int(rec.get("hand_count", 0)) == 0:
            findings.append(Finding(
                g.id, "body_parts", "high", ", ".join(sorted(parts)), "0 hands detected",
                "curated body_parts claims a hand but MediaPipe detected none",
            ))
    return findings


def check_similarity(gestures: list[Gesture], embeddings: dict) -> list[Finding]:
    """Flag the bottom decile of image-to-intent similarity."""
    scored = [(g, float(embeddings[g.id]["similarity_intent"]))
              for g in gestures
              if embeddings.get(g.id, {}).get("status") == "ok"]
    if len(scored) < 10:
        return []
    ordered = sorted(scored, key=lambda pair: pair[1])
    cutoff = max(1, len(ordered) // 10)
    return [
        Finding(g.id, "intent_similarity", "low", g.raw.get("intent", ""), f"{score:.3f}",
                "image-to-intent similarity is in the bottom decile; possible weak or "
                "mislabelled image")
        for g, score in ordered[:cutoff]
    ]


def build_findings(root: Path, out_dir: Path) -> list[Finding]:
    gestures = load_manifest(root)
    out_dir = Path(out_dir)
    faces = _ok_records(out_dir / "faces.json")
    va = _ok_records(out_dir / "valence_arousal.json")
    pose = _ok_records(out_dir / "pose.json")
    embeddings = _ok_records(out_dir / "embeddings.json")

    findings = (
        check_people_count(gestures, faces)
        + check_body_parts(gestures, pose)
        + check_valence(gestures, va)
        + check_arousal(gestures, va)
        + check_similarity(gestures, embeddings)
    )
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(findings, key=lambda f: (order[f.severity], f.check, f.gesture_id))


def write_report(root: Path, out_dir: Path) -> Path:
    gestures = {g.id: g for g in load_manifest(root)}
    findings = build_findings(root, out_dir)

    lines = [
        "# Annotation Quality Report",
        "",
        "Disagreements between the curated metadata in `manifest.json` and the model",
        "predictions under `annotations/`. A disagreement is a prompt for human review,",
        "not a verdict: either side may be wrong. Nothing here has been applied to",
        "`manifest.json`.",
        "",
        f"**{len(findings)} finding(s)** across {len(gestures)} gestures.",
        "",
    ]

    if not findings:
        lines.append("No disagreements found.")
    else:
        lines += [
            "| Severity | Check | Gesture | Curated | Predicted | Image |",
            "|---|---|---|---|---|---|",
        ]
        for f in findings:
            gesture = gestures.get(f.gesture_id)
            image = f"`{gesture.file}`" if gesture else ""
            curated = str(f.curated).replace("|", "\\|")[:60]
            lines.append(
                f"| {f.severity} | {f.check} | `{f.gesture_id}` | {curated} "
                f"| {f.predicted} | {image} |"
            )
        lines += ["", "## Notes", ""]
        for check in sorted({f.check for f in findings}):
            note = next(f.note for f in findings if f.check == check)
            lines.append(f"- **{check}** — {note}")

    path = Path(out_dir) / "quality_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_report.py -v
```

Expected: 21 passed.

- [ ] **Step 5: Generate the real report**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/og-annotate report
cat ../annotations/quality_report.md
```

Read the findings. Report the counts to the user — this is the substantive result of the
whole pipeline. Do **not** edit `manifest.json` in response to any finding.

- [ ] **Step 6: Commit**

```bash
cd /Users/jin/open-gesture
git add pipelines/open_gesture_annotate/report.py pipelines/tests/test_report.py
git add -f annotations/quality_report.md annotations/_meta.json
git commit -m "feat: add curated-vs-predicted quality report"
```

---

### Task 15: Documentation and the OpenPose licence fix

The README recommends OpenPose while the licence table promises commercial use. OpenPose
is under a CMU non-commercial licence, so those two claims contradict. This task resolves
the contradiction and documents the pipeline.

**Files:**
- Modify: `README.md` (the "Hierarchical Recognition Pipeline" section)
- Create: `pipelines/README.md`

- [ ] **Step 1: Find the OpenPose reference**

```bash
grep -n "OpenPose\|MediaPipe" README.md
```

Expected: one hit inside the Stage 2 block of "3. Hierarchical Recognition Pipeline",
reading `skeletal models (MediaPipe, OpenPose)`.

- [ ] **Step 2: Replace it**

Change that line to:

```
  skeletal models (MediaPipe, MMPose/RTMW)
```

MediaPipe is Apache-2.0 and MMPose is Apache-2.0; both are compatible with this repo's
commercial-use promise, and both are what `pipelines/` actually uses.

- [ ] **Step 3: Verify no non-commercial tool is still recommended**

```bash
cd /Users/jin/open-gesture && grep -n "OpenPose\|Ultralytics\|YOLOv8\|Sapiens\|LibreFace\|OpenFace" README.md || echo "clean"
```

Expected: `clean`.

- [ ] **Step 4: Write `pipelines/README.md`**

```markdown
# Annotation Pipeline

Runs face, affect, hand/body, and semantic models over the Open Gesture image set
and writes the results to `../annotations/`. `manifest.json` is never modified —
it stays the curated ground truth, and this pipeline is how you check it.

## Setup

Requires Python >=3.10,<3.15.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,face,aus,va,pose,embed]"
```

MediaPipe needs two task bundles downloaded into `.models/` — see `MODEL_URLS` in
`open_gesture_annotate/backends/pose_mediapipe.py`.

The `[wholebody]` extra (RTMW/mmpose) is optional and pins hard. If it fails to
install, every other backend still works.

## Usage

```bash
.venv/bin/og-annotate list                        # backends and availability
.venv/bin/og-annotate run                         # all available backends
.venv/bin/og-annotate run --backends face,pose    # a subset
.venv/bin/og-annotate run --force --limit 5       # re-run a sample
.venv/bin/og-annotate report                      # regenerate quality_report.md
.venv/bin/og-annotate export-npz                  # derive embeddings.npz
```

Runs are idempotent and resumable: successful records are skipped, previous errors
are retried. A backend that will not import is skipped with a reason, never fatally.

## Backends

| Key | Model | Sidecar | Licence |
|---|---|---|---|
| `face` | uniface — bbox, landmarks, head pose, gaze, emotion, demographics | `faces.json` | MIT (weights vary) |
| `aus` | py-feat — 20 FACS Action Units, emotion, 3D head pose | `action_units.json` | MIT |
| `va` | HSEmotion — continuous valence/arousal | `valence_arousal.json` | Apache-2.0 |
| `pose` | MediaPipe — 21x2 hand + 33 body landmarks | `pose.json` | Apache-2.0 |
| `embed` | CLIP ViT-B-32 — image and text embeddings | `embeddings.json` | Apache-2.0 |
| `wholebody` | RTMW — 133 COCO-WholeBody keypoints (optional) | `wholebody.json` | Apache-2.0 |

## Output

`annotations/quality_report.md` is the point of all this: it flags where curated
metadata and model predictions disagree. It reports, it never corrects.
`annotations/_meta.json` records every model version and weight licence, and warns
about any that is not permissive.

## Testing

```bash
.venv/bin/python -m pytest              # fast tests, no model weights
.venv/bin/python -m pytest -m slow      # real inference against real weights
```

## Design

See `../docs/superpowers/specs/2026-08-25-multimodal-annotation-pipeline-design.md`.
```

- [ ] **Step 5: Run the whole fast suite**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest -v
```

Expected: all fast tests pass, slow tests deselected.

- [ ] **Step 6: Commit**

```bash
cd /Users/jin/open-gesture
git add README.md pipelines/README.md
git commit -m "docs: document the annotation pipeline and fix the OpenPose licence contradiction"
```

---

### Task 16: Golden-file regression tests

Spec Testing item 4. The fake-backend suite proves the *plumbing* is correct; nothing yet
proves the *models* keep producing the same answers. These pin three representative images
so an upstream weight or version change shows up as a test failure rather than as silently
different data.

**Files:**
- Create: `pipelines/tests/test_golden.py`
- Create: `pipelines/tests/golden/README.md`
- Create (generated): `pipelines/tests/golden/*.json`

**Interfaces:**
- Consumes: every backend from Tasks 7–11; `load_image`, `load_manifest`, `repo_root`.
- Produces: `GOLDEN_CASES: dict[str, str]`; `golden_dir() -> Path`; `numeric_leaves(obj, prefix="") -> dict[str, float]`.

- [ ] **Step 1: Choose the three representative images**

```bash
cd /Users/jin/open-gesture
python3 -c "
import json
m=json.load(open('manifest.json'))
for g in m['gestures']:
    if g['number_of_people'] != 'single':
        print(g['id'], '|', g['number_of_people'], '|', g['name'])
"
```

Pick one id from that list as the multi-person case. Use `affirm-01` (single person, clear
face, one hand) as the first case. For the third, pick a gesture whose `body_parts` has no
`head`/`face` entry — a hands-only case that stresses the no-face path. Record all three
in `GOLDEN_CASES` below, replacing the placeholders with the ids you chose.

- [ ] **Step 2: Write the test**

`pipelines/tests/test_golden.py`:

```python
"""Golden-file regression tests.

Numeric outputs are compared with tolerance, never for exact equality: model
inference varies slightly across BLAS builds and platforms. Structural fields
(counts, labels, landmark cardinality) are compared exactly, because those
changing means the model or its output contract changed.

Regenerate after a deliberate model upgrade:
    REGENERATE_GOLDEN=1 .venv/bin/python -m pytest tests/test_golden.py -m slow
Then read the diff before committing it.
"""

import json
import os
from pathlib import Path

import pytest

from open_gesture_annotate.io import load_image, load_manifest, repo_root

# Replace with the ids chosen in Step 1.
GOLDEN_CASES = {
    "single_clear_face": "affirm-01",
    "multi_person": "<id from Step 1>",
    "hands_only": "<id from Step 1>",
}

BACKENDS = {
    "face": ("open_gesture_annotate.backends.face_uniface", "UniFaceBackend"),
    "aus": ("open_gesture_annotate.backends.affect_pyfeat", "PyFeatBackend"),
    "va": ("open_gesture_annotate.backends.affect_hsemotion", "HSEmotionBackend"),
    "pose": ("open_gesture_annotate.backends.pose_mediapipe", "MediaPipePoseBackend"),
}

TOLERANCE = 1e-3
STRUCTURAL = ("face_count", "hand_count", "person_count", "body_detected", "label", "dim")


def golden_dir() -> Path:
    return Path(__file__).parent / "golden"


def numeric_leaves(obj, prefix="") -> dict:
    """Flatten every float in a record to a dotted path, for tolerant comparison."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(numeric_leaves(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(numeric_leaves(v, f"{prefix}[{i}]"))
    elif isinstance(obj, bool):
        pass  # structural, compared exactly
    elif isinstance(obj, (int, float)):
        out[prefix] = float(obj)
    return out


def _load_backend(key):
    import importlib

    module, cls = BACKENDS[key]
    return getattr(importlib.import_module(module), cls)()


@pytest.mark.slow
@pytest.mark.parametrize("case,gesture_id", sorted(GOLDEN_CASES.items()))
@pytest.mark.parametrize("backend_key", sorted(BACKENDS))
def test_matches_golden(case, gesture_id, backend_key):
    if gesture_id.startswith("<"):
        pytest.skip("golden case id not yet chosen (Task 16 Step 1)")

    backend = _load_backend(backend_key)
    if not backend.available()[0]:
        pytest.skip(f"{backend_key} unavailable")

    gestures = {g.id: g for g in load_manifest(repo_root())}
    gesture = gestures[gesture_id]
    actual = backend.annotate(load_image(repo_root(), gesture), gesture)

    path = golden_dir() / f"{backend_key}__{case}.json"

    if os.environ.get("REGENERATE_GOLDEN"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip(f"regenerated {path.name}")

    if not path.is_file():
        pytest.fail(f"missing golden file {path.name}; run with REGENERATE_GOLDEN=1")

    expected = json.loads(path.read_text(encoding="utf-8"))

    # Structural fields must match exactly.
    for key in STRUCTURAL:
        if key in expected:
            assert actual.get(key) == expected[key], f"{key} changed"

    # Numeric leaves must match within tolerance, with the same shape.
    exp_nums, act_nums = numeric_leaves(expected), numeric_leaves(actual)
    assert set(exp_nums) == set(act_nums), (
        f"output shape changed: "
        f"{sorted(set(exp_nums) ^ set(act_nums))[:5]}"
    )
    drifted = {k: (exp_nums[k], act_nums[k])
               for k in exp_nums if abs(exp_nums[k] - act_nums[k]) > TOLERANCE}
    assert not drifted, f"{len(drifted)} value(s) drifted beyond {TOLERANCE}: {list(drifted)[:5]}"


def test_numeric_leaves_flattens_nested_structures():
    flat = numeric_leaves({"a": 1, "b": [{"c": 2.5}], "d": True, "e": "x"})
    assert flat == {"a": 1.0, "b[0].c": 2.5}


def test_numeric_leaves_excludes_booleans():
    assert numeric_leaves({"body_detected": True}) == {}
```

- [ ] **Step 3: Run the fast tests to verify they fail**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_golden.py -v
```

Expected: FAIL — `ModuleNotFoundError` or `NameError` until the file exists; once written,
2 passed (the `numeric_leaves` unit tests) and the parametrised ones deselected.

- [ ] **Step 4: Fill in the two remaining case ids**

Edit `GOLDEN_CASES` with the ids chosen in Step 1. Re-run Step 3; the two unit tests must
still pass.

- [ ] **Step 5: Generate the golden files**

```bash
cd /Users/jin/open-gesture/pipelines
REGENERATE_GOLDEN=1 .venv/bin/python -m pytest tests/test_golden.py -v -m slow
ls tests/golden/
```

Expected: up to 12 JSON files (4 backends x 3 cases), minus any skipped because a backend
is unavailable. **Read at least one file** before committing — confirm the values look
plausible (a face count of 1 for `affirm-01`, 21 landmarks per hand) rather than
rubber-stamping whatever the models emitted.

- [ ] **Step 6: Verify the golden files now pass**

```bash
cd /Users/jin/open-gesture/pipelines && .venv/bin/python -m pytest tests/test_golden.py -v -m slow
```

Expected: all parametrised cases pass (or skip if a backend is unavailable). If a case
fails immediately after generation, the backend is non-deterministic — record which one
and raise it rather than widening `TOLERANCE` to hide it.

- [ ] **Step 7: Document the golden directory**

`pipelines/tests/golden/README.md`:

```markdown
# Golden files

Recorded backend outputs for three representative gesture images, used as
regression tests. Numeric values are compared with a 1e-3 tolerance; structural
fields (counts, labels) are compared exactly.

These files are committed deliberately, unlike `annotations/`, because they are
test fixtures rather than generated data.

Regenerate after a deliberate model or version upgrade:

    REGENERATE_GOLDEN=1 .venv/bin/python -m pytest tests/test_golden.py -m slow

Always read the resulting diff before committing it. An unexplained change here
means a model started behaving differently.
```

- [ ] **Step 8: Commit**

```bash
cd /Users/jin/open-gesture
git add pipelines/tests/test_golden.py
git add -f pipelines/tests/golden/
git commit -m "test: pin backend outputs with golden-file regression tests"
```

---

## Verification

After all tasks, confirm end to end:

```bash
cd /Users/jin/open-gesture/pipelines
.venv/bin/python -m pytest -v                       # fast suite green
.venv/bin/python -m pytest -v -m slow                # real-weight + golden suite green
.venv/bin/og-annotate list                           # availability per backend
.venv/bin/og-annotate run                            # full run, all backends
.venv/bin/og-annotate report
cd /Users/jin/open-gesture
git status --short                                   # only intended files
git log --oneline 642ecfc..HEAD -- manifest.json manifest.md gesture_images/
```

That final `git log` lists every commit since this work began that touched curated data.
It **must be empty**. If it is not, curated data was modified and the change must be
reverted — `manifest.json` is ground truth, and this pipeline only reads it.
