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

# Chosen in Step 1:
#   single_clear_face -> affirm-01 (Thumbs Up; one clearly detected face, one hand)
#   multi_person       -> greet-05 (High Five, 2 person; exercises multi-hand detection)
#   hands_only          -> bicycle-01 (Left Turn Signal, shot from behind; no face at all,
#                           exercising every backend's zero-face fallback path)
GOLDEN_CASES = {
    "single_clear_face": "affirm-01",
    "multi_person": "greet-05",
    "hands_only": "bicycle-01",
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
