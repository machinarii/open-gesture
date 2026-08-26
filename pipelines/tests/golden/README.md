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

## The three cases

- `single_clear_face` -> `affirm-01` (Thumbs Up; one clearly detected face, one hand).
- `multi_person` -> `greet-05` (High Five, 2 person; exercises multi-hand detection).
- `hands_only` -> `bicycle-01` (Left Turn Signal, shot from behind; no detectable
  face at all, exercising every backend's zero-face fallback path).

## `va` (HSEmotion) depends on `annotations/faces.json`

The `va` backend crops to the detected face before running inference, reading
that crop from `annotations/faces.json` (written by the `face` backend). When no
face is recorded for a gesture, it falls back to running on the full image
instead (visible as `"face_source": "full-image"` in `va__hands_only.json`,
versus `"face_source": "uniface"` for the two cases with a detected face).

This means the `va` golden files are only reproducible as long as
`annotations/faces.json` exists and is current for these three gesture ids. If
that sidecar is ever regenerated (a new face-detection model, a re-run of the
`face` backend, etc.), the face crop fed into HSEmotion can shift, and the `va`
golden files may need to be regenerated too — even if the `va` backend itself
did not change.
