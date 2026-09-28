# Golden files

Recorded backend outputs for three representative gesture images, used as
regression tests. Numeric values are compared with a 1e-3 tolerance. Top-level
structural fields (counts, labels) are compared exactly, and so is every string
output at any depth. The nested check matters for uniface's per-face
`faces[i].emotion.label`: the top-level check never reached it, so the face
backend's only emotion output went unverified.

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

## Cross-platform drift

These files were generated on a different platform and library stack from the
Linux/aarch64 machine they were last run on (torch 2.14, opencv 5.0,
scikit-learn 1.9). There, three cases fail: `aus` and `face` on
`single_clear_face`, and `face` on `multi_person`. The cause is uniface landmark
drift of at most 0.13 px, carried into py-feat AUs (at most 0.026). Every
categorical output, including all emotion labels, still matches, and `va`
valence/arousal is bit-identical. Treat that pattern as platform noise. Treat a
changed label, count or shape as a real regression. Regenerating on one
machine would make the files fail on the other, so decide per platform rather
than regenerating reflexively.
