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

### macOS: `libomp` is required

Install it before installing the `aus` extra:

```bash
brew install libomp
```

The `aus` extra pulls in py-feat, whose action-unit classifier is xgboost.
xgboost's compiled extension links against `libomp.dylib` and will not load on
macOS without it — the import fails at runtime, not at pip-install time, so the
failure shows up later than you'd expect and can look unrelated to the `aus`
extra at first glance.

### macOS/arm64: the opencv conflict (not optional)

mediapipe (installed by the `pose` extra) depends on `opencv-contrib-python`,
which installs into the same `cv2/` package directory as this project's own
`opencv-python` dependency — roughly 150 overlapping files. The two
distributions are not meant to coexist. Having both installed at once causes an
intermittent `libc++abi` abort (`recursive_mutex lock failed` or similar) at
interpreter shutdown, affecting the *entire* test suite, not just pose tests —
a run can pass every test and still abort on exit. Exactly one opencv
distribution may be installed. After installing the `pose` extra, run:

```bash
.venv/bin/python -m pip uninstall -y opencv-contrib-python opencv-python
.venv/bin/python -m pip install --force-reinstall --no-deps "opencv-python>=4.8"
```

`pip check` will then report mediapipe's `opencv-contrib-python` requirement as
unmet — that is expected and safe to ignore; mediapipe's hand/pose landmarker
tasks use only plain `cv2`. This is not a corner case: skipping this step is
the single most likely way to get a flaky, hard-to-reproduce test suite here.

### The mediapipe version pin

`pyproject.toml` pins `mediapipe>=0.10,<1.0` deliberately. mediapipe 1.0.0 and
1.0.1 fatally abort on macOS/arm64 ("Check failed: service_ Service is
unavailable" inside `TensorsToDetectionsCalculator`'s Metal helper) when
constructing a `HandLandmarker` via the Tasks API — a native crash, not
anything in this project's code. mediapipe 0.10.35 creates both landmarkers
cleanly on CPU. Do not relax this pin without confirming upstream has fixed the
regression.

### `wholebody` does not install — by design, not a bug

The `[wholebody]` extra (mmpose/mmcv/mmdet, for RTMW) will fail to install on
a modern toolchain: `mmcv`'s legacy `setup.py` imports `pkg_resources` during
its build-requirements hook, and current `setuptools` no longer bundles
`pkg_resources` for isolated builds, so pip fails with
`ModuleNotFoundError: No module named 'pkg_resources'` before it can even
resolve versions. This is expected. Every other backend works without it —
`og-annotate list` will simply show `wholebody` as `UNAVAILABLE` with that
reason, and `run`/`report` skip it cleanly.

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

`og-annotate` (the console entry point) terminates via `os._exit` after flushing
stdout/stderr, deliberately skipping normal interpreter finalisation. This is
intentional, not a bug: MediaPipe's and torch's native static destructors race
at teardown and can abort the process even after all work has completed and
every sidecar file has been written. `main()` itself stays pure and returns a
plain `int`, so it remains normally testable; only the console script bypasses
finalisation.

## Backends

| Key | Model | Sidecar | Licence |
|---|---|---|---|
| `face` | uniface — bbox, landmarks, head pose, gaze, emotion, demographics | `faces.json` | MIT (weights vary) |
| `aus` | py-feat — 20 FACS Action Units, emotion, 3D head pose | `action_units.json` | MIT |
| `va` | HSEmotion — continuous valence/arousal | `valence_arousal.json` | Apache-2.0 |
| `pose` | MediaPipe — 21x2 hand + 33 body landmarks | `pose.json` | Apache-2.0 |
| `embed` | CLIP ViT-B-32 — image and text embeddings | `embeddings.json` | Apache-2.0 |
| `wholebody` | RTMW — 133 COCO-WholeBody keypoints (optional, unavailable by default) | `wholebody.json` | Apache-2.0 |

## Weight licences are not library licences

The table above states each backend *library's* licence, and every one is
permissive. That is not the whole story: a library's licence does not
automatically cover the pretrained weights it loads, and this pipeline's own
provenance checks found four places where the two diverge. Given this repo
promises commercial use, this is the single most consequential thing a
downstream user needs to know before shipping anything built on these
annotations:

- **uniface's FairFace weights are CC BY 4.0** — attribution is required, even
  though uniface itself is MIT.
- **py-feat's `mobilefacenet` landmark weights** (from the upstream
  `cunjian/pytorch_face_landmark` repo) have **no LICENSE file at all**,
  despite py-feat itself being MIT. Treat as all-rights-reserved until the
  upstream author clarifies.
- **open_clip's `ViT-B-32`/`laion2b_s34b_b79k` checkpoint is MIT**, not
  Apache-2.0 as is commonly assumed by association with the Apache-2.0
  `open_clip_torch` library.
- **RTMW's checkpoint licence is unspecified.** mmpose's code is Apache-2.0,
  but the RTMW "cocktail14" checkpoint was trained partly on datasets
  published under research-only/non-commercial licences (e.g. AI Challenger,
  InterHand2.6M, Human-Art), and OpenMMLab publishes no separate licence
  statement for the resulting weights.

`annotations/_meta.json` records every weight's licence per backend, as
verified against each upstream source, and `og-annotate run` prints a
`LICENCE:` warning to stderr for any weight that is not known-permissive.
Read `_meta.json` before using any of these weights' outputs commercially —
do not assume a backend's library licence tells you anything about its
weights.

## Output

`annotations/quality_report.md` is the point of all this: it flags where curated
metadata and model predictions disagree. It reports, it never corrects.
`annotations/_meta.json` records every model version and weight licence, and warns
about any that is not permissive.

From the real run over all 99 gestures: `quality_report.md` currently records
49 findings. Two results stand out:

- **`body_parts` produced zero findings.** MediaPipe's hand detection agreed
  with all 96 curated hand claims checked in this run — 100% recall on the
  one direction the check covers (a curated hand claim with no hand detected).
  Zero findings here is a genuine, verified result, not a sign the check
  didn't run.
- **Facial affect is a weak proxy for gesture affect.** 78 of 93 detected
  faces (84%) read as Neutral, while the curated `emotional_state` label is
  `neutral` for only 49 of 99 gestures. In this largely hand-first, close-up
  dataset, the face frequently doesn't carry the emotion the gesture does —
  which is itself evidence for the project's underlying thesis that gesture
  carries meaning the face alone does not.

## Testing

```bash
.venv/bin/python -m pytest              # fast tests, no model weights
.venv/bin/python -m pytest -m slow      # real inference against real weights
```

## Design

See `../docs/superpowers/specs/2026-08-25-multimodal-annotation-pipeline-design.md`.
