# Multimodal Annotation Pipeline — Design

**Date:** 2026-08-25
**Status:** Approved design, pending implementation plan
**Repo:** open-gesture

## Context

open-gesture is currently a pure dataset: 113 images across 21 category folders,
plus `manifest.json` (99 gestures, hand-curated metadata) and `manifest.md`.
There is no code.

The metadata schema asserts claims about each gesture — `emotional_state`,
`arousal`, `number_of_people`, `body_parts` — that no automated process has ever
checked. The README argues that machines need fluency across modalities, but the
repo ships no means of extracting any modality.

This design adds a pluggable annotation pipeline that runs face, body, and
semantic models over the image set and writes the results to sidecar files. Its
purpose is twofold: enrich the dataset with machine-derived signals, and turn the
curated metadata from *asserted* into *empirically cross-checked*.

## Goals

1. Extract per-image face, hand, body, and semantic annotations using permissively
   licensed models.
2. Write results to sidecar files, leaving `manifest.json` untouched as curated
   ground truth.
3. Produce a quality report flagging disagreements between curated labels and
   model predictions.
4. Keep every backend independently installable, independently testable, and
   non-fatal on failure.
5. Record model provenance — name, version, weight license — so the repo's
   commercial-use promise stays truthful.

## Non-goals

- Training any model. This pipeline is inference-only.
- Video or temporal gesture recognition. The dataset is static images.
  (Skeleton-based action recognition via MMAction2 is a future concern.)
- Modifying `manifest.json`, `manifest.md`, or any image.
- Bundling third-party datasets (HaGRID is CC-BY-SA-4.0; ShareAlike is
  incompatible with this repo's CC-BY-4.0 images).

## Architecture

```
open-gesture/
├── manifest.json               # UNTOUCHED — curated ground truth
├── manifest.md                 # UNTOUCHED
├── gesture_images/             # UNTOUCHED
├── annotations/                # NEW — machine output
│   ├── faces.json              # uniface
│   ├── action_units.json      # py-feat AUs, emotion, 3D head pose
│   ├── valence_arousal.json   # HSEmotion continuous valence/arousal
│   ├── pose.json               # MediaPipe hands + body
│   ├── wholebody.json          # RTMW 133 kpts  [optional extra]
│   ├── embeddings.npz          # CLIP/SigLIP vectors
│   ├── _meta.json              # model names, versions, weight SHAs, licenses, run date
│   └── quality_report.md       # curated-vs-predicted disagreements
└── pipelines/                  # NEW — the code
    ├── pyproject.toml
    ├── README.md
    ├── open_gesture_annotate/
    │   ├── __init__.py
    │   ├── base.py             # Backend protocol
    │   ├── registry.py         # backend discovery + availability
    │   ├── schema.py           # annotation record types + JSON Schema
    │   ├── io.py               # manifest loading, sidecar read/write, resume
    │   ├── report.py           # cross-check logic
    │   ├── cli.py              # og-annotate entry point
    │   └── backends/
    │       ├── face_uniface.py
    │       ├── affect_pyfeat.py
    │       ├── affect_hsemotion.py
    │       ├── pose_mediapipe.py
    │       ├── wholebody_rtmw.py
    │       └── embed_clip.py
    └── tests/
```

### Backend protocol

Every backend implements one small interface and nothing else. This is the only
contract the registry, CLI, and report logic depend on.

```python
class Backend(Protocol):
    name: str        # stable key, e.g. "uniface-face"
    version: str     # library version, recorded in _meta.json
    sidecar: str     # which file it writes, e.g. "faces.json"

    def available(self) -> tuple[bool, str]:
        """(importable and weights resolvable?, human-readable reason)"""

    def provenance(self) -> dict:
        """Model names, weight files, SHA256, and license of each weight."""

    def annotate(self, image: np.ndarray, gesture: dict) -> dict:
        """Annotate one image. Raises on failure; caller isolates."""
```

A backend is ~100 lines. It may be understood, tested, and replaced without
reading any other backend.

**One sidecar per backend.** Each backend owns exactly one output file and is the
only writer of it. This keeps `_backend` provenance unambiguous, lets any single
backend be re-run without rewriting another's output, and means a corrupt or
partial file implicates exactly one model. Backends that cover the same modality
(py-feat and HSEmotion both describe facial affect) still write separate files;
`report.py` joins them by gesture `id`.

| Backend | Sidecar |
|---|---|
| `uniface-face` | `faces.json` |
| `pyfeat-au` | `action_units.json` |
| `hsemotion-va` | `valence_arousal.json` |
| `mediapipe-pose` | `pose.json` |
| `rtmw-wholebody` | `wholebody.json` |
| `clip-embed` | `embeddings.npz` |

## Backends

APIs below were verified against upstream source on 2026-08-25.

### 1. `uniface-face` — face geometry, pose, gaze, emotion, demographics

- **Sidecar:** `faces.json`
- **Package:** `uniface` 4.0.0, MIT. Requires Python `>=3.10,<3.15`.
- **Install:** `pip install "uniface[cpu]"` (CPU and Apple Silicon).
- **API:** `FaceAnalyzer(predictors=[...])`; `analyzer.analyze(img)` yields `Face`
  objects. `bbox`, `confidence`, `landmarks`, `embedding` are always set;
  attribute fields are `None` unless the relevant predictor is passed.
- **Predictors used:** detection (RetinaFace/SCRFD), 106-pt landmarks, head pose
  (pitch/yaw/roll), MobileGaze, AffectNet emotion, FairFace demographics.
- **Caveat:** uniface's own README warns that some pretrained weights are not MIT.
  `provenance()` MUST record the license of each weight actually downloaded.

### 2. `pyfeat-au` — Facial Action Units

- **Sidecar:** `action_units.json`
- **Package:** `py-feat`, MIT (repo license file is MIT despite GitHub's
  NOASSERTION tag). Active as of 2026-08-04.
- **Install:** `pip install py-feat`.
- **API:** `Detectorv1(face_model="retinaface", landmark_model="mobilefacenet",
  au_model="xgb", emotion_model="resmasknet", identity_model=None, device="cpu")`,
  then `.detect(inputs, data_type="image", batch_size=..., progress_bar=False)`
  returning a `Fex` DataFrame.
- **Outputs consumed:** the 20 `Feat` Action Units — AU01, AU02, AU04, AU05, AU06,
  AU07, AU09, AU10, AU11, AU12, AU14, AU15, AU17, AU20, AU23, AU24, AU25, AU26,
  AU28, AU43 — plus `FEAT_EMOTION_COLUMNS` (anger, disgust, fear, happiness,
  sadness, surprise, neutral) and `FEAT_FACEPOSE_COLUMNS_3D` (Pitch, Roll, Yaw).
- **Why it matters:** the README cites Ekman & Friesen (1969). FACS is Ekman's own
  coding system, so AUs give the affect metadata the same anatomical grounding
  that BAP gives `body_parts`.
- **Note:** `identity_model=None` — face identity embeddings are not needed and
  ArcFace weights carry their own license.

### 3. `hsemotion-va` — continuous valence and arousal

- **Sidecar:** `valence_arousal.json`
- **Package:** `hsemotion`, Apache-2.0.
- **Install:** `pip install hsemotion`.
- **API:** `HSEmotionRecognizer(model_name="enet_b0_8_va_mtl", device="cpu")`,
  then `emotion, scores = fer.predict_emotions(face_img, logits=False)`.
- **Valence/arousal extraction:** undocumented but present. Source
  (`hsemotion/facial_emotions.py`) computes emotion softmax over `scores[:-2]`,
  which means the `va_mtl` multi-task model appends **valence and arousal as the
  final two elements** of `scores`. Read them as `scores[-2:]`.
  The implementation MUST assert `len(scores) == 10` for the 8-class `va_mtl`
  model so a silent upstream layout change fails loudly rather than writing
  garbage into the annotations.
- **Input:** requires a cropped face image. Reuse the bbox from `uniface-face`
  rather than running a second detector.
- **Why it matters:** produces the continuous valence/arousal pair that the
  README's PAD-model framing (Russell & Mehrabian) actually calls for, against
  which the categorical `emotional_state` and `arousal` fields can be checked.

### 4. `mediapipe-pose` — hands and body

- **Sidecar:** `pose.json`
- **Package:** `mediapipe`, Apache-2.0.
- **Tasks:** Hand Landmarker (21 landmarks × up to 2 hands, plus handedness) and
  Pose Landmarker (33 landmarks, 3D world coordinates).
- **Why it matters:** this is the largest gap. The `body_parts` field is
  overwhelmingly hand- and finger-valued, and no face model can see it.

### 5. `rtmw-wholebody` — 133-keypoint whole body *(optional extra)*

- **Sidecar:** `wholebody.json`
- **Package:** `mmpose` / RTMW, Apache-2.0.
- **Output:** COCO-WholeBody 133 keypoints — body 17, feet 6, face 68, hands 42.
- **Risk:** the mmcv/mmpose stack pins hard and installs badly. Isolated behind
  the `[wholebody]` extra. A failed install of this extra MUST NOT affect any
  other backend.

### 6. `clip-embed` — semantic embeddings

- **Sidecar:** `embeddings.npz`
- **Package:** `open_clip_torch` or `transformers` SigLIP. Apache-2.0 / MIT.
- **Output:** per gesture, an image embedding plus text embeddings of `intent` and
  `physical_description`, written to `embeddings.npz`.
- **Why it matters:** implements the embedding space described in README §4
  (valence-arousal plane, semantic similarity, taxonomic grouping), and yields an
  image↔text similarity score that surfaces mislabeled or weak images.

## Annotation schema

Each sidecar is a JSON object keyed by gesture `id` (e.g. `"affirm-01"`), so
records join to `manifest.json` without positional coupling.

```json
{
  "_backend": {"name": "uniface-face", "version": "4.0.0", "run_at": "..."},
  "records": {
    "affirm-01": {
      "status": "ok",
      "faces": [
        {
          "bbox": [x1, y1, x2, y2],
          "confidence": 0.99,
          "landmarks": [[x, y], ...],
          "head_pose": {"pitch": 0.0, "yaw": -3.2, "roll": 1.1},
          "gaze": {"pitch": 0.0, "yaw": 0.0},
          "emotion": {"label": "neutral", "scores": {}},
          "demographics": {"age_group": "...", "sex": "..."}
        }
      ]
    },
    "some-id": {"status": "error", "error": "no face detected"}
  }
}
```

`status` is always present and is either `"ok"` or `"error"`. Consumers branch on
it; a failed image is a recorded outcome, never a missing key.

`_meta.json` aggregates provenance across all backends: library versions, weight
filenames, SHA256 of each weight file, the license of each weight, and the run
timestamp.

## Quality report

`report.py` joins curated metadata against predictions and writes
`annotations/quality_report.md`. This is the deliverable that earns the pipeline.

| Curated field | Checked against | Flag when |
|---|---|---|
| `number_of_people` (94 single / 4 two / 1 three+) | uniface face count | count disagrees with the stated bucket |
| `emotional_state` (49 neutral / 31 positive / 19 negative) | HSEmotion valence sign; py-feat emotion | valence sign contradicts the label |
| `arousal` (low/medium/high) | HSEmotion arousal magnitude | stated bucket falls outside the predicted tercile |
| `body_parts` | MediaPipe hand/pose presence | claims `hand`/`arm` but none detected |
| `intent`, `physical_description` | CLIP image↔text similarity | similarity in the bottom decile |

The report ranks disagreements by severity and links each to its image path. It
asserts nothing about which side is wrong — a disagreement is a prompt for human
review, not an automated correction. `manifest.json` is never rewritten by this
pipeline.

## Failure isolation

The backend set spans a wide range of install risk, so isolation is a first-class
requirement rather than defensive polish.

- **Import failure** → `available()` returns `(False, reason)`. The backend is
  skipped and listed as unavailable in the run summary. Not fatal.
- **Per-image failure** → recorded as `{"status": "error", "error": ...}` in that
  backend's sidecar. The run continues to the next image.
- **Optional extras** → `[face] [affect] [pose] [embed] [wholebody]`, so RTMW's
  dependency stack cannot break the rest.
- **Resumability** → runs are idempotent. Existing `ok` records are skipped unless
  `--force`. A crashed run is resumed by re-invoking the same command.

## CLI

```
og-annotate list                                  # backends and availability
og-annotate run --backends face,aus,va,pose       # default: all available
og-annotate run --force --limit 5                 # re-run a subset
og-annotate report                                # regenerate quality_report.md
```

## Licensing constraints

The repo promises MIT code / CC-BY-4.0 images with commercial use permitted.
Every dependency was license-checked against that promise:

**Permitted:** uniface (MIT), py-feat (MIT), hsemotion (Apache-2.0), mediapipe
(Apache-2.0), mmpose/RTMW (Apache-2.0), open_clip / SigLIP (Apache-2.0 / MIT).

**Excluded and why:**

| Project | License | Verdict |
|---|---|---|
| OpenPose | CMU non-commercial | Excluded |
| Ultralytics YOLO-Pose | AGPL-3.0 | Excluded |
| Sapiens (Meta) | CC BY-NC-4.0 (verified) | Excluded |
| LibreFace (USC) | Research / non-profit only | Excluded |
| OpenFace | Academic non-commercial | Excluded |
| HaGRID | CC-BY-SA-4.0 | Not bundled; benchmark reference only |

**Required README fix:** the README currently recommends OpenPose in the
"Hierarchical Recognition Pipeline" section while the license table promises
commercial use. These contradict. Replace the OpenPose reference with MediaPipe
and MMPose as part of this work.

Pretrained weights are licensed separately from the libraries that load them.
`_meta.json` records the license of every weight actually downloaded, and the run
summary warns when a non-permissive weight is pulled.

## Testing

Test-driven, with the fast layer fully decoupled from model weights.

1. **Fake backend** exercising the registry, resume logic, sidecar I/O, error
   recording, and the report generator. No weights, no network, runs in CI.
2. **Schema validation** — every sidecar validates against the JSON Schema in
   `schema.py`; `status` present on every record.
3. **Report logic** — table-driven tests over synthetic curated/predicted pairs,
   including each disagreement class above.
4. **Golden-file tests** on 3 representative images (one single-person clear-face,
   one multi-person, one hands-only/face-occluded) with tolerance-based numeric
   comparison.
5. **Real-model smoke tests** marked `@pytest.mark.slow`, excluded from default runs.
6. **HSEmotion layout assertion** — an explicit test that the `va_mtl` score vector
   has the expected length, since valence/arousal extraction depends on an
   undocumented output layout.

## Environment

- Python `>=3.10,<3.15` (uniface's constraint, the tightest of the set).
- CPU-only by default; Apple Silicon supported via `uniface[cpu]`.
- **Known blocker:** the machine's pyenv 3.13.2 has a broken `hashlib` —
  `blake2b` and `blake2s` raise `unsupported hash type`, which breaks `pip`.
  Implementation begins by provisioning a working interpreter and a dedicated
  venv under `pipelines/`. This is a prerequisite task, not an incidental fix.
- `annotations/*.npz` and large sidecars are gitignored by default; `_meta.json`
  and `quality_report.md` are committed, since they are the reviewable artifacts.

## Future work (explicitly out of scope)

- MMAction2 skeleton-based recognition, once a video corpus exists.
- Benchmarking against HaGRID's 33 gesture classes.
- Aligning `emotional_state`/`arousal` scales to EMOTIC and BoLD VAD annotations.
- Emotion-LLaMA (BSD-3) for generated `emotional_adjectives` and `intent` prose.
