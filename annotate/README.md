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

Two optional extras are not installed by default. `relations` needs Python
>=3.12, installs `relsgg` from git, and carries DINOv3-licensed weights. `speech`
needs `embed` too, since it reuses the CLIP text tower. Both are described
under Usage below:

```bash
.venv/bin/python -m pip install -e ".[relations,speech]"
# relations pulls opencv-python-headless: repeat the opencv cleanup below afterwards
```

MediaPipe needs two task bundles downloaded into `.models/`. The URLs below are
copied verbatim from `MODEL_URLS` in `open_gesture_annotate/backends/pose_mediapipe.py`
— one of them was corrected during implementation (see that module's docstring),
so copy from source rather than retyping from memory if it ever changes:

```bash
mkdir -p .models
curl -L -o .models/hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
curl -L -o .models/pose_landmarker_lite.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
```

The `[wholebody]` extra (RTMW/mmpose) is optional and pins hard. If it fails to
install, every other backend still works.

### Where the other backends cache their weights

Unlike `pose`, the other backends download their weights automatically on
first use, via each library's own cache directory — no manual step needed, but
worth knowing about before a fresh install, both for disk budgeting and
because these are outside `.venv/` and outside this repo:

| Backend | Cache location | Approx. size |
|---|---|---|
| `face` (uniface) | `~/.uniface/models/` | ~240 MB |
| `aus` (py-feat) | inside the installed `feat` package (`site-packages/feat/resources/`) | ~850 MB |
| `va` (hsemotion) | `~/.hsemotion/` | ~15 MB |
| `embed` (open_clip) | `~/.cache/huggingface/hub/` (HF Hub) | ~580 MB |
| `pose` (mediapipe) | `.models/` (this repo, manual download above) | ~10 MB |
| `relations` (relsgg, optional) | `~/.cache/huggingface/hub/` (HF Hub) | ~310 MB |
| `speech` (optional): SpeechT5 1.1 GB, Whisper `base.en` 140 MB, DistilBERT 260 MB, HiFi-GAN 100 MB, x-vectors 18 MB | `~/.cache/huggingface/hub/` (HF Hub) | ~1.6 GB |

Budget roughly 1.7 GB of disk outside the repo for a first run of the six core
backends, or about 3.6 GB with `relations` and `speech` as well.

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
.venv/bin/og-annotate fusion                      # image + speech: fusion_report.md
```

Runs are idempotent and resumable: successful records are skipped, previous errors
are retried. A backend that will not import is skipped with a reason, never fatally.

### `speech` — the voice channel

The rest of the pipeline is image-only, but the project's premise is that
machines need more than one channel, and that voice is the one that fails in
noise. `speech` adds it so that `og-annotate fusion` can measure whether image +
speech beats either alone, and whether fusion fails safe as speech degrades.

There is no audio in this dataset, so the speech is **synthesised**. Each gesture
has one utterance in `open_gesture_annotate/utterances.json` — what a person
might say while making it (two are performed silently). The utterances were
written from gesture names *without reading the curated `intent` text*, so the
speech channel is not a paraphrase of the label it is scored against, and
`fusion_report.md` measures the residual word overlap. Per gesture the backend
synthesises the utterance once, mixes in white noise at clean / +10 / 0 / −5 dB
SNR, and transcribes each mix with Whisper — real STT on real audio, so
recognition errors propagate the way they would live. Each transcript is then
embedded with the **same** CLIP text tower `embed` uses for the intents, and
scored for sentiment as a signed valence on HSEmotion's [−1, 1] scale.

`og-annotate fusion` reads `embeddings.json`, `valence_arousal.json` and
`speech.json` and writes `fusion_report.md`. Fusion is late and parameter-free:
each query's scores are z-scored per channel and averaged, because image→text
and text→text cosines sit on different scales. A silent gesture, or a transcript
lost to noise, falls back to image-only. No weight is fitted, since tuning one on
99 gestures would report the tuning rather than the method. A shuffled-transcript
control shows whether any gain comes from *matched* speech.

**Licensing.** Piper, the usual offline TTS, is GPL-3.0-or-later and phonemises
with espeak-ng (also GPL), so it is excluded, just as the spec excluded AGPL
Ultralytics. SpeechT5 works from characters, needs no phonemiser, and is MIT.
Every weight in the chain is MIT or Apache-2.0: SpeechT5 + HiFi-GAN, CMU ARCTIC
x-vectors, Whisper `base.en` (faster-whisper) and DistilBERT SST-2.

Install with `.[embed,speech]`. First run downloads roughly 1.6 GB of weights into
`~/.cache/huggingface/`. TTS is autoregressive on CPU, so a full run takes tens
of minutes; transcription is well under a second per clip.

**Results** (`annotations/fusion_report.md`, 99 gestures, 97 with speech). Every
comparison below is a paired exact sign test on the same gestures, because with
99 gestures a few points is only a few gestures.

- **Speech adds real information to the image.** With clean or +10 dB speech,
  fusion finds the right `intent` in the top 5 for 15 gestures the image alone
  missed, and loses 4 (p = 0.02). Top-5 accuracy goes from 23% to 34%, against
  a chance rate of 5%.
- **It fails safe under noise.** At −5 dB Whisper gets 74% of words wrong, and
  fusion becomes statistically indistinguishable from image-only (9 gained,
  8 lost) — no better, but not worse. Garbled transcripts become semantically
  diffuse and stop pointing anywhere in particular.
- **The danger is fluent speech about something else, not noise.** Pairing each
  image with another gesture's transcript loses top-5 hits against image-only at
  every noise level (significant at 0 dB, a trend elsewhere). That's the
  bystander case, and no STT confidence can catch it, because Whisper is rightly
  confident about clearly heard speech.
- **Weighting speech by Whisper's own confidence measured no better than equal
  weighting** at any noise level (p ≥ 0.35). The confidence does track errors
  (Spearman −0.53 against WER), but it's also low for short correct answers like
  "Ten." (about 0.4), so it underweights them. Both variants are reported.
- **Emotion:** fused face + speech valence separates curated-positive from
  curated-negative gestures at AUC 0.94, against 0.74 for the face alone. The
  speech share of that is inflated by authorship — each utterance was written
  knowing its gesture — so it shows sentiment surviving TTS → noise → STT, not
  how often real speech matches gesture affect.

**Measurement note.** Whisper writes numbers as digits ("Ten." → "10."). WER
spells digits out before scoring; before that fix, 11 of the 20 "errors" on
clean speech were formatting, and clean WER read 16% instead of 8%. Transcripts
themselves are left exactly as Whisper produced them, since the fusion has to
face the digits too.

### `relations` depends on `pose` — run order matters

Same shape as the `va`/`face` dependency below, for the same reason. The
RelateAnything model scores relations between regions *you* supply; it detects
nothing itself. Those regions come from the MediaPipe landmarks in `pose.json`
— hand boxes from the 21 hand landmarks, and head, torso and arm boxes from the
visible subset of the 33 body landmarks. Handing it whole-image boxes instead
would yield confident triplets about nothing, so it reports itself
**unavailable** when `pose.json` is absent rather than degrading silently:

```
relations  unavailable: requires annotations/pose.json — run the pose backend first:
                        og-annotate run --backends pose
```

Plain `og-annotate run` handles this (backends run in sorted order and `pose`
precedes `relations`); an explicit `--backends relations` on a fresh checkout
will not. `--out` is honoured the same way `va` honours it.

Read the triplets with one constraint in mind: relsgg returns **one predicate
per region pair** — the argmax over the supplied vocabulary — so `relations.json`
holds the model's best single answer for each pair, not a ranked set of
everything true of it. The vocabulary in `GESTURE_PREDICATES` is picked around
that: strings that would be true *simultaneously* with contact (`near`,
`in front of`) are left out, because they would take the slot without
contradicting contact, while strings that are a real alternative claim
(`beside`, `above`) are kept, because the model preferring one of those *is* the
answer the `self_contact` check wants.

A gesture that yields fewer than two regions — a hand-first crop with no visible
body, of which this dataset has several — records a successful empty result, not
an error. One region cannot form a pair.

Installing this extra needs care: `relsgg` requires Python >=3.12, is not on
PyPI (the extra installs it from git), and depends on `opencv-python-headless`,
which lands in the same `cv2/` directory as this project's `opencv-python` —
the identical conflict the `pose` extra causes with `opencv-contrib-python`.

Verified on Linux/aarch64, Python 3.12: installing it into the same venv as the
other five backends works, provided you repeat the one-distribution cleanup
afterwards, which leaves all six backends available at once:

```bash
uv pip install -e ".[relations]"
uv pip uninstall opencv-python-headless opencv-python
uv pip install --no-deps --reinstall "opencv-python>=4.8"
```

A separate environment also works and sidesteps the conflict entirely, at one
cost worth knowing: `_meta.json` records the availability of every backend *as
seen by the environment that wrote it*, so a split install makes it report
`relations` unavailable even when `relations.json` was just produced. Running
everything from one environment keeps that record truthful.

### `va` depends on `face` — run order matters

`va` (HSEmotion) crops each face before inference, using the bounding boxes
`face` writes to `faces.json`. Feeding it a whole gesture photo instead would
resize the entire frame to the model input and yield near-meaningless
valence/arousal — and those values are exactly what `report.py` checks the
curated `emotional_state` and `arousal` labels against.

So `va` reports itself **unavailable** when `faces.json` is absent, rather than
silently degrading:

```
va   unavailable: requires annotations/faces.json — run the face backend first:
                  og-annotate run --backends face
```

Plain `og-annotate run` handles this on its own (backends run in sorted order, and
`face` precedes `va`), but an explicit `--backends va` on a fresh checkout will not.
Run `face` first, or just let `run` do everything.

This is a *whole-file* check. A `faces.json` that exists but records `face_count: 0`
for a given gesture is fine — that image falls back to full-image inference and is
still annotated successfully. Six gestures are framed hand-first with no face at all
and take that path every run.

`--out` is honoured throughout: `va` reads `faces.json` from the same directory the
run is writing to, not from the repo default, so a run into a scratch directory never
silently crops using a previous run's boxes.

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
| `relations` | RelateAnything (relsgg) — open-vocabulary relations between body regions (optional, not installed by default) | `relations.json` | Apache-2.0 (weights: **DINOv3 License**) |
| `speech` | SpeechT5 TTS → noise → Whisper STT; transcript CLIP embedding and valence (optional) | `speech.json` | MIT / Apache-2.0 (every weight) |

## Weight licences are not library licences

The table above states each backend *library's* licence, and every one is
permissive. That is not the whole story: a library's licence does not
automatically cover the pretrained weights it loads, and this pipeline's own
provenance checks found five places where the two diverge. Given this repo
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
- **RelateAnything's checkpoints are under the DINOv3 License**, not the
  Apache-2.0 of the `relsgg` library. They are derivatives of Meta's DINOv3
  pretrained weights (vision backbone *and* the distilled predicate text
  student), and RelateAnything's own `THIRD_PARTY_NOTICES.md` says so
  explicitly. These are the strictest weight terms in this pipeline, which is
  why `relations` is optional and off by default.
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
about any that is not permissive. `annotations/fusion_report.md` (`og-annotate
fusion`) is the voice-channel counterpart: whether image + speech beats either
alone, and how that holds up as speech degrades. Its results are summarised
under `speech` in Usage above.

From the real run over all 99 gestures: `quality_report.md` currently records
49 findings across six checks. Three results stand out:

- **`body_parts` produced zero findings.** MediaPipe's hand detection agreed
  with all 96 curated hand claims checked in this run — 100% recall on the
  one direction the check covers (a curated hand claim with no hand detected).
  Zero findings here is a genuine, verified result, not a sign the check
  didn't run.
- **`self_contact` produced zero findings too.** All five curated `adaptor`
  gestures predict `right hand -> touching -> head`, which is correct for every
  one of them (Facepalm, D'oh Facepalm, Tensed Fingers, Hand Visor, Hands Over
  Mouth). Ranking all 99 gestures by their strongest hand-to-body contact score
  puts hand-to-face gestures at the top — Nose Thumb, Facepalm, Zipped Lips,
  Hat Tip, Salute, Cuckoo Sign, Finger to Lips — across five different curated
  `gesture_type`s, so the score is a real signal even though its absolute scale
  is not (see below).
- **Score calibration does not survive a vocabulary swap.** This one cost a
  wrong answer before it was caught. The checkpoint ships a Platt map that makes
  a score read as a probability, but that fit belongs to the released
  vocabulary; against the custom one used here the median triplet score is
  0.027. An initial 0.5 threshold called 3 of the 5 adaptors non-contact — all
  three wrongly. The check now uses the model's per-pair argmax with no fitted
  constant, and records `top_contact_score` so strength stays inspectable.
  Fitting a threshold to the adaptor labels was rejected deliberately: it would
  make the check agree with the labels it exists to question.
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

The fast layer needs no weights and no network. It covers the registry, runner,
sidecar I/O, report and fusion logic, the relations region geometry, and the
speech helpers (noise at a target SNR, WER, confidence), all on synthetic data.

The slow layer adds three things:

- **Backend smoke and golden-file tests** (`test_backend_*`, `test_golden.py`)
  run the real models on three representative images. Golden files compare
  numbers within a tolerance and every categorical output (emotion labels,
  handedness, ...) exactly. **Known:** on Linux/aarch64 three golden cases
  (`aus` and `face` on the clear-face and multi-person images) fail on
  sub-pixel landmark drift of up to 0.13 px, which carries into py-feat AUs
  (up to 0.026). Every categorical output still matches. The golden files were
  generated on a different platform and library stack. They are deliberately
  left unregenerated rather than tuned to one machine.
- **`test_recognition.py`**, an acceptance test for the spec's intent and
  emotion claims, run against the real sidecars (it skips until `og-annotate
  run` has produced them). It checks conformance (every face has an emotion
  label, HSEmotion crops from uniface's box, the 20 planned AUs) and
  recognition beyond chance (image-to-text retrieval against a binomial null
  with a shuffled control, valence ordering and separation, arousal
  correlation, agreement between py-feat and HSEmotion by kappa rather than
  raw agreement). Thresholds are "better than chance" at conventional alphas,
  set up front; a mutation check confirmed each test fails on deliberately
  broken data.
- Measured on the current sidecars: image-to-intent top-5 retrieval 23% and
  image-to-description 37% (chance 5%); valence separates curated positive
  from negative at AUC 0.74 (p = 0.003); arousal Spearman +0.26 (p = 0.01);
  uniface and HSEmotion labels agree at kappa 0.65.

## Design

See `../docs/superpowers/specs/2026-08-25-multimodal-annotation-pipeline-design.md`
for the original six-backend design. Its "Amendments" section records what came
after: the `relations` and `speech` backends, the fusion evaluation, and the
places where the implementation deliberately differs from the spec. The
rationale for each later backend lives in its module docstring
(`backends/relations_relsgg.py`, `backends/speech_whisper.py`, `fusion.py`).
