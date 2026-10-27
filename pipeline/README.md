# Synthetic Data Pipeline

Generates training data for an on-device gesture classifier from the Open
Gesture manifest. The target runtime is **landmark-sequence classification**:
at inference time, `RGB → MediaPipe Holistic → landmarks → temporal classifier`.
The pipeline produces synthetic landmark sequences with matching labels so the
classifier never needs real labeled footage.

## Stage order

```
manifest.json (99 gestures × metadata)
      │
      ├─► export_labels.py ──► labels/label_schema.json   (shared class↔index contract)
      │                        labels/labels.jsonl        (per-gesture labels)
      │
      └─► generate_motion_specs.py ──► motion_specs/<id>.json   (temporal stubs to author)
                                            │
                                            ▼
                              BlenderProc [Mac Studio]
                              SMPL-X avatars perform each motion spec.
                              Randomize: lighting, background, distance,
                              angle, occlusion. Output: RGB video.
                                            │
                                            ▼
                              Cosmos Transfer [RunPod burst]
                              Photorealistic augmentation of the renders
                              (outdoor / industrial / AR-context variants).
                                            │
                                            ▼
                              MediaPipe Holistic [Mac Studio]
                              Extract landmark sequences — SAME path as real
                              inference. Attach labels via label_schema.json.
                                            │
                                            ▼
                              TFLite INT8 multi-head classifier
                              → compile to Hailo HEF (Dataflow Compiler)
```

## Why Cosmos runs *before* MediaPipe

The original sketch put MediaPipe before Cosmos. That wastes the augmentation:
the classifier trains on **landmarks**, which are largely appearance-invariant,
so photorealism applied *after* extraction never reaches the model.

Where photorealism actually matters is **MediaPipe's own reliability** —
MediaPipe Holistic was trained on real footage and can emit garbage landmarks on
raw Blender renders (plasticky skin, uncanny hands). Running Cosmos first closes
the sim-to-real gap at the point where it bites: landmark *accuracy*. So the
real-world domain variation (outdoor/industrial/AR) is baked in *before*
extraction, and the landmark sequences the classifier sees are both realistic
and correctly detected.

## Open design questions (see repo discussion)

- **Motion authoring is the gating task.** The manifest describes end poses, not
  motion. `motion_specs/*.json` are seeded stubs (heuristic `dynamic` flag,
  empty `smplx_pose` keyframes) — a human or motion-synthesis step must fill the
  SMPL-X poses. 28/99 gestures are flagged dynamic; review the flag before
  authoring.
- **PA, not PAD.** The manifest has valence + arousal but no dominance, so the
  affective heads are two, not three. Add a `dominance` field upstream to get
  the full PAD model.
- **Hailo-8L may be overkill.** A temporal classifier over landmark *vectors*
  (1D-CNN / TCN / GRU) is tiny — the runtime cost is MediaPipe on the host CPU,
  not the classifier. Confirm Hailo is buying anything before targeting it. Note
  Hailo runs HEF, not TFLite directly: TFLite/ONNX → Dataflow Compiler → HEF.

## Tools

| Script | Input | Output | Deps |
|--------|-------|--------|------|
| `export_labels.py` | `manifest.json` | `labels/` | stdlib |
| `generate_motion_specs.py` | `manifest.json` | `motion_specs/` | stdlib |
| `generate_render_configs.py` | `motion_specs/` | `render_configs/` | stdlib |
| `build_hand_presets.py` | `manifest.json` | `hand_presets/` | stdlib |
| `smplx_capture.py` | posed rig | preset / keyframe | Blender (add-on) |
| `import_interhand_hand.py` | InterHand MANO | `hand_presets/` | stdlib |
| `import_amass_body.py` | AMASS `.npz` | a motion spec | numpy |
| `apply_hand_presets.py` | `hand_presets/` | `motion_specs/` | stdlib |
| `report_license_tiers.py` | `motion_specs/` | report | stdlib |
| `render_clip.py` | config + spec | one clip | blenderproc (Blender) |
| `run_render_batch.py` | `render_configs/` | clips | stdlib (calls blenderproc) |
| `generate_cosmos_jobs.py` | `render_configs/` | `cosmos_jobs/` | stdlib |
| `run_cosmos_batch.py` | `cosmos_jobs/` | augmented clips | stdlib (calls Cosmos) |
| `landmark_format.py` | — | (imported module) | stdlib |
| `extract_landmarks.py` | clips + `labels/` | `sequences/` | mediapipe, opencv, numpy |
| `train_classifier.py` | `sequences/` + `labels/` | `model/` | tensorflow, numpy |
| `hagrid_hands.py` | HaGRID `annotations.zip` | `model/hagrid_hands.pt` | torch, numpy |
| `import_hagrid.py` | HaGRID images (remote) / curated images | `sequences_hagrid_*/`, `sequences_curated/` | mediapipe, opencv, numpy |
| `eval_curated.py` | `model/` + `sequences_curated/` | score on the 99 curated images | tensorflow, numpy |
| `import_ipn.py` | IPN Hand videos + annotations | `clips_ipn/` (ctrl-06..17, dir-01, none) | opencv |
| `record_gestures.py` | webcam + `manifest.json` | `recordings/` (your own test clips) | opencv, numpy |

`export_labels.py` writes the class↔index contract; re-run after any manifest
change. `generate_motion_specs.py` skips existing files so hand-authored specs
survive (`--force` to regenerate). `landmark_format.py` is the shared per-frame
feature layout (258-D: pose + both hands, face dropped) that the extractor and
model both import — the landmark-space twin of `label_schema.json`.

The first two stages are stdlib-only. The extraction and training stages need
external packages:

```bash
pip install -r requirements.txt   # mediapipe, opencv, tensorflow, numpy

# Stage 0 — derive labels + motion stubs from the manifest (stdlib)
python3 export_labels.py
python3 generate_motion_specs.py

# Stage 0b — domain-randomization sweep: one render config per synthetic clip
python3 generate_render_configs.py --variants 8

# Stage 0b' — source poses (two halves compose: arms from AMASS, fingers from
#   InterHand; or capture either by hand). Both external sources are research-
#   only and get tagged license_tier=non-commercial automatically.
python3 build_hand_presets.py                       # preset vocabulary + suggestions
python3 import_interhand_hand.py --mano MANO.json --capture C --frame F --preset ok_ring
python3 apply_hand_presets.py                        # snap fingers onto keyframes
python3 import_amass_body.py --npz SEQ.npz --spec motion_specs/greet-01.json   # arms
python3 report_license_tiers.py                      # audit commercial/research split

# ... or capture poses by hand on the rig (smplx_capture.py), clean-license ...

# Stage 0c — BlenderProc render (needs blenderproc + an SMPL-X model + GPU)
#   Dry run first: reports how many clips are blocked on unauthored poses.
python3 run_render_batch.py --smplx-model /path/SMPLX_NEUTRAL.npz --out clips/
#   Then render for real:
python3 run_render_batch.py --smplx-model /path/SMPLX_NEUTRAL.npz --out clips/ \
    --execute --skip-existing

# Stage 0d — Cosmos Transfer augmentation (RunPod GPU): photorealistic
#   outdoor/industrial/AR variants, conditioned on the depth+seg BlenderProc
#   emitted. Jobs inherit each clip's license tier; --tier renders one at a time.
python3 generate_cosmos_jobs.py --variants outdoor industrial ar_indoor
python3 run_cosmos_batch.py --out augmented/ --execute --skip-existing \
    --cmd "python cosmos_transfer1/inference.py --spec {spec} --output {out}"

# Stage 1 — MediaPipe extraction (run AFTER Cosmos; same path as real inference)
python3 extract_landmarks.py --clips augmented/

# Stage 2 — train multi-head classifier + export INT8 TFLite
python3 train_classifier.py
#   then: TFLite/ONNX -> Hailo Dataflow Compiler -> .hef
```

Clip filenames must encode the gesture id before a `__` separator, e.g.
`affirm-01__lighting3_angle12.mp4`, so the extractor can join each clip to its
labels.

### Real data: HaGRID v2

[HaGRID v2](https://github.com/hukenovs/hagrid) is 1.09M photos of ~34 static
hand gestures by thousands of people. It directly matches 25 of the 99
gestures (`hagrid_hands.HAGRID_TO_GESTURES`); arm, body, motion and
hand-to-face gestures it does not cover. Its licence is HaGRID's own rework
of CC BY-SA 4.0: commercial use allowed, attribution and share-alike apply,
portrait rights not granted. Per the design spec it is never bundled here --
its annotations live outside the repo and everything derived from it is
gitignored.

Two paths, both scored on the 99 curated images, which neither trains on:

```bash
# Fast: HaGRID's own 2D hand landmarks (719 MB annotation zip, no images)
python3 hagrid_hands.py train --annotations /path/to/annotations.zip
python3 hagrid_hands.py eval

# Full: sampled images -> Holistic -> 258-D shards for train_classifier.py.
#   Images are range-read out of the 119 GB 512px zip; nothing else downloads.
python3 import_hagrid.py --annotations /path/to/annotations.zip --per-class 3000
python3 import_hagrid.py --curated
python3 train_classifier.py --sequences sequences_hagrid_train --batch 128 --mirror --seq-len 32
python3 eval_curated.py --seq-len 32
```

Results (2026-10-20), on the 25 curated gestures HaGRID directly matches.
Groups of gestures that share one hand shape (e.g. the three timeout
variants) count as one answer:

| Model | Training data | HaGRID held-out | Curated |
|---|---|---|---|
| Fast | 2D hand landmarks, ≤20k/class + `no_gesture` | 99.1% (person split) | 24/25 |
| Full | 18,286 images → Holistic, `--mirror` | 92–97% (random split) | 20–22/25 over 3 seeds |
| Full | 56,280 images (~3k/class + 1,378 `none`) | 96% (random split) | 21–22/25 over 2 seeds |
| Fast | as above, 24 classes incl. fist/grabbing/grip/little_finger/two_up (2026-10-26) | 99.0% (person split) | 29/30 |

The 2026-10-26 row covers the 116-gesture manifest: the five new HaGRID
shapes (ctrl-01..05) are all correct, fist at 0.50 confidence. The full
model's 2026-10-26 numbers are in the IPN section below.

The fast model's one miss is Evil Horns read as `one`, which the full
model misses too. With 25 test gestures, one gesture is 4 points: treat
differences of a gesture or two as noise.

Tripling the data and adding `none` did not move the full model, and the
gap to the fast model is mostly that noise: run on Holistic's hand
landmarks instead of HandLandmarker's, the fast model scores 23/25. The
remaining misses are index-finger shapes (point, horns) read as `mute` and
crossed forearms read as praying hands. More signal now needs a bigger
test set, not more HaGRID.

To get that test set, record yourself (gitignored, stays local):

```bash
python3 record_gestures.py --who me --takes 3      # space=record, s=skip, q=quit; resumable
python3 extract_landmarks.py --clips recordings --out sequences_recorded
python3 eval_curated.py --sequences sequences_recorded --all --seq-len 32
```

`--all` scores every recorded gesture, not just HaGRID's 25. Gestures
without training data will miss until the synthetic path covers them;
that is the baseline.

`--seq-len 32` because 56k samples x 64 frames does not fit in 32 GB once
mirrored; a HaGRID sample is one frame tiled, so 32 loses nothing.
`import_hagrid.py` is resumable -- re-run the same command after an
interruption.

The `none` class (export_labels.NONE) is appended to every head. Adding
gestures to the manifest does shift class indices (ids sort), so
`train_classifier.py` reads each shard's labels from its `gesture_id` in the
current `labels/`, never the indices stored at extraction -- old shards stay
usable, but old models must be retrained.

Per-frame CPU cost (one thread, 640px frame): MediaPipe HandLandmarker
23 ms, Holistic 37 ms, full INT8 TFLite classifier 0.04 ms per 64-frame
window. Landmark extraction is the whole cost; the classifiers are noise.

`--mirror` adds left-right mirrored copies. Leave it off once direction-
sensitive gestures (the bicycle turn signals) are in the training set.

`train_classifier.py` normalises every sequence with
`landmark_format.normalize` (body-relative pose, wrist-relative hand shape,
aspect-corrected x). Inference must apply the same function to its frames.

### Real data: IPN Hand (touchless-control motions)

[IPN Hand](https://gibranbenitez.github.io/IPN_Hand/) is 13 touchless-control
gestures (point, click, double click, throw, zoom, open twice) plus
non-gesture segments, by 50 people at 640x480/30 fps. CC BY 4.0: commercial
use allowed with attribution. It covers ctrl-06..17, dir-01 and `none`. Its
official split is by person, so the test set is people never trained on.

Three sets, all split by PERSON. A video name's first two fields are the
person (`1CM42_12_R_#205` -> `1CM42_12`; 50 people, 4 videos each). Tune
on `val` -- 7 of the 37 training people, `random.Random(0).sample` -- and
touch `test` once, at the end:

```bash
# videos (5 tgz, 4.6 GB) + annotations from IPN's Google Drive into .cache/ipn/
python3 - <<'EOF'   # Annot_TrainList -> Annot_TrainSubList + Annot_ValList (7 people)
import random
rows = [l for l in open(".cache/ipn/Annot_TrainList.txt") if l.strip()]
person = lambda l: tuple(l.split(",")[0].split("_")[:2])
val = set(random.Random(0).sample(sorted({person(l) for l in rows}), 7))
open(".cache/ipn/Annot_ValList.txt", "w").writelines(l for l in rows if person(l) in val)
open(".cache/ipn/Annot_TrainSubList.txt", "w").writelines(l for l in rows if person(l) not in val)
EOF
for s in TrainSub:train Val:val Test:test; do
  python3 import_ipn.py --videos .cache/ipn/videos/videos --annotations .cache/ipn/Annot_List.txt \
      --videos-from .cache/ipn/Annot_${s%%:*}List.txt --out clips_ipn_${s##*:}
  python3 extract_landmarks.py --clips clips_ipn_${s##*:} --out sequences_ipn_${s##*:} \
      --workers 6 --min-hand-detections 0.5
done
I=sequences_ipn_train
python3 train_classifier.py --sequences sequences_hagrid_train $I $I $I --val-sequences sequences_ipn_val \
    --batch 128 --seq-len 64 --out model/combined
python3 eval_curated.py --model model/combined/open_gesture.keras --sequences sequences_ipn_test --all --seq-len 64
```

- `--min-hand-detections 0.5`: about 2% of IPN clips have the hand off
  camera or undetected for most of the clip; pose alone would pass the
  plain `--min-detections` check and label a still body as a throw. It
  kept 3,943/4,039 train+val and 1,590/1,610 test clips (worst: throw down, 88%).
- No `--mirror`: mirroring turns throw left into throw right.
- `--val-sequences` keeps the best epoch on the held-out people and halves
  the learning rate when it stalls. Without it, held-out accuracy swings
  by up to 30 points between epochs and the saved last epoch is a lottery.

Results (2026-10-27), one seed each. Tuning, on the 7 validation people
(786 clips), IPN repeated k times next to HaGRID's ~3,000 per shape:

| IPN x | 1 | 3 | 5 | 10 |
|---|---|---|---|---|
| last epoch (no checkpointing) | 60% | 61% | 62% | 43% |
| best epoch on val | 69% | **71%** | 67% | 70% |

Keeping the best epoch is worth ~9 points; oversampling is within noise.
x3 was picked on val, then scored once:

| Model | IPN test (13 people, 1,590 clips) | Curated (30 covered) |
|---|---|---|
| HaGRID 70,784 + IPN x3, best epoch | **73%** (1,164/1,590) | 23/30 |

An earlier 79% for "IPN x10" (2026-10-26) is withdrawn: it was a last-epoch
model, and x10 was chosen after looking at the test score.

Per gesture on test: point (1 or 2 fingers) 87-91%, throw left 88%, zoom
out 78%, `none` 78%, throw down 74%, open twice 70%. Weak: zoom in 23% (read
as zoom out), clicks and double clicks 37-41% (read as each other or as
pointing), throw up 48%, throw right 54%.

Read the 73% as a ceiling, not a real-world figure:

- IPN's test people were filmed in the same rooms with the same cameras as
  its training people. A different webcam, room or distance will score
  lower -- `record_gestures.py` clips are the test that measures that.
- One seed per setting; gaps of a few points are noise.
- Labels are IPN's segment boundaries; our clips inherit any error there.

Not used yet, possibly later: GVHMR (arXiv:2409.06662) recovers 3D body
motion (SMPL) from ordinary video, which could replace hand-authoring the
body keyframes in `motion_specs/` for the dynamic gestures. Body only -- the
fingers would still need a hand model (MANO) -- and its licence and SMPL's
(believed non-commercial) need checking first.

`extract_landmarks.py` runs clips in parallel (`--workers`) and resumes:
a clip whose shard exists is skipped. Each worker writes its own shard,
and the pool is rebuilt every 25 clips per worker -- MediaPipe workers grow
to ~1.4 GB, and returning frames to the parent piled up gigabytes.

### Smoke test (no video / no MediaPipe)

To validate the training + INT8-export wiring before any rendering exists,
`make_smoke_data.py` fabricates `sequences/`-format shards (random landmarks
with a per-gesture bias) so `train_classifier.py` itself runs end to end:

```bash
python3 make_smoke_data.py
python3 train_classifier.py --sequences sequences_smoke --epochs 4 --seq-len 48
```

This produces `model/open_gesture_int8.tflite` (~180 KB, INT8 in/out, one head
per metadata dimension). It proves the heads/losses/converter wire up; the
accuracy is meaningless (the data is noise).

### Authoring poses (the gating task)

`render_clip.py` refuses to render until a gesture's keyframes have real poses.
Each keyframe's `smplx_pose` must be filled with SMPL-X axis-angle params:

```json
"smplx_pose": {
  "global_orient":   [0,0,0],          //  3
  "body_pose":       [/* 63 */],       // 21 body joints × 3
  "left_hand_pose":  [/* 45 */],       // 15 finger joints × 3
  "right_hand_pose": [/* 45 */]
}
```

These come from a rig tool, mocap, or a motion-synthesis model — not hand-typed.
`render_clip.py`'s `load_avatar` / `apply_pose` are isolated so you can wire them
to your SMPL-X integration; verify them on a machine with BlenderProc + a GPU.
