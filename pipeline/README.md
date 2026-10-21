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

The fast model's one miss is Evil Horns read as `one`, which the full
model misses too. With 25 test gestures, one gesture is 4 points: treat
differences of a gesture or two as noise.

Tripling the data and adding `none` did not move the full model, and the
gap to the fast model is mostly that noise: run on Holistic's hand
landmarks instead of HandLandmarker's, the fast model scores 23/25. The
remaining misses are index-finger shapes (point, horns) read as `mute` and
crossed forearms read as praying hands. More signal now needs a bigger
test set, not more HaGRID.

`--seq-len 32` because 56k samples x 64 frames does not fit in 32 GB once
mirrored; a HaGRID sample is one frame tiled, so 32 loses nothing.
`import_hagrid.py` is resumable -- re-run the same command after an
interruption.

The `none` class (export_labels.NONE) is appended to every head, so no
existing class index moved.

Per-frame CPU cost (one thread, 640px frame): MediaPipe HandLandmarker
23 ms, Holistic 37 ms, full INT8 TFLite classifier 0.04 ms per 64-frame
window. Landmark extraction is the whole cost; the classifiers are noise.

`--mirror` adds left-right mirrored copies. Leave it off once direction-
sensitive gestures (the bicycle turn signals) are in the training set.

`train_classifier.py` normalises every sequence with
`landmark_format.normalize` (body-relative pose, wrist-relative hand shape,
aspect-corrected x). Inference must apply the same function to its frames.

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
