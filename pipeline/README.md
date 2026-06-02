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
| `landmark_format.py` | — | (imported module) | stdlib |
| `extract_landmarks.py` | clips + `labels/` | `sequences/` | mediapipe, opencv, numpy |
| `train_classifier.py` | `sequences/` + `labels/` | `model/` | tensorflow, numpy |

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

# ... author motion_specs/*.json, render in BlenderProc, augment in Cosmos ...

# Stage 1 — MediaPipe extraction (run AFTER Cosmos; same path as real inference)
python3 extract_landmarks.py --clips path/to/cosmos_clips/

# Stage 2 — train multi-head classifier + export INT8 TFLite
python3 train_classifier.py
#   then: TFLite/ONNX -> Hailo Dataflow Compiler -> .hef
```

Clip filenames must encode the gesture id before a `__` separator, e.g.
`affirm-01__lighting3_angle12.mp4`, so the extractor can join each clip to its
labels.
