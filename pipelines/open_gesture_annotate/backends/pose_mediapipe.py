"""MediaPipe backend: 21x2 hand landmarks and 33 body pose landmarks.

The `body_parts` field in manifest.json is overwhelmingly hand- and finger-valued.
No face model can see it, so this is the backend that makes `body_parts`
checkable at all.

Landmark coordinates are normalised to [0, 1] in image space; z is relative depth.

Model URL correction (Task 10, 2026-08-25): the pose_landmarker URL initially
drafted from memory --
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker.task
-- 404s. The current URL, verified against the official model index at
https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker, uses a
"latest" version segment and an underscore-qualified filename rather than a
pinned version "1" and a bare filename:
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
The hand_landmarker URL drafted from memory (float16/1/hand_landmarker.task)
was verified to work as written (HTTP 200, ~7.5 MB) and was left unchanged.

Installation note (Task 10): the `pose` extra pins mediapipe below 1.0.0 (see
pyproject.toml for why) and requires a post-install cleanup of the
opencv-contrib-python / opencv-python conflict mediapipe's own dependency
introduces -- also documented in pyproject.toml, next to the `pose` extra.

Licence verification (Task 10): unlike uniface's FairFace (CC BY 4.0, Task 7)
and py-feat's mobilefacenet (no LICENSE, Task 8), Apache-2.0 for these two
bundles is *not* an assumption carried over from the mediapipe library. Each
bundle publishes its own Model Card PDF, and both explicitly state
"LICENSED UNDER: Apache License, Version 2.0":
  - hand_landmarker: "Model Card MediaPipe Hands (Lite/Full)" --
    https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf
  - pose_landmarker: "Model Card MediaPipe BlazePose GHUM 3D" --
    https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf
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
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
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
