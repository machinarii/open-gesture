"""uniface backend: face geometry, head pose, gaze, emotion, demographics.

Predictor class names discovered on 2026-08-25 (see Task 7 Steps 1-2, uniface 4.0.0,
installed via the `face` extra i.e. `uniface[cpu]`):

    detection    -> SCRFD (uniface.detection.SCRFD)
                    FaceAnalyzer's own default detector (SCRFD_500M_KPS weights) when
                    no `detector=` is passed; used unmodified here.
    landmarks    -> Landmark106 (uniface.landmark.Landmark106), 106-point "2d_106"
                    weights from InsightFace's alignment/coordinate_reg.
    head pose    -> HeadPose (uniface.headpose.HeadPose), ResNet-18 6D-rotation model.
    gaze         -> MobileGaze (uniface.gaze.MobileGaze), ResNet-34 backbone,
                    trained on Gaze360.
    emotion      -> Emotion (uniface.attribute.emotion.Emotion), AffectNet7 weights.
                    NOT wired up: this class requires the optional 'torch' dependency
                    (`Emotion requires optional dependency 'torch'`), and this
                    project's `face` extra (`uniface[cpu]>=4.0.0`) does not install
                    torch. Every record's "emotion" field is therefore None on this
                    installation. See the class docstring below for how to enable it.
    demographics -> FairFace (uniface.attribute.fairface.FairFace)

Key API finding from Step 1 that shaped this module: `FaceAnalyzer(predictors=[...])`
only accepts `BaseAttribute` subclasses (in this version: AgeGender, Emotion,
FairFace, FaceAttribNet) -- these run once per detected face and enrich the `Face`
object in place. HeadPose, MobileGaze and Landmark106 are *not* `BaseAttribute`
subclasses; they are standalone estimators with their own `estimate(face_crop)` /
`get_landmarks(image, bbox)` methods (confirmed against their docstring examples)
and must be invoked manually per face after `FaceAnalyzer.analyze()` returns. This
backend therefore builds four separate uniface objects: a `FaceAnalyzer` (detector +
FairFace predictor) plus a standalone `Landmark106`, `HeadPose` and `MobileGaze`.

Step 2 confirmed the `Face` object's real attributes (uniface 4.0.0): bbox,
bbox_xywh, bbox_xyxy, confidence, landmarks (5-point, from the detector),
embedding, gender, age, age_group, race, sex, emotion, emotion_confidence,
left_eye_open, right_eye_open, eyeglasses, mask, sunglasses, quality, track_id.
There is no head_pose/gaze attribute on `Face` at all, consistent with those two
families not being `BaseAttribute` predictors.
"""

from __future__ import annotations

import numpy as np

from open_gesture_annotate.io import Gesture
from open_gesture_annotate.schema import ok


def _as_list(value):
    """Convert numpy scalars/arrays to plain JSON-serialisable Python."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_as_list(v) for v in value]
    if isinstance(value, dict):
        return {k: _as_list(v) for k, v in value.items()}
    return value


class UniFaceBackend:
    name = "uniface-face"
    sidecar = "faces.json"

    def __init__(self) -> None:
        self._impl = None

    @property
    def version(self) -> str:
        try:
            import uniface

            return getattr(uniface, "__version__", "unknown")
        except ImportError:
            return "unavailable"

    def available(self) -> tuple[bool, str]:
        try:
            import uniface  # noqa: F401
        except ImportError as exc:
            return False, f"uniface not installed ({exc}); pip install -e '.[face]'"
        try:
            self._analyzer()
        except Exception as exc:
            return False, f"uniface installed but weights failed to load: {exc}"
        return True, f"uniface {self.version}"

    def _analyzer(self):
        """Build the FaceAnalyzer and the standalone estimators once.

        Predictor names come from the module docstring (Step 1/2 discovery).
        `FaceAnalyzer.predictors` only accepts `BaseAttribute` subclasses, so
        FairFace is passed there; head pose, gaze and landmarks are separate
        objects invoked per-face in `annotate()`. Emotion is attempted and
        silently omitted from the predictor list if 'torch' is unavailable --
        that role then yields `None` via the existing getattr pattern rather
        than raising or being fabricated.
        """
        if self._impl is None:
            from uniface import Emotion, FairFace, HeadPose, Landmark106, MobileGaze
            from uniface import FaceAnalyzer

            predictors = [FairFace()]
            try:
                predictors.append(Emotion())
            except ImportError:
                pass  # torch not installed under the `face` (uniface[cpu]) extra

            self._impl = {
                "analyzer": FaceAnalyzer(predictors=predictors),
                "landmarker": Landmark106(),
                "headpose": HeadPose(),
                "gaze": MobileGaze(),
            }
        return self._impl

    def provenance(self) -> dict:
        return {
            "library": {"name": "uniface", "version": self.version, "license": "MIT"},
            "models": [
                {
                    "name": "SCRFD (detector, SCRFD_500M_KPS)",
                    "role": "detection",
                    "source": "InsightFace",
                    "license": "MIT",
                },
                {
                    "name": "Landmark106 (2d_106)",
                    "role": "landmarks",
                    "source": "InsightFace alignment/coordinate_reg",
                    "license": "check upstream weight licence",
                },
                {
                    "name": "HeadPose (ResNet-18, 6D rotation)",
                    "role": "head_pose",
                    "source": "yakhyo/head-pose-estimation",
                    "license": "check upstream weight licence",
                },
                {
                    "name": "MobileGaze (ResNet-34)",
                    "role": "gaze",
                    "source": "yakhyo/gaze-estimation",
                    "license": "MIT",
                },
                {
                    "name": "FairFace",
                    "role": "demographics",
                    "source": "yakhyo/fairface-onnx",
                    "license": "CC BY 4.0",
                },
            ],
            "warning": (
                "uniface's own README states some pretrained weights are not MIT; see "
                "https://yakhyo.github.io/uniface/license-attribution/ for the maintained "
                "table. FairFace is CC BY 4.0, not MIT -- do not assume MIT for it. "
                "Landmark106 and HeadPose weights are not listed on that page at all; "
                "confirm their licence with the upstream repos above before commercial "
                "redistribution. The Emotion (AffectNet7) predictor requires the optional "
                "'torch' dependency, which this project's `face` extra (uniface[cpu]) does "
                "not install, so it is never loaded and its weight is not listed here."
            ),
        }

    def annotate(self, image: np.ndarray, gesture: Gesture) -> dict:
        impl = self._analyzer()
        analyzer = impl["analyzer"]
        landmarker = impl["landmarker"]
        headpose = impl["headpose"]
        gaze = impl["gaze"]

        height, width = image.shape[:2]
        faces = []
        for face in analyzer.analyze(image):
            bbox = _as_list(getattr(face, "bbox", None))

            # Dense 106-point landmarks from the bbox; fall back to the detector's
            # own 5-point alignment landmarks if the dedicated model errors out.
            try:
                landmarks = _as_list(landmarker.get_landmarks(image, face.bbox))
            except Exception:
                landmarks = _as_list(getattr(face, "landmarks", None))

            # Head pose and gaze operate on a face crop (per their docstring
            # examples), not on the full image. Clip the bbox to the image
            # bounds first so a detection near an edge can't produce an empty
            # crop and raise inside the estimator.
            crop = None
            if bbox is not None:
                x1, y1, x2, y2 = bbox[:4]
                cx1, cy1 = max(0, int(x1)), max(0, int(y1))
                cx2, cy2 = min(width, int(x2)), min(height, int(y2))
                if cx2 > cx1 and cy2 > cy1:
                    crop = image[cy1:cy2, cx1:cx2]

            head_pose = None
            gaze_out = None
            if crop is not None:
                try:
                    hp = headpose.estimate(crop)
                    head_pose = {"pitch": hp.pitch, "yaw": hp.yaw, "roll": hp.roll}
                except Exception:
                    head_pose = None
                try:
                    gz = gaze.estimate(crop)
                    gaze_out = {"pitch": gz.pitch, "yaw": gz.yaw}
                except Exception:
                    gaze_out = None

            emotion_label = getattr(face, "emotion", None)
            emotion = None
            if emotion_label is not None:
                emotion = {
                    "label": _as_list(emotion_label),
                    "scores": {"confidence": _as_list(getattr(face, "emotion_confidence", None))},
                }

            faces.append(
                {
                    "bbox": bbox,
                    "confidence": _as_list(getattr(face, "confidence", None)),
                    "landmarks": landmarks,
                    "head_pose": head_pose,
                    "gaze": gaze_out,
                    "emotion": emotion,
                    "demographics": {
                        "age_group": _as_list(getattr(face, "age_group", None)),
                        "sex": _as_list(getattr(face, "sex", None)),
                    },
                }
            )
        return ok(faces=faces, face_count=len(faces))
