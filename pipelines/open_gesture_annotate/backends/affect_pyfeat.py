"""py-feat backend: FACS Action Units, categorical emotion, 3D head pose.

FACS is Ekman & Friesen's coding system, which the project README cites. Action
Units give the affect metadata the same anatomical grounding that BAP gives
`body_parts`.

py-feat's detector works on file paths, not arrays, so `annotate` writes the
image to a temporary file. `identity_model=None` deliberately: face identity
embeddings are not needed and ArcFace weights carry a separate licence.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from open_gesture_annotate.io import Gesture
from open_gesture_annotate.schema import ok

FEAT_AUS = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
    "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24",
    "AU25", "AU26", "AU28", "AU43",
]
FEAT_EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
FEAT_POSE = ["Pitch", "Roll", "Yaw"]


class PyFeatBackend:
    name = "pyfeat-au"
    sidecar = "action_units.json"

    def __init__(self) -> None:
        self._impl = None

    @property
    def version(self) -> str:
        try:
            from importlib.metadata import version

            return version("py-feat")
        except Exception:
            return "unknown"

    def available(self) -> tuple[bool, str]:
        try:
            from feat.detector import Detectorv1  # noqa: F401
        except ImportError as exc:
            return False, f"py-feat not installed ({exc}); pip install -e '.[aus]'"
        return True, f"py-feat {self.version}"

    def _detector(self):
        if self._impl is None:
            from feat.detector import Detectorv1

            self._impl = Detectorv1(
                face_model="retinaface",
                landmark_model="mobilefacenet",
                au_model="xgb",
                emotion_model="resmasknet",
                identity_model=None,
                device="cpu",
            )
        return self._impl

    def provenance(self) -> dict:
        return {
            "library": {"name": "py-feat", "version": self.version, "license": "MIT"},
            "models": [
                {
                    "name": "retinaface",
                    "role": "detection",
                    "source": "biubug6/Pytorch_Retinaface (mobilenet0.25_Final.pth)",
                    "license": "MIT",
                },
                {
                    "name": "mobilefacenet",
                    "role": "landmarks",
                    "source": "cunjian/pytorch_face_landmark (mobilefacenet_model_best.pth.tar)",
                    "license": "unknown -- upstream repo carries no LICENSE file",
                },
                {
                    "name": "xgb",
                    "role": "action units",
                    "source": "py-feat's own trained weights (Cheong et al.), not third-party",
                    "license": "MIT (covered by py-feat's own licence)",
                },
                {
                    "name": "resmasknet",
                    "role": "emotion",
                    "source": "phamquiluan/ResidualMaskingNetwork "
                    "(ResMaskNet_Z_resmasking_dropout1_rot30.pth)",
                    "license": "MIT",
                },
            ],
            "warning": (
                "py-feat's own bundled LICENSE links model provenance per-weight rather than "
                "asserting one blanket licence; verified independently against each upstream "
                "repo on 2026-08-25. retinaface (biubug6/Pytorch_Retinaface) carries an explicit "
                "LICENSE.MIT. mobilefacenet's landmark weights come from "
                "cunjian/pytorch_face_landmark, which has NO LICENSE file at all -- do not "
                "assume MIT for it; treat as all-rights-reserved until the upstream author "
                "clarifies. resmasknet's source repo (phamquiluan/ResidualMaskingNetwork) added "
                "a formal MIT LICENSE file in 2021; note py-feat's own LICENSE instead points to "
                "a 2020-2021 GitHub issue where the author gave only an informal 'free to use "
                "for research' reply before that file existed -- the current MIT LICENSE file is "
                "the more authoritative source and is what is recorded here. xgb is py-feat's own "
                "trained AU classifier (not a third-party model), so it is covered by py-feat's "
                "own MIT licence."
            ),
        }

    def annotate(self, image: np.ndarray, gesture: Gesture) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.png"
            cv2.imwrite(str(path), image)
            fex = self._detector().detect([str(path)], data_type="image", progress_bar=False)

        if fex is None or len(fex) == 0:
            return ok(face_count=0, action_units={}, emotions={}, head_pose={})

        # py-feat's detect() always returns exactly one row per input image, even
        # when no face was found: FaceRectX/Y/Width/Height are NaN and FaceScore is
        # 0.0 in that case (confirmed empirically -- len(fex) is 1 either way, so it
        # cannot be used as the "no face" signal). Filter to rows with a real
        # detection before counting faces or reading AU/emotion/pose columns.
        if "FaceRectX" in fex.columns:
            fex = fex[fex["FaceRectX"].notna()]
        if len(fex) == 0:
            return ok(face_count=0, action_units={}, emotions={}, head_pose={})

        row = fex.iloc[0]

        def _grab(columns):
            return {c: float(row[c]) for c in columns if c in fex.columns and row[c] == row[c]}

        return ok(
            face_count=int(len(fex)),
            action_units=_grab(FEAT_AUS),
            emotions=_grab(FEAT_EMOTIONS),
            head_pose=_grab(FEAT_POSE),
        )
