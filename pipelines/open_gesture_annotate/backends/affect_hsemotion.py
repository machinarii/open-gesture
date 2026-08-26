"""HSEmotion backend: continuous valence and arousal.

The valence/arousal output is UNDOCUMENTED. HSEmotion's README describes only
discrete emotions. Source inspection of hsemotion/facial_emotions.py on
2026-08-25 shows predict_emotions() computes its softmax over scores[:-2],
which means the enet_b0_8_va_mtl multi-task head appends valence and arousal as
the final two elements. split_scores() asserts that layout so an upstream change
fails loudly instead of writing garbage.

Valence and arousal are what the README's PAD framing (Russell & Mehrabian 1977)
actually calls for, and are the signal `report.py` checks `emotional_state` and
`arousal` against.

Face cropping: HSEmotion expects a face crop, not a full gesture photo (where the
face is often small and off-centre). This backend reads the face bounding boxes
Task 7's uniface backend already wrote to annotations/faces.json (read-only; this
backend never writes that file), expands the first face's bbox by a small margin
and crops to it, and only falls back to the full image when no usable face record
exists. See `_crop_for_recognition()`.

Environment note (2026-08-25, torch 2.13.0 / timm 1.0.28 installed by Task 8):
hsemotion 0.3.0's HSEmotionRecognizer.__init__ calls `torch.load(path)` with no
`weights_only=` argument. Torch >=2.6 flipped that default to True, and the
enet_b0_8_va_mtl checkpoint pickles a full timm EfficientNet *instance* (not a
state_dict), so it fails to unpickle under weights_only=True. Separately,
unpickling reconstructs each block's __dict__ directly (bypassing __init__), so
blocks lack attributes newer timm added after this checkpoint was pickled (a
`conv_s2d`/`bn_s2d` space-to-depth stem and an `aa` anti-alias submodule) -- both
optional, off-by-default behaviour whose absence should be a no-op; forward()
just guards `if self.conv_s2d is not None` and unconditionally calls `self.aa`.
`_build_recognizer()` works around both issues without touching the installed
torch or timm versions (so Task 8's py-feat backend is unaffected): it loads the
checkpoint with weights_only=False -- the same trust decision hsemotion's authors
made when they wrote this code before torch 2.6 existed, for a checkpoint that
ships from HSEmotion's own GitHub release -- then patches the missing attributes
onto the model's already-constructed submodules so forward() runs unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from open_gesture_annotate.io import Gesture, repo_root
from open_gesture_annotate.schema import ok

MODEL_NAME = "enet_b0_8_va_mtl"
N_EMOTIONS = 8
EXPECTED_SCORE_LEN = N_EMOTIONS + 2  # 8 emotion logits + valence + arousal

_FACE_BBOX_MARGIN = 0.20  # expand the detected face bbox by 20% before cropping


def split_scores(scores, classes: list[str]) -> tuple[dict, float, float]:
    """Split the va_mtl output into emotion scores, valence and arousal."""
    scores = np.asarray(scores).ravel()
    if scores.size != EXPECTED_SCORE_LEN:
        raise ValueError(
            f"HSEmotion {MODEL_NAME} returned {scores.size} scores, expected "
            f"{EXPECTED_SCORE_LEN} ({N_EMOTIONS} emotions + valence + arousal). "
            "The undocumented output layout this backend relies on has changed."
        )
    emotions = {name: float(scores[i]) for i, name in enumerate(classes[:N_EMOTIONS])}
    return emotions, float(scores[-2]), float(scores[-1])


def _build_recognizer():
    """Construct the va_mtl HSEmotionRecognizer, working around the torch/timm
    version skew described in the module docstring. Scoped and non-persistent:
    it patches torch.load only for the duration of this call, and patches
    attributes only on this specific model instance -- it cannot affect any
    other backend or any other HSEmotionRecognizer.
    """
    import torch
    import torch.nn as nn

    original_load = torch.load

    def _load_full_checkpoint(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = _load_full_checkpoint
    try:
        from hsemotion.facial_emotions import HSEmotionRecognizer

        fer = HSEmotionRecognizer(model_name=MODEL_NAME, device="cpu")
    finally:
        torch.load = original_load

    # Patch attributes newer timm block classes reference but this pickled,
    # older-era model instance never had __init__ called with. Materialize the
    # module list first: mutating _modules while .modules() is still lazily
    # recursing through it corrupts the generator's traversal.
    for module in list(fer.model.modules()):
        for attr in ("conv_s2d", "bn_s2d"):
            if attr not in module.__dict__:
                module.__dict__[attr] = None
        if "aa" not in module.__dict__ and "aa" not in getattr(module, "_modules", {}):
            module.add_module("aa", nn.Identity())

    return fer


def _load_faces_index(root: Path) -> dict:
    """Read annotations/faces.json (Task 7) read-only. Never raises -- any
    problem (missing file, bad JSON, unexpected shape) yields an empty index,
    which makes every gesture fall back to the full image.
    """
    path = Path(root) / "annotations" / "faces.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data.get("records", {})
        return records if isinstance(records, dict) else {}
    except (OSError, ValueError):
        return {}


def _expand_and_clamp_bbox(bbox, width: int, height: int, margin: float = _FACE_BBOX_MARGIN):
    """Expand a [x1, y1, x2, y2] bbox by `margin` on each side, then clamp to
    the image bounds. Returns an (x1, y1, x2, y2) int tuple, or None if the
    result is degenerate (zero or negative area).
    """
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    dx = (x2 - x1) * margin
    dy = (y2 - y1) * margin
    x1, x2 = x1 - dx, x2 + dx
    y1, y2 = y1 - dy, y2 + dy
    cx1, cy1 = max(0, int(x1)), max(0, int(y1))
    cx2, cy2 = min(width, int(x2)), min(height, int(y2))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return cx1, cy1, cx2, cy2


class HSEmotionBackend:
    name = "hsemotion-va"
    sidecar = "valence_arousal.json"

    def __init__(self) -> None:
        self._impl = None
        self._faces_index: dict | None = None

    @property
    def version(self) -> str:
        try:
            from importlib.metadata import version

            return version("hsemotion")
        except Exception:
            return "unknown"

    def available(self) -> tuple[bool, str]:
        try:
            from hsemotion.facial_emotions import HSEmotionRecognizer  # noqa: F401
        except ImportError as exc:
            return False, f"hsemotion not installed ({exc}); pip install -e '.[va]'"
        return True, f"hsemotion {self.version} ({MODEL_NAME})"

    def _recognizer(self):
        if self._impl is None:
            self._impl = _build_recognizer()
        return self._impl

    def _faces(self) -> dict:
        """Load annotations/faces.json once and cache it for the lifetime of
        this backend instance -- not re-read per image.
        """
        if self._faces_index is None:
            self._faces_index = _load_faces_index(repo_root())
        return self._faces_index

    def provenance(self) -> dict:
        return {
            "library": {"name": "hsemotion", "version": self.version, "license": "Apache-2.0"},
            "models": [
                {
                    "name": MODEL_NAME,
                    "role": "emotion + valence/arousal",
                    "source": "HSE-asavchenko/face-emotion-recognition "
                    "(models/affectnet_emotions/enet_b0_8_va_mtl.pt)",
                    "license": "Apache-2.0",
                }
            ],
            "note": (
                "valence/arousal read from scores[-2:]; layout is undocumented "
                "and asserted at runtime by split_scores(). Verified the "
                "HSE-asavchenko/face-emotion-recognition repository (which hosts "
                "the enet_b0_8_va_mtl weights this backend downloads) carries an "
                "Apache-2.0 LICENSE file at its root, same as the hsemotion "
                "PyPI package itself -- unlike Task 7's FairFace (CC BY 4.0) and "
                "Task 8's mobilefacenet (no LICENSE at all), this weight's "
                "licence does match the library's."
            ),
            "warning": (
                "This installation works around a torch/timm version skew: "
                "hsemotion 0.3.0 unpickles the va_mtl checkpoint with "
                "torch.load(weights_only=True) unavailable (torch >=2.6 default), "
                "so this backend loads it with weights_only=False and patches "
                "attributes newer timm added after the checkpoint was pickled. "
                "See the module docstring and _build_recognizer() for detail."
            ),
        }

    def _crop_for_recognition(self, image: np.ndarray, gesture: Gesture) -> tuple[np.ndarray, str]:
        """Return (crop, face_source) per the CONTROLLER RULING: use the face
        bbox Task 7 already found when one exists, else fall back to the full
        image. Never raises.
        """
        height, width = image.shape[:2]
        record = self._faces().get(gesture.id)
        if not isinstance(record, dict) or record.get("status") != "ok":
            return image, "full-image"
        if not isinstance(record.get("face_count"), int) or record["face_count"] < 1:
            return image, "full-image"
        faces = record.get("faces")
        if not isinstance(faces, list) or not faces:
            return image, "full-image"
        bbox = faces[0].get("bbox") if isinstance(faces[0], dict) else None
        if not isinstance(bbox, list) or len(bbox) < 4:
            return image, "full-image"
        clamped = _expand_and_clamp_bbox(bbox, width, height)
        if clamped is None:
            return image, "full-image"
        x1, y1, x2, y2 = clamped
        return image[y1:y2, x1:x2], "uniface"

    def annotate(self, image: np.ndarray, gesture: Gesture) -> dict:
        fer = self._recognizer()
        crop, face_source = self._crop_for_recognition(image, gesture)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        label, scores = fer.predict_emotions(rgb, logits=False)
        classes = [fer.idx_to_class[i] for i in sorted(fer.idx_to_class)]
        emotions, valence, arousal = split_scores(scores, classes)
        return ok(
            label=str(label),
            valence=valence,
            arousal=arousal,
            emotion_scores=emotions,
            face_source=face_source,
        )
