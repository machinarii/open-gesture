"""RelateAnything (relsgg) backend: open-vocabulary spatial relations between body regions.

Why this backend exists. `physical_description` is the richest curated field in
manifest.json -- it is where the BAP-style claims live ("index finger pressed
vertically against the lips", "palm resting on the forehead") -- and until now
nothing checked it. `pose` proves a hand is *present*; no backend said anything
about where that hand is *relative to the rest of the body*. RelateAnything
scores (subject, predicate, object) triplets over regions the caller supplies,
against a predicate vocabulary the caller supplies, so it is the one model here
that can speak to spatial configuration.

One caveat governs how every triplet here should be read: relsgg returns ONE
predicate per region pair -- the argmax over the supplied vocabulary -- so
relations.json is the model's best single answer for each pair, not a ranked
set of everything that might be true. See GESTURE_PREDICATES for what that
costs and how the vocabulary is chosen around it.

It detects nothing itself, which is exactly why it fits: the regions come from
the landmarks `pose` already wrote. This backend reads pose.json read-only and
never writes it, mirroring how `va` reads `face`'s faces.json -- and it reports
itself unavailable when pose.json is absent for the same reason `va` does:
inventing whole-image boxes would produce confident, meaningless triplets
instead of an honest "did not run".

LICENCE WARNING -- read before redistributing anything built on relations.json.
relsgg's *code* is Apache-2.0, but its released checkpoints are derivatives of
Meta DINOv3 weights and are distributed under the DINOv3 License, NOT
Apache-2.0 (RelateAnything's own THIRD_PARTY_NOTICES.md states this for both
the vision backbone and the distilled text student). That is a stricter term
than any other weight in this pipeline, so `provenance()` declares it verbatim
and `og-annotate run` will warn on every run -- the warning is correct and is
not to be silenced. This is also why the `relations` extra is NOT installed by
default, the same call made for `wholebody`.

INSTALL CONFLICT -- relsgg depends on `opencv-python-headless`, and this project
pins `opencv-python`. Those two distributions install into the same `cv2/`
package directory; having both present is the documented cause of the
intermittent `libc++abi ... recursive_mutex lock failed` abort described in
pyproject.toml next to the `pose` extra. Install relsgg in a separate
environment, or repeat the one-distribution cleanup afterwards. See
annotate/README.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from open_gesture_annotate.io import Gesture, repo_root
from open_gesture_annotate.schema import ok

MODEL_ID = "maelic/relsgg-vits16plus"

# Supplied at inference time; relsgg embeds these strings once and scores every
# region pair against them. Chosen for gesture configuration rather than the
# object-interaction relations the released vocabulary leans towards.
# ONE PREDICATE PER PAIR. relsgg's predict() takes an argmax over the vocabulary
# for each region pair (relsgg/api.py: `best_s, best_p = scores.max(dim=-1)`), so
# a pair yields the model's single best answer, not a ranked list. Every string
# here competes with every other for that one slot, which makes the vocabulary a
# design decision rather than a wish list:
#
#   - Predicates that are true SIMULTANEOUSLY with contact are excluded. A hand
#     touching a face is also unavoidably "near" it and "in front of" it, so
#     those strings would win the slot without contradicting contact -- masking
#     it rather than competing with it.
#   - Predicates that are a genuine ALTERNATIVE claim are kept. "beside" beating
#     "touching" is the model saying this is not contact, which is exactly what
#     the self_contact check should conclude.
#
# Synonyms also split the ranking in this model (relsgg docs, "Changing the
# vocabulary"), so there is one phrasing per relation.
GESTURE_PREDICATES = [
    "touching",
    "covering",
    "resting on",
    "pressed against",
    "pointing at",
    "above",
    "below",
    "beside",
]

# Contact, as opposed to mere adjacency. "beside" is excluded on purpose: a hand
# held beside the head is not a hand touching it, and the adaptor check turns on
# that distinction.
CONTACT_PREDICATES = frozenset({"touching", "covering", "resting on", "pressed against"})

# NO ABSOLUTE SCORE THRESHOLD -- measured, not assumed. The checkpoint ships a
# Platt map that makes a score read as "probability a person would call this
# true", but that fit belongs to the released vocabulary. Swap the vocabulary
# and the scale goes with it: over all 99 gestures the median triplet score is
# 0.027 and the maximum is 0.785, so a 0.5 gate would have called 3 of the 5
# curated adaptors non-contact -- every one of them wrongly (see
# has_self_contact).
#
# What survives the swap is the RANKING, and it is sound. Sorting the 99
# gestures by their best hand-to-body contact score puts genuine hand-to-face
# gestures at the top -- Nose Thumb, Facepalm, Zipped Lips, Hat Tip, Salute,
# Cuckoo Sign, Finger to Lips -- spanning five different curated gesture_types.
# So contact is read off the model's own per-pair argmax, which needs no
# constant fitted to this dataset. Fitting one to the adaptor labels would make
# the check agree with the labels it exists to question.
#
# The strength behind that decision is still recorded per gesture, as
# `top_contact_score`, so a human can weigh a weak call.

# MediaPipe BlazePose 33-point indices, grouped into the regions a gesture
# description actually talks about.
BODY_REGIONS = {
    "head": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    "torso": (11, 12, 23, 24),
    "left arm": (11, 13, 15),
    "right arm": (12, 14, 16),
}
BODY_REGION_NAMES = frozenset(BODY_REGIONS)

MIN_VISIBILITY = 0.5  # MediaPipe landmark visibility below this is a guess, not a sighting
MIN_LANDMARKS = 2  # a region needs at least two sighted landmarks to bound a box
BOX_PAD = 0.02  # pad each derived box by 2% of the image dimensions, then clamp


def _bbox(points: list[tuple[float, float]], width: int, height: int) -> list[float] | None:
    """Pixel [x1, y1, x2, y2] bounding the normalised points, padded and clamped.

    Returns None for a degenerate result, so a region that collapses to a point
    is dropped rather than fed to the model as a zero-area box.
    """
    if len(points) < MIN_LANDMARKS:
        return None
    xs = [p[0] * width for p in points]
    ys = [p[1] * height for p in points]
    pad_x, pad_y = BOX_PAD * width, BOX_PAD * height
    x1 = max(0.0, min(xs) - pad_x)
    y1 = max(0.0, min(ys) - pad_y)
    x2 = min(float(width), max(xs) + pad_x)
    y2 = min(float(height), max(ys) + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def regions_from_pose(record: dict, width: int, height: int) -> list[dict]:
    """Derive labelled boxes from one pose.json record.

    Hands come from the hand landmarker (all 21 points); head, torso and arms
    from the body landmarker, using only landmarks MediaPipe reports as visible.
    Returns [] when the record yields nothing usable -- an absent pose is a
    reason not to predict, not an error.
    """
    if not isinstance(record, dict) or record.get("status") != "ok":
        return []

    regions: list[dict] = []
    for hand in record.get("hands") or []:
        if not isinstance(hand, dict):
            continue
        landmarks = hand.get("landmarks")
        if not isinstance(landmarks, list):
            continue
        box = _bbox([(p[0], p[1]) for p in landmarks if len(p) >= 2], width, height)
        if box is not None:
            side = str(hand.get("handedness", "")).lower()
            regions.append({"label": f"{side} hand".strip(), "kind": "hand", "box": box})

    body = record.get("body")
    landmarks = body.get("landmarks") if isinstance(body, dict) else None
    if isinstance(landmarks, list):
        for name, indices in BODY_REGIONS.items():
            points = [
                (landmarks[i][0], landmarks[i][1])
                for i in indices
                if i < len(landmarks)
                and len(landmarks[i]) >= 4
                and landmarks[i][3] >= MIN_VISIBILITY
            ]
            box = _bbox(points, width, height)
            if box is not None:
                regions.append({"label": name, "kind": "body", "box": box})

    return regions


def _hand_body_contacts(triplets: list[dict]) -> list[dict]:
    """Triplets where a hand and a body region are the model's contact answer.

    Directional either way: the model may return `hand -> touching -> head` or
    `head -> touching -> hand` for the same configuration, and both mean the
    gesture is self-touching.
    """
    return [
        t
        for t in triplets
        if t.get("predicate") in CONTACT_PREDICATES
        and {t.get("subject_kind"), t.get("object_kind")} == {"hand", "body"}
    ]


def has_self_contact(triplets: list[dict]) -> bool:
    """True when contact is the model's chosen relation for some hand-body pair.

    No score gate -- see the note above CONTACT_PREDICATES for why an absolute
    threshold does not transfer to a custom vocabulary. Because predict()
    returns one predicate per pair, a contact predicate appearing here already
    means contact beat every alternative in the vocabulary for that pair,
    including `beside`, which is the competing claim that matters.
    """
    return bool(_hand_body_contacts(triplets))


def top_contact_score(triplets: list[dict]) -> float:
    """Strongest hand-to-body contact score, or 0.0 if there is none.

    Only comparable across gestures within one run, like the report's other
    ranked signals -- never as an absolute probability.
    """
    contacts = _hand_body_contacts(triplets)
    return max((float(t.get("score", 0.0)) for t in contacts), default=0.0)


def _load_pose_index(path: Path) -> dict:
    """Read pose.json read-only. Never raises: any problem yields an empty index,
    which makes every gesture fall through to 'no regions'.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        records = data.get("records", {})
        return records if isinstance(records, dict) else {}
    except (OSError, ValueError):
        return {}


class RelSGGBackend:
    name = "relsgg-relations"
    sidecar = "relations.json"

    def __init__(self, device: str = "cpu", model_id: str = MODEL_ID) -> None:
        self._device = device
        self._model_id = model_id
        self._impl = None
        self._pose_index: dict | None = None
        self._output_dir: Path | None = None

    @property
    def version(self) -> str:
        try:
            from importlib.metadata import version

            return version("relsgg")
        except Exception:
            return "unknown"

    def set_output_dir(self, out_dir: Path) -> None:
        """Optional Backend-protocol hook (see base.Backend): read pose.json from
        the directory this run is writing to, not the repo default, so a run into
        a scratch --out never derives regions from a previous run's landmarks.
        """
        self._output_dir = Path(out_dir)
        self._pose_index = None

    def _pose_path(self) -> Path:
        base = self._output_dir if self._output_dir is not None else repo_root() / "annotations"
        return base / "pose.json"

    def available(self) -> tuple[bool, str]:
        try:
            import relsgg  # noqa: F401
        except ImportError as exc:
            return False, f"relsgg not installed ({exc}); pip install -e '.[relations]'"
        if not self._pose_path().is_file():
            return (
                False,
                "requires annotations/pose.json — run the pose backend first: "
                "og-annotate run --backends pose",
            )
        return True, f"relsgg {self.version} ({self._model_id}, {self._device})"

    def _pose(self) -> dict:
        if self._pose_index is None:
            self._pose_index = _load_pose_index(self._pose_path())
        return self._pose_index

    def _model(self):
        if self._impl is None:
            from relsgg import RelateAnything

            model = RelateAnything.from_pretrained(self._model_id, device=self._device)
            model.set_vocabulary(list(GESTURE_PREDICATES))
            self._impl = model
        return self._impl

    def provenance(self) -> dict:
        return {
            "library": {"name": "relsgg", "version": self.version, "license": "Apache-2.0"},
            "models": [
                {
                    "name": self._model_id,
                    "role": "open-vocabulary relation prediction over supplied regions",
                    "source": "https://github.com/Maelic/RelateAnything",
                    # Deliberately not Apache-2.0: this is the library/weight
                    # divergence provenance.py exists to surface, and the
                    # strictest one in this pipeline. licence_warnings() will
                    # flag it on every run. That is the intended behaviour.
                    "license": "DINOv3 License",
                }
            ],
            "note": (
                "Regions are NOT detected by this model — they are derived from the "
                "MediaPipe landmarks in pose.json (hands, head, torso, arms). The "
                "predicate vocabulary is supplied at inference time (GESTURE_PREDICATES) "
                "rather than trained in, so triplets are only as meaningful as those "
                "strings. Scores use the checkpoint's shipped calibration, fitted on "
                "adjudicated negatives."
            ),
            "warning": (
                "relsgg checkpoints are derivatives of Meta DINOv3 weights and carry the "
                "DINOv3 License, not the library's Apache-2.0 — see RelateAnything's "
                "THIRD_PARTY_NOTICES.md. Verify those terms before redistributing any "
                "work derived from relations.json. relsgg's [detector] extra is AGPL-3.0 "
                "and is not used here; this backend supplies its own regions."
            ),
        }

    def annotate(self, image: np.ndarray, gesture: Gesture) -> dict:
        height, width = image.shape[:2]
        regions = regions_from_pose(self._pose().get(gesture.id), width, height)
        if len(regions) < 2:
            # One region cannot form a pair. Recorded as a successful "nothing to
            # relate", not an error: hand-first crops with no visible body are a
            # normal framing in this dataset, not a failure.
            return ok(regions=regions, triplets=[], self_contact=False, reason="fewer than 2 regions")

        boxes = np.array([r["box"] for r in regions], dtype=np.float32)
        predictions = self._model().predict(
            image,  # relsgg accepts HWC numpy; OpenCV BGR is documented as fine
            boxes,
            box_labels=[r["label"] for r in regions],
            topk=len(regions) * (len(regions) - 1),
        )
        triplets = [
            {
                "subject": regions[t.subject_idx]["label"],
                "subject_kind": regions[t.subject_idx]["kind"],
                "predicate": str(t.predicate),
                "score": float(t.score),
                "object": regions[t.object_idx]["label"],
                "object_kind": regions[t.object_idx]["kind"],
            }
            for t in predictions
        ]
        return ok(
            regions=regions,
            triplets=triplets,
            self_contact=has_self_contact(triplets),
            top_contact_score=top_contact_score(triplets),
            vocabulary=list(GESTURE_PREDICATES),
        )
