"""RTMW backend: COCO-WholeBody 133 keypoints (optional extra).

Higher-fidelity hand keypoints than MediaPipe, at the cost of the mmcv/mmpose
dependency stack, which pins hard and installs badly. Isolated behind the
[wholebody] extra: available() reports the reason and the runner skips it, so a
failed (or skipped) install of this backend cannot affect any other.

Environment note (Task 12, 2026-08-25): five other backends (uniface, py-feat,
HSEmotion, MediaPipe, CLIP) share a delicately balanced dependency set --
torch 2.13.0, a single opencv distribution (opencv-python, never
opencv-contrib-python; see the `pose` extra's note in pyproject.toml for why two
distributions cannot coexist), mediapipe>=0.10,<1.0, and pinned numpy/protobuf
ranges. The mmcv/mmpose/mmdet stack pins its own torch/numpy/opencv versions
hard enough that installing `[wholebody]` was judged, before ever running pip,
to be very likely to downgrade one or more of those and destabilise the five
working backends the way an opencv distribution conflict already did once in
Task 10. Per the controller's ruling for this task, the install was dry-run
first (`pip install --dry-run -e ".[wholebody]"`); if it proposed changing
torch/numpy/opencv/protobuf/mediapipe, the install was skipped entirely rather
than risking the working environment. See task-12-report.md for the dry-run
output and the resulting decision. Whatever that decision was, this module and
its tests are written to work identically either way: `available()` always
returns a bool and a reason, and never raises.

Licence verification (Task 12, 2026-08-25): as with Tasks 7, 8 and 11, a
library's own licence is not assumed to propagate to a checkpoint it ships.
mmpose's repository LICENSE (github.com/open-mmlab/mmpose) is Apache License
2.0 -- that is what `provenance()["library"]["license"]` records, and it is a
direct read of the repo's LICENSE file, not an inference from the model type.

The RTMW wholebody checkpoint used by `MMPoseInferencer("wholebody")` is
documented (configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw_cocktail14.md
in the mmpose repo) as trained on "cocktail14", a mix of 14 public datasets:
AI Challenger, CrowdPose, MPII, sub-JHMDB, Halpe, PoseTrack18, COCO-WholeBody,
UBody, Human-Art, WFLW, 300W, COFW, LaPa and InterHand. Several of those
(e.g. AI Challenger, InterHand2.6M, Human-Art) are published under
research-only / non-commercial licences, and OpenMMLab does not publish a
separate licence statement for the resulting checkpoint weights themselves --
no LICENSE file, model card, or download page accompanies the checkpoint with
explicit terms distinct from the code. Apache-2.0 covers the *code* that loads
and runs the checkpoint; it is not assumed to cover the checkpoint's weights,
which were fit to data under mixed and partly restrictive licences. The models
entry below is therefore recorded as "unspecified" with that caveat, rather
than guessed at Apache-2.0.
"""

from __future__ import annotations

import numpy as np

from open_gesture_annotate.io import Gesture
from open_gesture_annotate.schema import ok

# COCO-WholeBody index ranges, [start, end).
KEYPOINT_GROUPS = {
    "body": (0, 17),
    "feet": (17, 23),
    "face": (23, 91),
    "hands": (91, 133),
}


class RTMWBackend:
    name = "rtmw-wholebody"
    sidecar = "wholebody.json"

    def __init__(self) -> None:
        self._impl = None

    @property
    def version(self) -> str:
        try:
            from importlib.metadata import version

            return version("mmpose")
        except Exception:
            return "unknown"

    def available(self) -> tuple[bool, str]:
        try:
            from mmpose.apis import MMPoseInferencer  # noqa: F401
        except Exception as exc:
            return False, (
                f"mmpose not usable ({type(exc).__name__}: {exc}); "
                "pip install -e '.[wholebody]' — this stack pins hard and may not install"
            )
        return True, f"mmpose {self.version}"

    def _inferencer(self):
        if self._impl is None:
            from mmpose.apis import MMPoseInferencer

            self._impl = MMPoseInferencer("wholebody")
        return self._impl

    def provenance(self) -> dict:
        return {
            "library": {"name": "mmpose", "version": self.version, "license": "Apache-2.0"},
            "models": [
                {
                    "name": "RTMW wholebody (cocktail14)",
                    "keypoints": 133,
                    "license": "unspecified",
                    "note": (
                        "OpenMMLab does not publish a separate licence for this checkpoint. "
                        "Trained on cocktail14, a mix of 14 datasets, some under "
                        "research-only/non-commercial licences (e.g. AI Challenger, "
                        "InterHand2.6M, Human-Art). mmpose's Apache-2.0 covers the code that "
                        "loads it, not necessarily the weights. Verify before commercial use."
                    ),
                }
            ],
        }

    def annotate(self, image: np.ndarray, gesture: Gesture) -> dict:
        people = []
        for result in self._inferencer()(image):
            for pred in result["predictions"][0]:
                people.append(
                    {
                        "keypoints": [[float(x), float(y)] for x, y in pred["keypoints"]],
                        "scores": [float(s) for s in pred["keypoint_scores"]],
                    }
                )
        return ok(
            person_count=len(people),
            people=people,
            groups={k: list(v) for k, v in KEYPOINT_GROUPS.items()},
        )
