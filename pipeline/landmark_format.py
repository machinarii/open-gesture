"""Shared landmark feature layout.

The extractor and the model must agree on exactly which landmarks become
features and in what order -- this module is the single source of truth for
that, the same way label_schema.json is the source of truth for class indices.

MediaPipe Holistic emits four landmark groups. For gesture recognition we use
pose + both hands and DROP the 468-point face mesh: it dominates the feature
vector, is mostly irrelevant to manual gestures, and inflates the on-device
cost for no benefit. (Add FACE back here if a gesture set ever needs it.)

Per-frame feature vector (concatenated in this fixed order):

    pose         33 landmarks × (x, y, z, visibility) = 132
    left_hand    21 landmarks × (x, y, z)             =  63
    right_hand   21 landmarks × (x, y, z)             =  63
                                              FEATURE_DIM = 258

Missing groups in a frame (hand out of view) are zero-filled, so every frame
is exactly FEATURE_DIM long regardless of what MediaPipe detected.
"""
from __future__ import annotations

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21

POSE_DIM = POSE_LANDMARKS * 4   # x, y, z, visibility
HAND_DIM = HAND_LANDMARKS * 3   # x, y, z

# Offsets into the concatenated per-frame vector.
POSE_OFFSET = 0
LEFT_HAND_OFFSET = POSE_OFFSET + POSE_DIM
RIGHT_HAND_OFFSET = LEFT_HAND_OFFSET + HAND_DIM
FEATURE_DIM = RIGHT_HAND_OFFSET + HAND_DIM   # 258


def frame_features(results) -> "list[float]":
    """Flatten one MediaPipe Holistic result into a FEATURE_DIM vector.

    Zero-fills any group MediaPipe didn't detect this frame. Coordinates are
    left in MediaPipe's normalized image space; per-clip normalization (e.g.
    re-centering on the torso) is the extractor's job, not this layout's.
    """
    vec = [0.0] * FEATURE_DIM

    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            base = POSE_OFFSET + i * 4
            vec[base : base + 4] = [lm.x, lm.y, lm.z, lm.visibility]

    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            base = LEFT_HAND_OFFSET + i * 3
            vec[base : base + 3] = [lm.x, lm.y, lm.z]

    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            base = RIGHT_HAND_OFFSET + i * 3
            vec[base : base + 3] = [lm.x, lm.y, lm.z]

    return vec
