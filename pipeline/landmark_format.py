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


LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12


def normalize(seq, aspect: float = 1.0):
    """Make a [T, FEATURE_DIM] sequence invariant to where the person stands and how big they are.

    `aspect` is the source frame's width / height. MediaPipe normalises x and y
    to the frame's own width and height, so x is first rescaled by it to put
    both axes in the same units -- otherwise a 3:4 portrait frame and a square
    one give the same hand two different shapes.

    Raw features are in image space, so the same hand shape at a different
    position or distance is a different vector -- a model then learns framing
    instead of gesture. Per frame:

      pose   x, y, z centred on the shoulder midpoint and divided by shoulder
             width (visibility untouched), so arm placement stays body-relative;
      hands  x, y, z centred on that hand's wrist and divided by the hand's
             extent, so finger shape is independent of where the hand is (the
             pose wrists still carry where it is).

    Zero-filled (undetected) groups stay zero. Apply at training AND inference;
    shards on disk stay raw. Requires numpy.
    """
    import numpy as np

    x = np.array(seq, dtype=np.float32, copy=True)
    x[:, POSE_OFFSET:POSE_OFFSET + POSE_DIM:4] *= aspect
    x[:, LEFT_HAND_OFFSET::3] *= aspect  # both hands: x is every 3rd value from here on
    pose = x[:, POSE_OFFSET:POSE_OFFSET + POSE_DIM].reshape(len(x), POSE_LANDMARKS, 4)
    has_pose = pose.any(axis=(1, 2))
    shoulders = pose[:, [LEFT_SHOULDER, RIGHT_SHOULDER], :3]
    centre = shoulders.mean(axis=1, keepdims=True)
    width = np.linalg.norm(shoulders[:, 0, :2] - shoulders[:, 1, :2], axis=1)
    width = np.maximum(width, 1e-3)[:, None, None]
    pose[has_pose, :, :3] = ((pose[:, :, :3] - centre) / width)[has_pose]

    for offset in (LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET):
        hand = x[:, offset:offset + HAND_DIM].reshape(len(x), HAND_LANDMARKS, 3)
        present = hand.any(axis=(1, 2))
        rel = hand - hand[:, :1]
        extent = np.maximum(np.linalg.norm(rel[:, :, :2], axis=2).max(axis=1), 1e-6)[:, None, None]
        hand[present] = (rel / extent)[present]
    return x


# MediaPipe BlazePose left/right landmark pairs (eyes, ears, mouth, arms, hands, legs, feet).
POSE_MIRROR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
                     (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]


def mirror(x):
    """Left-right mirror of NORMALIZED [N, T, FEATURE_DIM] features (see normalize).

    Normalised x is centred (shoulders for pose, wrist for hands), so mirroring
    is negating x, then swapping the left/right pose landmarks and hand slots.
    Only valid for gestures whose meaning survives a mirror: NOT e.g. the left
    and right bicycle turn signals.
    """
    import numpy as np

    m = np.array(x, dtype=np.float32, copy=True)
    m[..., POSE_OFFSET:POSE_OFFSET + POSE_DIM:4] *= -1
    m[..., LEFT_HAND_OFFSET::3] *= -1
    pose = m[..., POSE_OFFSET:POSE_OFFSET + POSE_DIM].reshape(*m.shape[:-1], POSE_LANDMARKS, 4)
    left, right = zip(*POSE_MIRROR_PAIRS)
    pose[..., list(left) + list(right), :] = pose[..., list(right) + list(left), :]
    hands = m[..., LEFT_HAND_OFFSET:].copy()
    m[..., LEFT_HAND_OFFSET:RIGHT_HAND_OFFSET] = hands[..., HAND_DIM:]
    m[..., RIGHT_HAND_OFFSET:] = hands[..., :HAND_DIM]
    return m


if __name__ == "__main__":
    # Self-check: moving and scaling the whole person must not change the features.
    import numpy as np

    rng = np.random.default_rng(0)
    seq = rng.uniform(0.2, 0.8, size=(3, FEATURE_DIM)).astype(np.float32)
    seq[2, RIGHT_HAND_OFFSET:] = 0  # an undetected hand stays zero
    moved = seq.copy()
    for offset, dim, step in ((POSE_OFFSET, POSE_DIM, 4), (LEFT_HAND_OFFSET, HAND_DIM, 3), (RIGHT_HAND_OFFSET, HAND_DIM, 3)):
        block = moved[:, offset:offset + dim].reshape(3, -1, step)
        live = block.any(axis=(1, 2))
        block[live, :, :2] = block[live, :, :2] * 0.5 + 0.1
        block[live, :, 2] *= 0.5
    assert np.allclose(normalize(seq), normalize(moved), atol=1e-4)
    assert not normalize(seq)[2, RIGHT_HAND_OFFSET:].any()
    # A hand stretched along x by a 3:4 frame must match the square-frame original.
    wide = seq.copy()
    wide[:, POSE_OFFSET:POSE_DIM:4] /= 0.75
    wide[:, LEFT_HAND_OFFSET::3] /= 0.75
    assert np.allclose(normalize(seq), normalize(wide, aspect=0.75), atol=1e-4)
    assert np.allclose(mirror(mirror(normalize(seq))), normalize(seq))
    print("normalize, mirror: ok")
