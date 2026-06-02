#!/usr/bin/env python3
"""Capture the posed SMPL-X armature into smplx_pose params (Blender-side).

This is the add-on -> JSON exporter. You pose the SMPL-X body in Blender (arms
via IK, fingers via the add-on's FK/curl controls or a hand preset), then run
this from Blender's Python to read the armature back out as MANO/SMPL-X
axis-angle params.

Run inside Blender (the SMPL-X add-on must be installed and the SMPL-X armature
selected/active):

    blender --background your.blend --python smplx_capture.py -- \
        --preset ok_ring --hand right \
        --library hand_presets/library.json

    # or capture a full body+hands pose into one motion-spec keyframe:
    blender --background your.blend --python smplx_capture.py -- \
        --keyframe motion_specs/greet-01.json:1

═══════════════════════════════════════════════════════════════════════════
VERIFY-ON-YOUR-RIG (unexecuted here — no Blender/add-on in the authoring env):

  * Bone naming below assumes the Meshcapade SMPL-X add-on's armature
    ('pelvis', 'left_wrist', 'left_index1'...). Confirm against your version;
    only read_smplx_pose() touches bone names.
  * PCA HANDS: the add-on can store hands as a reduced PCA basis. This exporter
    reads per-joint bone rotations (full 45-dim), which is what apply_pose and
    the library expect. If your add-on is in PCA mode, switch hand pose to
    'axis-angle'/full before capturing, or the finger detail won't round-trip.
═══════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# SMPL-X joint order (subset that we read). The full body_pose is 21 joints
# after pelvis; each hand is 15 finger joints. Names follow the SMPL-X skeleton.
BODY_JOINTS = [
    "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]
FINGERS = ["index", "middle", "pinky", "ring", "thumb"]
HAND_JOINTS = [f"{finger}{i}" for finger in FINGERS for i in (1, 2, 3)]  # 15 per hand


def _args(argv):
    parser = argparse.ArgumentParser(description="Capture SMPL-X pose from Blender.")
    parser.add_argument("--preset", help="Save the active hand as this library preset")
    parser.add_argument("--hand", choices=["left", "right"], default="right", help="Which hand for --preset")
    parser.add_argument("--keyframe", help="Save full pose into SPEC.json:KEYFRAME_INDEX")
    parser.add_argument("--library", type=Path, default=Path("hand_presets/library.json"))
    return parser.parse_args(argv)


def read_smplx_pose(armature) -> dict:
    """Read the active SMPL-X armature into an smplx_pose dict (axis-angle radians).

    ADAPT bone names if your add-on differs. Each pose_bone rotation is read as
    an axis-angle vector (angle * axis) to match the MANO/SMPL-X convention.
    """
    from mathutils import Vector  # Blender-only import

    def bone_axis_angle(name) -> list:
        pb = armature.pose.bones.get(name)
        if pb is None:
            return [0.0, 0.0, 0.0]
        q = pb.rotation_quaternion if pb.rotation_mode == "QUATERNION" else pb.matrix_basis.to_quaternion()
        angle = q.angle
        axis = q.axis if angle else Vector((0.0, 0.0, 0.0))
        return [axis.x * angle, axis.y * angle, axis.z * angle]

    pose = {
        "global_orient": bone_axis_angle("pelvis"),
        "body_pose": [c for j in BODY_JOINTS for c in bone_axis_angle(j)],
        "left_hand_pose": [c for j in HAND_JOINTS for c in bone_axis_angle(f"left_{j}")],
        "right_hand_pose": [c for j in HAND_JOINTS for c in bone_axis_angle(f"right_{j}")],
    }
    return pose


def get_active_armature():
    import bpy
    arm = bpy.context.view_layer.objects.active
    if arm is None or arm.type != "ARMATURE":
        sys.exit("error: active object is not an SMPL-X armature. Select it before capturing.")
    return arm


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = _args(argv)
    if not args.preset and not args.keyframe:
        sys.exit("error: pass --preset NAME or --keyframe SPEC:INDEX")

    armature = get_active_armature()
    pose = read_smplx_pose(armature)

    if args.preset:
        lib = json.loads(args.library.read_text())
        if args.preset not in lib["presets"]:
            sys.exit(f"error: '{args.preset}' not in library; add it to build_hand_presets.py first")
        hand_key = "left_hand_pose" if args.hand == "left" else "right_hand_pose"
        lib["presets"][args.preset][args.hand] = pose[hand_key]
        args.library.write_text(json.dumps(lib, indent=2) + "\n")
        print(f"Captured {args.hand} hand -> preset '{args.preset}' in {args.library}")

    if args.keyframe:
        spec_path_str, _, idx_str = args.keyframe.partition(":")
        spec_path, idx = Path(spec_path_str), int(idx_str)
        spec = json.loads(spec_path.read_text())
        spec["keyframes"][idx]["smplx_pose"] = pose
        spec["dynamic_source"] = "manual"
        spec_path.write_text(json.dumps(spec, indent=2) + "\n")
        print(f"Captured full pose -> {spec_path.name} keyframe {idx}")


if __name__ == "__main__":
    main()
