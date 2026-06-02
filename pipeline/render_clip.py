#!/usr/bin/env python3
"""BlenderProc entry script: render one gesture clip from a render config.

Run via BlenderProc's bundled Python, NOT plain python3:

    blenderproc run render_clip.py \
        --config render_configs/greet-01__v000.json \
        --spec   motion_specs/greet-01.json \
        --smplx-model /path/to/SMPLX_NEUTRAL.npz \
        --out    clips/

It reads ONE render config (camera/lighting/background/occlusion/avatar) and the
referenced motion spec (keyframed SMPL-X poses), drives an SMPL-X avatar through
the motion, and renders the configured outputs to <out>/<output_clip>.

═══════════════════════════════════════════════════════════════════════════
VERIFY-ON-YOUR-RIG NOTES (this could not be executed in the authoring env):

  * The scene setup (camera / lighting / world / renderer) is written against
    the documented BlenderProc API and should be close, but confirm against
    your installed BlenderProc version.

  * The SMPL-X body load + pose application (`load_avatar` / `apply_pose`) is
    the part most likely to need adjustment: it depends on which SMPL-X
    integration you use (the smplx Blender addon vs. a custom armature). Those
    two functions are isolated so you can adapt them without touching the rest.

  * Unauthored poses are a HARD STOP: if the motion spec's keyframes still have
    `smplx_pose: null`, the script exits with an error instead of rendering a
    rest-pose clip that would silently poison training data.
═══════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

# BlenderProc must be imported first; it bootstraps Blender before anything else.
import blenderproc as bproc  # noqa: E402  (must precede other imports)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one gesture clip.")
    parser.add_argument("--config", type=Path, required=True, help="render_configs/<id>__v<NNN>.json")
    parser.add_argument("--spec", type=Path, required=True, help="motion_specs/<id>.json")
    parser.add_argument("--smplx-model", type=Path, required=True, help="SMPL-X model file (.npz/.pkl)")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for the clip")
    parser.add_argument("--hdri-dir", type=Path, default=None, help="Optional HDRI library for backgrounds")
    return parser.parse_args()


def require_authored_poses(spec: dict) -> None:
    """Hard-stop if the motion spec's keyframes haven't been authored yet."""
    missing = [kf for kf in spec["keyframes"] if kf.get("smplx_pose") is None]
    if missing:
        raise SystemExit(
            f"error: motion spec '{spec['id']}' has {len(missing)}/{len(spec['keyframes'])} "
            f"keyframes with smplx_pose=null. Author the poses before rendering — "
            f"rendering now would produce rest-pose clips and poison training data."
        )


def camera_matrix(distance: float, azimuth_deg: float, elevation_deg: float, target=(0.0, 0.0, 1.0)):
    """World matrix for a camera orbiting `target` at the given spherical coords."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    tx, ty, tz = target
    cam_pos = np.array([
        tx + distance * math.cos(el) * math.sin(az),
        ty - distance * math.cos(el) * math.cos(az),
        tz + distance * math.sin(el),
    ])
    forward = np.array(target) - cam_pos
    forward /= np.linalg.norm(forward)
    return bproc.math.build_transformation_mat(cam_pos, bproc.camera.rotation_from_forward_vec(forward))


def setup_world(cfg: dict, hdri_dir: Path | None) -> None:
    """Background + ambient light. Prefer an HDRI per the config's background tag."""
    if hdri_dir is not None:
        hdri = hdri_dir / f"{cfg['background']}.hdr"
        if hdri.exists():
            bproc.world.set_world_background_hdr_img(str(hdri))
            return
    # Fallback: flat ambient keyed off the lighting color temp so renders aren't black.
    temp = cfg["lighting"]["color_temp_k"]
    warmth = max(0.0, min(1.0, (6500 - temp) / 3300))
    bproc.world.set_world_background([1.0, 0.85 + 0.15 * (1 - warmth), 0.7 + 0.3 * (1 - warmth)], strength=0.3)


def setup_lighting(cfg: dict) -> None:
    light = bproc.types.Light()
    light.set_type("SUN" if cfg["lighting"]["preset"] in {"direct_sun", "golden_hour"} else "AREA")
    light.set_energy(cfg["lighting"]["intensity"] * (5.0 if light.get_type() == "SUN" else 200.0))
    light.set_location([2.0, -2.0, 3.0])


def load_avatar(smplx_model: Path):
    """Load an SMPL-X body. ADAPT to your SMPL-X integration (see header notes)."""
    # BlenderProc's AMASS loader pulls in the SMPL-X body mesh + armature. If you
    # drive poses through a different path, swap this out — apply_pose() below is
    # the only other place that touches the avatar.
    return bproc.loader.load_AMASS(
        str(smplx_model.parent),
        sub_dataset_id="CMU",
        body_model_path=str(smplx_model),
    )[0]


def apply_pose(avatar, smplx_pose: dict, frame: int) -> None:
    """Set the avatar's SMPL-X pose for one frame and keyframe it. ADAPT as needed.

    Expected smplx_pose schema (axis-angle radians):
        global_orient    [3]
        body_pose        [63]   (21 body joints × 3)
        left_hand_pose   [45]   (15 finger joints × 3)
        right_hand_pose  [45]
    """
    pose_vec = (
        smplx_pose["global_orient"]
        + smplx_pose["body_pose"]
        + smplx_pose["left_hand_pose"]
        + smplx_pose["right_hand_pose"]
    )
    # Placeholder hook: route pose_vec into your SMPL-X armature, then keyframe.
    # e.g. set armature bone rotations from pose_vec, or call the smplx addon API.
    avatar.set_cp("smplx_pose", pose_vec)  # custom property; replace with real rig drive
    bproc.utility.insert_keyframe(avatar, "location", frame)


def interpolate_keyframes(keyframes: list[dict], num_frames: int) -> list[dict]:
    """Map authored keyframes (normalized t in [0,1]) onto `num_frames` render frames.

    Nearest-keyframe hold for now; swap to slerp on the joint rotations once the
    pose schema is filled in and the visual result is checked on the rig.
    """
    out = []
    for f in range(num_frames):
        t = f / max(1, num_frames - 1)
        nearest = min(keyframes, key=lambda kf: abs(kf["t"] - t))
        out.append(nearest["smplx_pose"])
    return out


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text())
    spec = json.loads(args.spec.read_text())
    require_authored_poses(spec)

    bproc.init()
    setup_world(cfg, args.hdri_dir)
    setup_lighting(cfg)
    avatar = load_avatar(args.smplx_model)

    num_frames = cfg["num_frames"]
    poses = interpolate_keyframes(spec["keyframes"], num_frames)

    cam2world = camera_matrix(
        cfg["camera"]["distance_m"], cfg["camera"]["azimuth_deg"], cfg["camera"]["elevation_deg"]
    )
    bproc.camera.set_intrinsics_from_blender_params(
        lens=cfg["camera"]["focal_length_mm"], lens_unit="MILLIMETERS"
    )

    for frame, pose in enumerate(poses):
        apply_pose(avatar, pose, frame)
        bproc.camera.add_camera_pose(cam2world, frame=frame)  # static camera, per-frame keyframe

    if "depth" in cfg["outputs"]:
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
    if "segmentation" in cfg["outputs"]:
        bproc.renderer.enable_segmentation_output(map_by=["instance"])

    data = bproc.renderer.render()

    args.out.mkdir(parents=True, exist_ok=True)
    write_clip(args.out / cfg["output_clip"], data["colors"], cfg["fps"])
    print(f"Rendered {num_frames} frames -> {args.out / cfg['output_clip']}")


def write_clip(path: Path, frames, fps: int) -> None:
    """Encode RGB frames to mp4. Uses imageio (needs imageio-ffmpeg in Blender's
    Python); falls back to a PNG sequence the extractor can also be pointed at."""
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8) as w:
            for f in frames:
                w.append_data(np.asarray(f))
    except Exception as exc:  # noqa: BLE001 — fall back rather than lose the render
        seq_dir = path.with_suffix("")
        seq_dir.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio
        for i, f in enumerate(frames):
            imageio.imwrite(str(seq_dir / f"{i:04d}.png"), np.asarray(f))
        print(f"  mp4 encode unavailable ({exc}); wrote PNG sequence -> {seq_dir}")


if __name__ == "__main__":
    main()
