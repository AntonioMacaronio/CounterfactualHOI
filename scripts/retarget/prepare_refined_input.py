"""Build a Do-as-I-Do retargeting input dir (retargeting/launch.py --raw-dir) from an agent-refined reconstruction.

  python scripts/retarget/prepare_refined_input.py --refine-dir RUNS/refine_r5/VID --video-dir CHOIR/output/VID \
      --out RUNS/retarget_in/VID_r5      (fpose env; numpy + scipy only)

Retargeting stage 1 (retargeting/pipeline/process_dataset.py) reads, relative to --raw-dir: config.json
(object_names[0], anchor_hand, frame_number), <task>/all_hand_meshes.npz, gravity.json (vec3d = up in the camera
frame), obj_tracking_out/<obj>/combined_visualization/layout_camera_frame_optimized.json and
video_segmentation/masks/frame_<frame_number>_masks/<obj>/<obj>.obj (+ material.mtl / texture). It tracks the
wrist (joint 0 + MANO global orientation) and the fingertips (OpenPose joints 4, 8, 12, 16, 20); the MANO finger
pose is only carried along. So from the refined run:
  * vertices = export/hand_verts.npy, joints = MANO's regressed joints + fingertip vertices of those vertices, in
    OpenPose order;
  * MANO parameters: export/mano_params.npz when the refinement wrote one (rounds >= 6: the hand is MANO parameters,
    camera frame); else (round 5) the refinement's rigid rotation * Dyn-HaMR's (world-to-camera * root_orient), with
    Dyn-HaMR's finger pose and shape;
  * object = the refined per-frame object -> camera poses and the fprecon run's metric textured mesh (scale 1);
  * gravity = GeoCalib's up, or the table normal when they disagree by more than --max-gravity-deg.
Large files are symlinked; the task name is the output dir's basename.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

STORAGE = os.environ.get("PROJECT_STORAGE_ROOT", "/data/sky/projects/CounterfactualHOI")
TIPS = {"thumb": 745, "index": 317, "middle": 444, "ring": 556, "pinky": 673}
OPENPOSE = [0, 13, 14, 15, "thumb", 1, 2, 3, "index", 4, 5, 6, "middle", 10, 11, 12, "ring", 7, 8, 9, "pinky"]


def regressor(side):
    J = np.load(os.path.join(STORAGE, "models", "mano", f"joint_regressor_{side}.npz"))["J"]
    reg = np.zeros((21, 778))
    for r, o in enumerate(OPENPOSE):
        if isinstance(o, str):
            reg[r, TIPS[o]] = 1
        else:
            reg[r] = J[o]
    return reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refine-dir", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--object-id", default=None, help="default: object_<n> from the clip name")
    ap.add_argument("--max-gravity-deg", type=float, default=10.0)
    args = ap.parse_args()
    rd, vd, out = (os.path.abspath(p) for p in (args.refine_dir, args.video_dir, args.out))
    vid = os.path.basename(vd)
    ex = os.path.join(rd, "export")
    run = os.path.realpath(os.path.join(ex, "object_mesh.obj"))
    run = os.path.dirname(run)                                      # the fprecon run the export links to
    side = json.load(open(os.path.join(run, "hand_scale.json")))["side"]
    obj = args.object_id or f"object_{vid.split('_')[-1]}"
    task = os.path.basename(out)
    os.makedirs(os.path.join(out, task), exist_ok=True)

    # ---- hand
    verts = np.load(os.path.join(ex, "hand_verts.npy")).astype(np.float64)        # (N, 778, 3) camera frame
    N = len(verts)
    rig = None if os.path.exists(os.path.join(ex, "mano_params.npz")) else np.load(os.path.join(ex, "hand_rigid.npz"))
    mdir = os.path.join(vd, "stage1", "hand", "mano_params", side)
    if not os.path.isdir(mdir):
        mdir = os.path.join(vd, "stage1", "hand", "mano_params")
    hdir = os.path.join(vd, "stage1", "hand", "hand_meshes", side)
    if not os.path.isdir(hdir):
        hdir = os.path.join(vd, "stage1", "hand", "hand_meshes")
    valid = np.array([os.path.exists(os.path.join(hdir, f"hand_{i:05d}.obj")) and
                      os.path.exists(os.path.join(mdir, f"{i:05d}.json")) for i in range(N)])
    rot, pose, betas = np.zeros((N, 3)), np.zeros((N, 45)), np.zeros((N, 10))
    ok = np.nonzero(valid)[0]
    if rig is None:                                                 # MANO parameters (rounds >= 6)
        mp = np.load(os.path.join(ex, "mano_params.npz"))
        rot, pose, betas = mp["global_orient"][:N].astype(float), mp["hand_pose"][:N].astype(float), np.tile(mp["betas"], (N, 1))
    for i in range(N if rig is not None else 0):
        j = i if valid[i] else ok[np.argmin(np.abs(ok - i))]
        m = json.load(open(os.path.join(mdir, f"{j:05d}.json")))
        R_cam = np.array(m["T_w2c"], float)[:3, :3] @ Rotation.from_rotvec(np.array(m["root_orient"], float)).as_matrix()
        rot[i] = (Rotation.from_rotvec(rig["hand_rot"][i]) * Rotation.from_matrix(R_cam)).as_rotvec()
        pose[i], betas[i] = np.array(m["pose"], float).reshape(45), np.array(m["betas"], float)
    joints = np.einsum("jv,nvi->nji", regressor(side), verts)
    faces = None                                                    # the hand tracker's MANO faces (any frame's mesh)
    with open(os.path.join(hdir, f"hand_{ok[0]:05d}.obj")) as f:
        faces = np.array([[int(t.split("/")[0]) - 1 for t in ln.split()[1:4]] for ln in f if ln.startswith("f ")], np.int32)
    hands = {}
    for s in ("left", "right"):
        mine = s == side
        hands[f"{s}_vertices"] = verts if mine else np.zeros_like(verts)
        hands[f"{s}_joints"] = joints if mine else np.zeros_like(joints)
        hands[f"{s}_valid"] = valid if mine else np.zeros(N, bool)
        hands[f"{s}_rot"] = rot if mine else np.zeros((N, 3))
        hands[f"{s}_trans"] = joints[:, 0] if mine else np.zeros((N, 3))
        hands[f"{s}_hand_pose"] = pose if mine else np.zeros((N, 45))
        hands[f"{s}_betas"] = betas if mine else np.zeros((N, 10))
        hands[f"{s}_faces"] = faces
    np.savez(os.path.join(out, task, "all_hand_meshes.npz"), **hands)
    json.dump({"frame_number": 0, "object_names": [obj], "anchor_hand": side}, open(os.path.join(out, "config.json"), "w"), indent=2)

    # ---- object: refined poses (mesh -> camera), the metric textured mesh at scale 1
    poses = np.load(os.path.join(ex, "object_poses.npy"))[:N]
    objs = [{"frame_idx": int(i), "local_to_scene": {
        "translation_camera_frame": poses[i, :3, 3].tolist(),
        "quat_wxyz_camera_frame": Rotation.from_matrix(poses[i, :3, :3]).as_quat()[[3, 0, 1, 2]].tolist()}}
        for i in range(len(poses))]
    cv = os.path.join(out, "obj_tracking_out", obj, "combined_visualization")
    os.makedirs(cv, exist_ok=True)
    json.dump({"objects": objs, "translation_scale_optimization": {"mesh_scale": 1.0}},
              open(os.path.join(cv, "layout_camera_frame_optimized.json"), "w"))
    md = os.path.join(out, "video_segmentation", "masks", "frame_000000_masks", obj)
    os.makedirs(md, exist_ok=True)
    for src, dst in (("object_mesh.obj", f"{obj}.obj"), ("material.mtl", "material.mtl"), ("material_0.png", "material_0.png")):
        if os.path.exists(os.path.join(run, src)) and not os.path.lexists(os.path.join(md, dst)):
            os.symlink(os.path.join(run, src), os.path.join(md, dst))

    # ---- gravity
    grav = json.load(open(os.path.join(STORAGE, "runs", "choir_gravity", vid, "gravity.json")))
    g = np.array(grav["vec3d"], float)
    g /= np.linalg.norm(g)
    tj = os.path.join(ex, "table.json")
    if os.path.exists(tj):
        n = np.array(json.load(open(tj))["normal_up"], float)
        ang = float(np.degrees(np.arccos(np.clip(abs(g @ n), -1, 1))))
        grav = dict(grav, vec3d=n.tolist(), source=f"table_normal (GeoCalib off by {ang:.1f} deg)") if ang > args.max_gravity_deg \
            else dict(grav, source=f"geocalib ({ang:.1f} deg from table normal)")
    json.dump(grav, open(os.path.join(out, "gravity.json"), "w"), indent=2)
    print(f"{vid}: task={task} side={side} frames={N} valid={int(valid.sum())} object={obj} gravity={grav['source']} -> {out}")


if __name__ == "__main__":
    sys.exit(main())
