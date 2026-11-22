"""Self-contained viser HTML of one Open-CHOIR reconstruction: hand + object + frame-0 scene points, gravity aligned.

  python scripts/choir/view_choir.py --video-dir $CHOIR_ROOT/output/VID --export-html out/VID.html   (sam3d env)

Inputs (all in CHOIR's metric OpenCV camera frame: x right, y down, z forward):
  stage3/final/optimized_meshes/{hand,object}/NNNN_{hand,mesh}.ply   posed meshes per frame
  stage3/final/optimized_hoi_contact_seq/obj_{canonical.obj,NNNNN.json}   object mesh + per-frame scale/R/t
  stage1/depth/env_depths/pc_0.ply   SAM3D's frame-0 point map (one point per pixel), colored from frame 0
  runs/choir_gravity/VID/gravity.json   GeoCalib up vector in the camera frame (vec3d)
The object mesh is uploaded once and moved per frame (pose from the JSONs, which are in PyTorch3D's camera frame,
flipped to OpenCV and checked against the posed PLYs);
the hand mesh is uploaded once and only its vertices change. Everything sits under a /world node rotated so
GeoCalib's up is +z (see scripts/daid/view_recon.py for why not scene.set_up_direction). The table plane is
fitted in the scene points (RANSAC, normal within 25 deg of up) to report its tilt from gravity as a check (it is
not used to align anything), to rescale the scene points so the frame-0 object rests on it (the point map's
depth scale differs from CHOIR's metric hand/object by ~15-20%) and to place the ground grid.
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "daid"))
from view_recon import free_port, write_embed_html  # noqa: E402
from table_fit import fit_table  # noqa: E402,F401  (view_fprecon uses vc.fit_table)
from pick_hand_side import read_mask  # noqa: E402

STORAGE = os.environ.get("PROJECT_STORAGE_ROOT", "/data/sky/projects/CounterfactualHOI")


def load_mesh(path):
    return trimesh.load(path, force="mesh", process=False)   # keep vertex order (posed PLYs share it)


def object_poses(seq_dir, canon):
    """Per-frame OpenCV-camera (R, t) with posed = R @ (s * v) + t, from obj_NNNNN.json. The JSON pose is in
    PyTorch3D's camera frame (x left, y up); the posed PLYs are flipped to OpenCV by diag(-1, -1, 1)."""
    js = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(seq_dir, "obj_[0-9]*.json")))]
    scale = np.broadcast_to(np.asarray(js[0]["scale"], float).reshape(-1), 3)
    flip = np.diag([-1.0, -1.0, 1.0])
    rot = np.array([flip @ np.asarray(j["rotation"], float).reshape(3, 3) for j in js])
    trans = np.array([flip @ np.asarray(j["translation"], float).reshape(3) for j in js])
    return canon * scale, rot, trans


def scene_points(vd, stride):
    """Frame-0 point map -> colored points, minus hand/object pixels and depth-edge flyers."""
    img = cv2.cvtColor(cv2.imread(os.path.join(vd, "inputs", "frames", "rgb", "0.png")), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    pts = np.asarray(trimesh.load(os.path.join(vd, "stage1", "depth", "env_depths", "pc_0.ply")).vertices,
                     np.float32).reshape(H, W, 3)
    fg = np.zeros((H, W), np.uint8)
    for m in glob.glob(os.path.join(vd, "inputs", "masks", "*", "0.png")):
        m = read_mask(m)
        if m.mean() < 0.06:          # skip spurious tracks, e.g. a "hand" mask on the whole table (pp_2)
            fg |= m.astype(np.uint8)
    fg = cv2.dilate(fg, np.ones((15, 15), np.uint8)) > 0
    z = pts[..., 2]
    zf = np.where(np.isfinite(z) & (z > 0), z, 0).astype(np.float32)
    edge = (cv2.dilate(zf, np.ones((5, 5))) - cv2.erode(zf, np.ones((5, 5)))) > 0.04 * np.maximum(zf, 1e-3)
    keep = (zf > 0.05) & ~fg & ~edge & np.isfinite(pts).all(-1)
    sel = np.zeros_like(keep)
    sel[::stride, ::stride] = True
    keep &= sel
    return pts[keep], img[keep], img, H, W


def load_choir(args):
    vd = os.path.abspath(args.video_dir)
    vid = os.path.basename(vd)
    fin = os.path.join(vd, "stage3", "final")
    hand_files = sorted(glob.glob(os.path.join(fin, "optimized_meshes", "hand", "*_hand.ply")))
    obj_files = sorted(glob.glob(os.path.join(fin, "optimized_meshes", "object", "*_mesh.ply")))
    if not hand_files or len(hand_files) != len(obj_files):
        sys.exit(f"{vid}: no complete stage3 output ({len(hand_files)} hand / {len(obj_files)} object meshes)")
    h0 = load_mesh(hand_files[0])
    hv = np.array([load_mesh(f).vertices for f in hand_files], np.float32)
    canon = load_mesh(os.path.join(fin, "optimized_hoi_contact_seq", "obj_canonical.obj"))
    sv, rot, trans = object_poses(os.path.join(fin, "optimized_hoi_contact_seq"), np.asarray(canon.vertices))
    # Check the poses against a few posed PLYs; fall back to per-frame posed vertices if they disagree.
    check = np.linspace(0, len(obj_files) - 1, 5).astype(int)
    err = max(np.abs(sv @ rot[i].T + trans[i] - load_mesh(obj_files[i]).vertices).max() for i in check)
    obj_v = None if err < 1e-3 else np.array([load_mesh(f).vertices for f in obj_files], np.float32)
    obj_mesh = trimesh.Trimesh(sv, canon.faces, process=False)

    grav_path = args.gravity or os.path.join(STORAGE, "runs", "choir_gravity", vid, "gravity.json")
    grav = json.load(open(grav_path))
    up = np.array(grav["vec3d"], float)
    up /= np.linalg.norm(up)
    R_world = Rotation.align_vectors([[0.0, 0.0, 1.0]], [up])[0]

    pts, cols, img0, H, W = scene_points(vd, args.scene_stride)
    table = fit_table(pts, up, np.random.default_rng(0))
    # The frame-0 point map has its own depth scale (SAM3D/MoGe); CHOIR's hand/object are metric through MANO.
    # Rescale the scene about the camera (rays, hence the reprojection, are unchanged) so the object rests on
    # the table at frame 0, and record how far it is from the table at the last frame (CHOIR depth drift).
    scene_scale, heights = 1.0, None
    if table is not None:
        n, c = table["normal"], table["normal"] @ table["point"]
        V0 = sv @ rot[0].T + trans[0]
        low = V0[V0 @ n <= np.percentile(V0 @ n, 3)]
        scene_scale = float(np.clip(np.median((low @ n) / c), 0.5, 2.0))   # ray scale that puts them on the plane
        pts = pts * scene_scale
        table["point"] = table["point"] * scene_scale
        c *= scene_scale
        heights = [float((sv @ rot[i].T + trans[i]).dot(n).min() - c) for i in (0, len(rot) - 1)]
    K = np.array(json.load(open(os.path.join(vd, "inputs", "intrinsics.json")))["intrinsics"], float)
    fps = cv2.VideoCapture(os.path.join(vd, "inputs", "video.mp4")).get(cv2.CAP_PROP_FPS) or 30.0
    return dict(vd=vd, vid=vid, n=len(hand_files), hv=hv, hf=np.asarray(h0.faces, np.uint32), obj_mesh=obj_mesh,
                rot=rot, trans=trans, obj_v=obj_v, pose_err=float(err), up=up, grav=grav, R_world=R_world,
                pts=pts, cols=cols, table=table, scene_scale=scene_scale, obj_heights=heights, H=H, W=W, fov_y=float(2 * np.arctan(H / (2 * K[1, 1]))),
                fps=float(fps), frames_dir=os.path.join(vd, "inputs", "frames", "rgb"))


def default_camera(d):
    """From above and behind the hand/object, on the real camera's side: reads like the video, with a level
    (gravity-aligned) horizon, but closer and looking down more, so the motion above the table is visible."""
    up = d["up"]
    fwd = np.array([0.0, 0, 1]) - up * up[2]                   # camera forward, made horizontal
    fwd /= np.linalg.norm(fwd)
    side = np.cross(fwd, up)                                   # camera x (right) in the horizontal plane
    act = np.concatenate([d["trans"], d["hv"].mean(1)]).mean(0)
    dist = float(np.linalg.norm(act))
    to_world = d["R_world"].apply
    return {"position": to_world(act + dist * (-0.6 * fwd + 0.15 * side + 0.5 * up)).tolist(),
            "look_at": to_world(act).tolist(), "up": [0.0, 0.0, 1.0], "fov": None}


def build_scene(server, d, args):
    q = d["R_world"].as_quat()                                   # xyzw
    server.scene.add_frame("/world", wxyz=(q[3], q[0], q[1], q[2]), show_axes=False)
    server.scene.add_light_ambient("/lights/ambient", intensity=0.7)
    server.scene.add_light_directional("/lights/key", intensity=1.5, wxyz=(0.92, -0.38, 0, 0))
    server.scene.add_point_cloud("/world/scene", points=d["pts"], colors=d["cols"], point_size=args.point_size)
    if d["table"] is not None:
        # Ground grid on the table height, centered under the action, in the gravity-aligned world (z up).
        act = d["R_world"].apply(np.concatenate([d["trans"], d["hv"].mean(1)]).mean(0))
        z = float(d["R_world"].apply(d["table"]["point"])[2])
        server.scene.add_grid("/grid", width=1.0, height=1.0, cell_size=0.05, position=(act[0], act[1], z - 0.002),
                              cell_color=(150, 150, 150), section_color=(110, 110, 110), section_size=0.25)
    obj_frame = server.scene.add_frame("/world/object", show_axes=False)
    om = d["obj_mesh"]
    if om.visual.kind in ("vertex", "texture") and d["obj_v"] is None:   # colored mesh (FoundationPose pipeline)
        obj = server.scene.add_mesh_trimesh("/world/object/mesh", mesh=om)
    else:
        obj = server.scene.add_mesh_simple("/world/object/mesh", vertices=np.asarray(om.vertices, np.float32),
                                           faces=np.asarray(om.faces, np.uint32), color=(235, 150, 60), side="double")
    hand = server.scene.add_mesh_simple("/world/hand", vertices=d["hv"][0], faces=d["hf"], color=(120, 140, 220),
                                        side="double")
    kw = dict(fov=d["fov_y"], aspect=d["W"] / d["H"], scale=args.frustum_scale, color=(40, 40, 40))

    def update(i):
        if d["obj_v"] is None:
            obj_frame.wxyz = Rotation.from_matrix(d["rot"][i]).as_quat()[[3, 0, 1, 2]]
            obj_frame.position = d["trans"][i]
        else:
            obj.vertices = d["obj_v"][i]
        hand.vertices = d["hv"][i]
        hand.visible = bool(d.get("hvalid", np.ones(d["n"], bool))[i])
        img = cv2.imread(os.path.join(d["frames_dir"], f"{i}.png"))
        if img is not None:
            w = args.image_width
            img = cv2.cvtColor(cv2.resize(img, (w, int(w * d["H"] / d["W"]))), cv2.COLOR_BGR2RGB)
            server.scene.add_camera_frustum("/world/camera", image=img, format="jpeg", jpeg_quality=70, **kw)
        else:
            server.scene.add_camera_frustum("/world/camera", **kw)

    return update


def summary(d):
    t = d["table"]
    return {"video_id": d["vid"], "frames": d["n"], "fps": d["fps"], "object_pose_max_err_m": d["pose_err"],
            "gravity_up_cam": d["up"].tolist(), "gravity_pitch_deg": d["grav"].get("pitch_deg"),
            "gravity_roll_deg": d["grav"].get("roll_deg"),
            "table_tilt_vs_gravity_deg": None if t is None else t["tilt_deg"],
            "table_inlier_frac": None if t is None else t["inlier_frac"], "scene_points": int(len(d["pts"])),
            "scene_scale": d["scene_scale"],
            "object_height_above_table_first_last_m": d["obj_heights"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True, help="$CHOIR_ROOT/output/<VID>")
    ap.add_argument("--export-html", required=True)
    ap.add_argument("--gravity", default=None, help="default: $PROJECT_STORAGE_ROOT/runs/choir_gravity/<VID>/gravity.json")
    ap.add_argument("--scene-stride", type=int, default=4, help="keep every Nth pixel of the frame-0 point map")
    ap.add_argument("--point-size", type=float, default=0.006)
    ap.add_argument("--image-width", type=int, default=320, help="frustum image width")
    ap.add_argument("--frustum-scale", type=float, default=0.08, help="camera frustum size (m)")
    args = ap.parse_args()

    d = load_choir(args)
    server = viser.ViserServer(host="127.0.0.1", port=free_port(), verbose=False)
    serializer = server.get_scene_serializer()                # create first so the static scene is recorded too
    update = build_scene(server, d, args)
    for i in range(d["n"]):
        update(i)
        serializer.insert_sleep(1.0 / d["fps"])
    data = serializer.serialize()
    write_embed_html(args.export_html, data, default_camera(d))
    info = summary(d)
    json.dump(info, open(os.path.splitext(args.export_html)[0] + ".json", "w"), indent=2)
    try:
        server.stop()
    except RuntimeError:          # viser 1.0.24 sometimes raises "Event loop is closed" on shutdown
        pass
    print(f"wrote {args.export_html} ({os.path.getsize(args.export_html) / 1e6:.1f} MB, {d['n']} frames @ "
          f"{d['fps']:.0f} fps); object pose err {d['pose_err'] * 1e3:.2f} mm"
          f"{' -> per-frame vertices' if d['obj_v'] is not None else ''}; table tilt vs gravity "
          f"{info['table_tilt_vs_gravity_deg'] if info['table_tilt_vs_gravity_deg'] is None else round(info['table_tilt_vs_gravity_deg'], 1)} deg")


if __name__ == "__main__":
    main()
