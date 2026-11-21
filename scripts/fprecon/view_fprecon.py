"""Self-contained viser HTML of one FoundationPose + Depth Pro 2 reconstruction, gravity aligned (step 4).

  python scripts/fprecon/view_fprecon.py --video-dir $CHOIR_ROOT/output/VID --run-dir $PROJECT_STORAGE_ROOT/runs/fprecon/VID \
      --export-html out/VID.html      (sam3d env)

Same viewer as scripts/choir/view_choir.py (scene, grid, camera, GeoCalib up), fed from this pipeline: the object
is RUN/object_mesh.obj (metric, SAM3D-textured; .ply vertex-colored for CHOIR's glb) moved by RUN/object_poses.npy; the hand is Dyn-HaMR's, moved
per frame onto the depth map (hand_scale.corrected_hands); the scene points are the hand-scaled Depth Pro 2 depth
of frame 0, so hand, object and table share one metric depth and need no rescaling.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "choir"))
sys.path.insert(0, HERE)
import view_choir as vc  # noqa: E402
from hand_scale import corrected_hands  # noqa: E402
from masks import hand_mask, object_mask, tracked_mesh  # noqa: E402
from view_recon import free_port, write_embed_html  # noqa: E402


def depth_scene_points(vd, rd, side, depth0, K, stride):
    """Frame-0 metric depth -> colored points, minus hand/object pixels and depth-edge flyers."""
    img = cv2.cvtColor(cv2.imread(os.path.join(vd, "inputs", "frames", "rgb", "0.png")), cv2.COLOR_BGR2RGB)
    H, W = depth0.shape
    fg = np.zeros((H, W), np.uint8)
    for m in (object_mask(vd, rd, 0), hand_mask(vd, rd, 0, side)):
        if m is not None and m.mean() < 0.2:
            fg |= m.astype(np.uint8)
    fg = cv2.dilate(fg, np.ones((15, 15), np.uint8)) > 0
    z = depth0.astype(np.float32)
    edge = (cv2.dilate(z, np.ones((5, 5))) - cv2.erode(z, np.ones((5, 5)))) > 0.04 * np.maximum(z, 1e-3)
    keep = (z > 0.05) & (z < 8.0) & ~fg & ~edge
    sel = np.zeros_like(keep)
    sel[::stride, ::stride] = True
    keep &= sel
    v, u = np.nonzero(keep)
    pts = np.stack([(u - K[0, 2]) / K[0, 0] * z[keep], (v - K[1, 2]) / K[1, 1] * z[keep], z[keep]], 1)
    return pts.astype(np.float32), img[keep], H, W


def load_fprecon(args):
    vd, rd = os.path.abspath(args.video_dir), os.path.abspath(args.run_dir)
    vid = os.path.basename(vd)
    K = np.array(json.load(open(os.path.join(vd, "inputs", "intrinsics.json")))["intrinsics"], float)
    hs = json.load(open(os.path.join(rd, "hand_scale.json")))
    scale = hs["scale"]
    qc = json.load(open(os.path.join(rd, "fpose_qc.json")))
    poses = np.load(os.path.join(rd, "object_poses.npy"))
    mesh = trimesh.load(tracked_mesh(rd), force="mesh", process=False)
    hv, hf, hvalid, _ = corrected_hands(vd, rd)
    n = min(len(poses), len(hv))
    poses, hv, hvalid = poses[:n], hv[:n], hvalid[:n]
    ok = np.nonzero(hvalid)[0]
    for i in np.nonzero(~hvalid)[0]:                      # hidden in the viewer; nearest valid pose keeps the
        hv[i] = hv[ok[np.argmin(np.abs(ok - i))]]        # default-camera centroid sensible
    grav = json.load(open(args.gravity or os.path.join(vc.STORAGE, "runs", "choir_gravity", vid, "gravity.json")))
    up = np.array(grav["vec3d"], float)
    up /= np.linalg.norm(up)
    depth0 = np.load(os.path.join(rd, "depth_dp2.npy"), mmap_mode="r")[0].astype(np.float32) * scale
    pts, cols, H, W = depth_scene_points(vd, rd, hs["side"], depth0, K, args.scene_stride)
    table = vc.fit_table(pts, up, np.random.default_rng(0))
    rot, trans = poses[:, :3, :3], poses[:, :3, 3]
    heights = None
    if table is not None:
        nrm, c = table["normal"], table["normal"] @ table["point"]
        heights = [float((mesh.vertices @ rot[i].T + trans[i]).dot(nrm).min() - c) for i in (0, n - 1)]
    fps = cv2.VideoCapture(os.path.join(vd, "inputs", "video.mp4")).get(cv2.CAP_PROP_FPS) or 30.0
    return dict(vd=vd, vid=vid, n=n, hv=hv.astype(np.float32), hf=hf, hvalid=hvalid, obj_mesh=mesh, rot=rot,
                trans=trans, obj_v=None, pose_err=0.0, up=up, grav=grav,
                R_world=Rotation.align_vectors([[0.0, 0.0, 1.0]], [up])[0], pts=pts, cols=cols, table=table,
                scene_scale=1.0, obj_heights=heights, H=H, W=W, fov_y=float(2 * np.arctan(H / (2 * K[1, 1]))),
                fps=float(fps), frames_dir=os.path.join(vd, "inputs", "frames", "rgb"), depth_scale=scale, fpose_qc=qc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True, help="$CHOIR_ROOT/output/<VID> (frames, masks, intrinsics, hands)")
    ap.add_argument("--run-dir", required=True, help="$PROJECT_STORAGE_ROOT/runs/fprecon/<VID>")
    ap.add_argument("--export-html", required=True)
    ap.add_argument("--gravity", default=None)
    ap.add_argument("--scene-stride", type=int, default=4)
    ap.add_argument("--point-size", type=float, default=0.006)
    ap.add_argument("--image-width", type=int, default=320)
    ap.add_argument("--frustum-scale", type=float, default=0.08)
    args = ap.parse_args()

    d = load_fprecon(args)
    server = viser.ViserServer(host="127.0.0.1", port=free_port(), verbose=False)
    serializer = server.get_scene_serializer()
    update = vc.build_scene(server, d, args)
    for i in range(d["n"]):
        update(i)
        serializer.insert_sleep(1.0 / d["fps"])
    data = serializer.serialize()
    write_embed_html(args.export_html, data, vc.default_camera(d))
    info = vc.summary(d)
    q = d["fpose_qc"]
    info.update(method="foundationpose+depthpro2", depth_scale_from_hand=d["depth_scale"],
                object_size_m=q["object_size_m"], iou_first_last=q["iou_first_last"],
                iou_visible_frames_mean=q["iou_visible_frames_mean"], reregistered_frames=len(q["reregistered_frames"]))
    json.dump(info, open(os.path.splitext(args.export_html)[0] + ".json", "w"), indent=2)
    try:
        server.stop()
    except RuntimeError:
        pass
    h = d["obj_heights"]
    print(f"wrote {args.export_html} ({os.path.getsize(args.export_html) / 1e6:.1f} MB, {d['n']} frames); "
          f"table tilt vs gravity {info['table_tilt_vs_gravity_deg']:.1f} deg; object height above table "
          f"first/last {100 * h[0]:.1f}/{100 * h[1]:.1f} cm; IoU first/last {q['iou_first_last'][0]:.2f}/{q['iou_first_last'][1]:.2f}")


if __name__ == "__main__":
    main()
