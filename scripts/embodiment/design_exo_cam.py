"""Where should the exo camera stand? Score fixed viewpoints around the workspace on every clip.

  python scripts/embodiment/design_exo_cam.py [--dataset DATA] [--stride 3] [--workers 10]
  -> $PROJECT_STORAGE_ROOT/runs/embodiment/exo_cam/<tag>/candidates.json    (venv: $PROJECT_STORAGE_ROOT/venvs/arm)

A candidate looks at the middle of the workspace (the centre of all object paths, 5 cm above the table) from
  azimuth    the compass direction the camera stands in, seen from that point: 0 = +x, 90 = +y (the far side of the
             table, facing the source video's camera), 270 = -y (where the video camera stood)
  elevation  degrees above the table
  distance   metres from the look-at point
with the image's up along the world's up. The scene is the floating-wrist Sharpa hand with a generic forearm behind
it (no particular arm), the clip's object and the table.

Measured on every `stride`-th frame of every clip:
  object / fingertips measured   share of 96 points on the object's surface (ceiling about 50%) and of the five
                                 fingertips that give a valid stereo depth: seen by both imagers, in range, not
                                 hidden by the hand, the forearm or the object
  window                         the smallest window of the stream (106k x 60k pixels, k = 3..8) that contains the
                                 object and the fingertips in every frame of every clip, so that the policy image
                                 (106 x 60) can show that window instead of the whole 848 x 480 frame
  object size                    mean number of policy-image pixels on the object, for that window
  depth noise                    1-sigma depth error at the look-at point: z^2 * sigma_d / (f * baseline)
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

import common as C
from cameras import Camera
from preview import look_at

N_OBJ = 96
TOL = np.r_[np.full(N_OBJ, 0.003), np.full(5, 0.012)]   # object surface points; fingertip centres sit ~1 cm inside the pad


def workspace_centre(clips):
    P = np.concatenate([c["obj_pos"] for c in clips])
    lo, hi = P.min(0), P.max(0)
    return np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, C.TABLE_Z + 0.05])


def candidate_pose(centre, az, el, dist):
    a, e = np.radians(az), np.radians(el)
    eye = centre + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    return look_at(eye, centre)


def _worker(job):
    clip, cands, cam_cfg, sensor, stride, centre = job
    from scene import Scene
    import trimesh
    sc = Scene("floating", clip, forearm=True)
    m, d = sc.model, sc.data
    cam = Camera("exo", cam_cfg, sensor)
    cam.roi = (0, 0, cam.W, cam.H)                               # score the whole frame; the window is chosen after
    mesh = trimesh.load(Path(clip["mesh_dir"]) / "visual.obj", force="mesh", process=False, skip_materials=True)
    pts, _ = trimesh.sample.sample_surface(mesh, N_OBJ, seed=0)
    pts = np.asarray(pts) * clip.get("mesh_scale", 1.0)
    Q, OP, OR = clip["joint_pos"], clip["obj_pos"], clip["obj_rot"]
    frames = np.arange(0, len(Q), stride)
    poses = [candidate_pose(centre, c["azimuth"], c["elevation"], c["distance"]) for c in cands]
    nC, nF = len(cands), len(frames)
    vo, vt = np.zeros((nC, nF), np.float32), np.zeros((nC, nF), np.float32)
    box = np.zeros((nC, 4))                                      # u_min, u_max, v_min, v_max over the clip
    box[:, [0, 2]], box[:, [1, 3]] = np.inf, -np.inf
    area = np.zeros((nC, nF), np.float32)                        # object's bounding box area in stream pixels
    for fi, t in enumerate(frames):
        sc.set_floating(Q[t])
        sc.set_object(OP[t], OR[t])
        sc.fk()
        P = np.concatenate([pts @ OR[t].T + OP[t], sc.tips()])
        for ci, T in enumerate(poses):
            ok = cam.measures(m, d, T, P, tol=TOL)
            vo[ci, fi], vt[ci, fi] = ok[:N_OBJ].mean(), ok[N_OBJ:].mean()
            u, v, z = cam.project((P - T[:3, 3]) @ T[:3, :3])
            box[ci] = [min(box[ci, 0], u.min()), max(box[ci, 1], u.max()), min(box[ci, 2], v.min()), max(box[ci, 3], v.max())]
            area[ci, fi] = (u[:N_OBJ].max() - u[:N_OBJ].min()) * (v[:N_OBJ].max() - v[:N_OBJ].min())
    return clip["name"], vo, vt, box, area


def choose_window(box, W, H, margin=24):
    """Smallest 106k x 60k window (k = 3..8) holding the box [u0, u1, v0, v1] plus a margin; None if none does."""
    u0, u1, v0, v1 = box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin
    for k in range(3, 9):
        w, h = 106 * k, 60 * k
        if u1 - u0 > w or v1 - v0 > h:
            continue
        x0 = int(round(np.clip((u0 + u1) / 2 - w / 2, 0, W - w)))
        y0 = int(round(np.clip((v0 + v1) / 2 - h / 2, 0, H - h)))
        if x0 <= u0 and x0 + w >= u1 and y0 <= v0 and y0 + h >= v1:
            return [x0, y0, w, h], k
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(C.STORAGE / "runs" / "flashdex" / "data_r10_flat"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--azimuth", type=float, nargs="*", default=list(range(0, 360, 15)))
    ap.add_argument("--elevation", type=float, nargs="*", default=[30, 40, 50, 60, 75])
    ap.add_argument("--distance", type=float, nargs="*", default=[0.6, 0.75, 0.9])
    args = ap.parse_args()
    cams = C.load_cameras()
    cam_cfg, sensor = cams["exo"], cams["sensors"][cams["exo"]["sensor"]]
    clips = C.load_pack(args.dataset)
    centre = workspace_centre(clips)
    out = C.RUNS / "exo_cam" / (args.tag or Path(args.dataset).name)
    out.mkdir(parents=True, exist_ok=True)
    cands = [{"azimuth": a, "elevation": e, "distance": d} for a, e, d in itertools.product(args.azimuth, args.elevation, args.distance)]
    jobs = [({k: c[k] for k in ("name", "mesh_dir", "mesh_scale", "joint_pos", "obj_pos", "obj_rot")}, cands, cam_cfg, sensor,
             args.stride, centre) for c in clips]
    with mp.get_context("fork").Pool(min(args.workers, len(jobs))) as pool:
        res = pool.map(_worker, jobs)
    cam = Camera("exo", cam_cfg, sensor)
    sig = cam.noise.get("disparity_sigma_px", 0.05)
    rows = []
    for ci, c in enumerate(cands):
        vo = np.concatenate([r[1][ci] for r in res])
        vt = np.concatenate([r[2][ci] for r in res])
        boxes = np.array([r[3][ci] for r in res])
        box = [boxes[:, 0].min(), boxes[:, 1].max(), boxes[:, 2].min(), boxes[:, 3].max()]
        roi, k = choose_window(box, cam.W, cam.H)
        area = np.concatenate([r[4][ci] for r in res])
        T = candidate_pose(centre, c["azimuth"], c["elevation"], c["distance"])
        z = c["distance"]
        row = dict(c, object_measured=float(vo.mean() * 100), object_any=float((vo >= 0.1).mean() * 100),
                   object_min_clip=float(min(r[1][ci].mean() for r in res) * 100),
                   tips_measured=float(vt.mean() * 100), in_frame=bool(box[0] >= 0 and box[1] <= cam.W and box[2] >= 0 and box[3] <= cam.H),
                   window=roi, window_k=k, object_px=float(area.mean() / (k * k)) if k else 0.0,
                   depth_sigma_mm=float(z * z * sig / (cam.fx * cam.baseline) * 1000),
                   pos=T[:3, 3].round(4).tolist(), x_axis=T[:3, 0].round(5).tolist(), z_axis=T[:3, 2].round(5).tolist())
        rows.append(row)
    rows.sort(key=lambda r: -(r["object_measured"] + r["tips_measured"] / 2))
    json.dump({"dataset": args.dataset, "centre": centre.tolist(), "stride": args.stride, "candidates": rows},
              open(out / "candidates.json", "w"), indent=1)
    print(f"look-at {centre.round(3)}; {len(rows)} candidates -> {out}/candidates.json")
    for r in rows[:12]:
        print(f"  az {r['azimuth']:5.0f} el {r['elevation']:3.0f} d {r['distance']:.2f} | object {r['object_measured']:5.1f}% "
              f"(worst clip {r['object_min_clip']:5.1f}%, any {r['object_any']:5.1f}%) tips {r['tips_measured']:5.1f}% | window "
              f"{r['window']} (k={r['window_k']}) object {r['object_px']:.0f} px | sigma {r['depth_sigma_mm']:.1f} mm")


if __name__ == "__main__":
    main()
