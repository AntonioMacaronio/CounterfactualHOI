"""Where on the wrist should the D405 sit? Score candidate poses on every clip.

  python scripts/embodiment/design_wrist_cam.py --dataset $PROJECT_STORAGE_ROOT/runs/flashdex/data_r10_flat \
      [--rollouts RUN/evals/step<N>] [--stride 3] [--workers 10]         (venv: $PROJECT_STORAGE_ROOT/venvs/arm)
  -> $PROJECT_STORAGE_ROOT/runs/embodiment/wrist_cam/<tag>/candidates.json, summary.md

A candidate is the D405's housing beside the wrist, looking along the fingers:
  side   where around the hand's long axis the housing sits, in degrees from the palm side (volar, +x of the palm)
         towards the thumb side: 0 palm, 90 thumb (radial), 180 back of the hand (dorsal), 270 little finger (ulnar)
  r      distance of the housing's centre from the hand's axis [mm]
  z      position of the housing's centre along the hand, 0 = the palm's wrist end, negative towards the arm [mm]
  tilt   the view direction leans this far towards the hand's axis [deg]
  pan    and this far towards the palm side (only for housings that are not on the palm side or the back) [deg]
The image's "up" points at the hand, so the hand hangs from the top edge of every candidate's image.

Measured on every `stride`-th frame of every clip (the floating-wrist Sharpa hand, the clip's object, the table):
  what the camera measures  the share of 96 points on the object's surface, and of the five fingertips, that give a
                            valid stereo depth: inside the image of both imagers, inside the depth range, and
                            hidden from neither imager by the hand or the object. Averaged over the frames in which
                            the object is within 25 cm of the palm. About half of an object's surface faces away
                            from any camera, so 50% is the ceiling for the object.
  clearances                housing to table (lowest corner above the table plane), housing to fingers and thumb,
                            housing to object: the minimum over all frames, and the share of frames under a limit.
The housing is the bare D405 (42 x 42 x 23 mm); a bracket adds a few mm, which is what the limits are for.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
from pathlib import Path

import mujoco
import numpy as np

import common as C
from cameras import GROUPS_VISIBLE, Camera

CORNERS = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
SIDES = {0: "palm", 45: "palm-thumb", 90: "thumb", 180: "back", 270: "little finger", 315: "palm-little finger"}
NEAR_OBJECT = 0.25
TOL = np.r_[np.full(96, 0.003), np.full(5, 0.012)]      # object surface points; fingertip centres sit ~1 cm inside the pad
TABLE_LIMIT, FINGER_LIMIT, OBJECT_LIMIT = 0.008, 0.006, 0.006


def candidate_pose(side_deg, r, z, tilt_deg, housing, pan_deg=0.0):
    """-> (T_palm_optical, T_palm_housing-centre). The housing box is aligned with the optical frame."""
    a, t, pn = np.radians(side_deg), np.radians(tilt_deg), np.radians(pan_deg)
    s = np.array([np.cos(a), np.sin(a), 0.0])                    # from the hand's axis out to the camera
    ez = np.array([0, 0, 1.0])
    ep = np.array([1.0, 0, 0]) - s[0] * s                        # towards the palm side, across s
    ep = ep / np.linalg.norm(ep) if np.linalg.norm(ep) > 1e-6 else np.zeros(3)
    view = np.cos(t) * np.cos(pn) * ez - np.sin(t) * s + np.cos(t) * np.sin(pn) * ep
    view /= np.linalg.norm(view)
    down = s - (s @ view) * view                                 # image "down" = away from the hand
    down /= np.linalg.norm(down)
    R = np.stack([np.cross(down, view), down, view], 1)
    centre = r * s + z * ez
    o = centre - R @ (np.asarray(housing["centre_mm"]) / 1000)   # the left imager (depth origin)
    return C.make_T(R, o), C.make_T(R, centre)


def configured_pose(cam_cfg, housing):
    """The pose in cameras.yaml as (T_palm_optical, T_palm_housing-centre)."""
    T = C.cfg_T(cam_cfg["pose"])
    return T, C.make_T(T[:3, :3], T[:3, 3] + T[:3, :3] @ (np.asarray(housing["centre_mm"]) / 1000))


def make_candidates(sides, rs, zs, tilts, pans):
    """Housings that would sit inside the wrist or the flange are left out: r >= 48 mm beside the palm's end, 54 mm
    beside the Sharpa flange (z < -10 mm), 60 mm beside the arm's flange (z < -30 mm)."""
    out = []
    for s, r, z, t, p in itertools.product(sides, rs, zs, tilts, pans):
        if p and int(s) % 180 == 0:                              # no "towards the palm" for the palm side or the back
            continue
        if r < (60 if z < -30 else 54 if z < -10 else 48):
            continue
        out.append({"side": s, "r_mm": r, "z_mm": z, "tilt": t, "pan": p})
    return out


def _worker(job):
    clip, cands, cam_cfg, sensor, stride, zmins, source = job
    from scene import Scene
    import trimesh
    sc = Scene("floating", clip)
    m, d = sc.model, sc.data
    cam = Camera("wrist", cam_cfg, sensor)
    size = np.asarray(sensor["housing"]["size_mm"]) / 1000
    mesh = trimesh.load(Path(clip["mesh_dir"]) / "visual.obj", force="mesh", process=False, skip_materials=True)
    pts_obj, _ = trimesh.sample.sample_surface(mesh, 96, seed=0)
    pts_obj = np.asarray(pts_obj) * clip.get("mesh_scale", 1.0)
    dense_obj, _ = trimesh.sample.sample_surface(mesh, 3000, seed=1)       # for the housing-object clearance
    dense_obj = np.asarray(dense_obj) * clip.get("mesh_scale", 1.0)
    half = size / 2
    Q, OP, OR = source["joint_pos"], source["obj_pos"], source["obj_rot"]
    frames = np.arange(0, len(Q), stride)
    poses = [configured_pose(cam_cfg, sensor["housing"]) if c.get("configured") else
             candidate_pose(c["side"], c["r_mm"] / 1000, c["z_mm"] / 1000, c["tilt"], sensor["housing"], c["pan"]) for c in cands]
    nC, nF = len(cands), len(frames)
    res = {k: np.zeros((nC, nF), np.float32) for k in ("table", "fingers", "object")}
    vis_obj = np.zeros((len(zmins), nC, nF), np.float32)
    vis_tip = np.zeros((len(zmins), nC, nF), np.float32)
    near = np.zeros(nF, bool)

    for fi, t in enumerate(frames):
        sc.set_floating(Q[t])
        sc.set_object(OP[t], OR[t])
        sc.fk()
        Tp = sc.palm()
        tips = sc.tips()
        Pw = pts_obj @ OR[t].T + OP[t]
        near[fi] = np.linalg.norm(OP[t] - Tp[:3, 3]) < NEAR_OBJECT
        allP = np.concatenate([Pw, tips])
        Rp, pp = Tp[:3, :3], Tp[:3, 3]
        fing_palm = (sc.world_points(sc.finger_geoms, 250) - pp) @ Rp           # in the palm frame, once per frame
        obj_palm = (dense_obj @ OR[t].T + OP[t] - pp) @ Rp
        for ci, (T_cam, T_box) in enumerate(poses):
            corners_w = (T_box[:3, 3] + CORNERS * half @ T_box[:3, :3].T) @ Rp.T + pp
            res["table"][ci, fi] = corners_w[:, 2].min() - C.TABLE_Z
            res["fingers"][ci, fi] = sc.points_to_box(fing_palm, T_box, half)
            res["object"][ci, fi] = sc.points_to_box(obj_palm, T_box, half)
            Tw = Tp @ T_cam
            for zi, zmin in enumerate(zmins):
                ok = cam.measures(m, d, Tw, allP, z_min=zmin, tol=TOL)
                vis_obj[zi, ci, fi] = ok[:96].mean()
                vis_tip[zi, ci, fi] = ok[96:].mean()
    return clip["name"], res, vis_obj, vis_tip, near


def summarise(cands, results, zmins):
    """Per candidate: metrics pooled over all clips' frames, and per clip the table clearance."""
    rows = []
    names = [r[0] for r in results]
    for ci, c in enumerate(cands):
        row = dict(c)
        tab = np.concatenate([r[1]["table"][ci] for r in results])
        fin = np.concatenate([r[1]["fingers"][ci] for r in results])
        obj = np.concatenate([r[1]["object"][ci] for r in results])
        near = np.concatenate([r[4] for r in results])
        row.update(
            table_min_mm=float(tab.min() * 1000), table_under_pct=float((tab < TABLE_LIMIT).mean() * 100),
            table_clips_under=int(sum(r[1]["table"][ci].min() < TABLE_LIMIT for r in results)),
            fingers_min_mm=float(fin.min() * 1000), fingers_under_pct=float((fin < FINGER_LIMIT).mean() * 100),
            object_min_mm=float(obj.min() * 1000), object_under_pct=float((obj < OBJECT_LIMIT).mean() * 100),
            table_min_per_clip_mm={n: float(r[1]["table"][ci].min() * 1000) for n, r in zip(names, results)})
        for zi, zmin in enumerate(zmins):
            vo = np.concatenate([r[2][zi, ci] for r in results])[near]
            vt = np.concatenate([r[3][zi, ci] for r in results])[near]
            k = f"{int(round(zmin * 1000))}"
            row[f"object_seen_{k}"] = float(vo.mean() * 100)
            row[f"object_any_{k}"] = float((vo >= 0.1).mean() * 100)
            row[f"tips_seen_{k}"] = float(vt.mean() * 100)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(C.STORAGE / "runs" / "flashdex" / "data_r10_flat"))
    ap.add_argument("--rollouts", default=None, help="score FDR rollouts (RUN/evals/step<N>) instead of the reference clips")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--sides", type=float, nargs="*", default=[0, 45, 90, 180, 270, 315])
    ap.add_argument("--r", type=float, nargs="*", default=[48, 54, 62])
    ap.add_argument("--z", type=float, nargs="*", default=[-40, -22, -8, 6])
    ap.add_argument("--tilt", type=float, nargs="*", default=[0, 10, 20, 30])
    ap.add_argument("--pan", type=float, nargs="*", default=[0, 20, 35])
    ap.add_argument("--evaluate", action="store_true", help="score only the pose configured in cameras.yaml")
    ap.add_argument("--zmin", type=float, nargs="*", default=[0.07, 0.04],
                    help="minimum depths to score: 0.07 m is the D405 at 848x480, 0.04 m at 424x240")
    args = ap.parse_args()
    cams = C.load_cameras()
    cam_cfg, sensor = cams["wrist"], cams["sensors"][cams["wrist"]["sensor"]]
    clips = C.load_pack(args.dataset)
    tag = args.tag or (Path(args.dataset).name + ("_rollouts" if args.rollouts else "_reference") + ("_configured" if args.evaluate else ""))
    out = C.RUNS / "wrist_cam" / tag
    out.mkdir(parents=True, exist_ok=True)
    roll = C.load_rollouts(args.rollouts)[0] if args.rollouts else {}
    cands = make_candidates(args.sides, args.r, args.z, args.tilt, args.pan)
    if args.evaluate:                                            # only the pose that is in cameras.yaml
        T, Tb = configured_pose(cam_cfg, sensor["housing"])
        cands = [{"configured": True, "side": float(np.degrees(np.arctan2(Tb[1, 3], Tb[0, 3])) % 360),
                  "r_mm": float(np.hypot(Tb[0, 3], Tb[1, 3]) * 1000), "z_mm": float(Tb[2, 3] * 1000),
                  "tilt": float(np.degrees(np.arccos(np.clip(T[2, 2], -1, 1)))), "pan": 0.0}]
    jobs = []
    for c in clips:
        src = roll.get(c["name"]) if args.rollouts else c
        if src is None:
            continue
        light = {k: c[k] for k in ("name", "mesh_dir", "mesh_scale")}
        jobs.append((light, cands, cam_cfg, sensor, args.stride, args.zmin,
                     {k: src[k] for k in ("joint_pos", "obj_pos", "obj_rot")}))
    with mp.get_context("fork").Pool(min(args.workers, len(jobs))) as pool:
        results = pool.map(_worker, jobs)
    rows = summarise(cands, results, args.zmin)
    json.dump({"dataset": args.dataset, "rollouts": args.rollouts, "stride": args.stride, "zmin": args.zmin,
               "limits_mm": {"table": TABLE_LIMIT * 1000, "fingers": FINGER_LIMIT * 1000, "object": OBJECT_LIMIT * 1000},
               "frames": int(sum(len(r[4]) for r in results)), "frames_near_object": int(sum(r[4].sum() for r in results)),
               "candidates": rows}, open(out / "candidates.json", "w"), indent=1)
    print(f"{len(rows)} candidates x {sum(len(r[4]) for r in results)} frames -> {out}/candidates.json")
    k = f"{int(round(args.zmin[0] * 1000))}"
    if args.evaluate:
        b = rows[0]
        print(f"configured pose (side {b['side']:.0f}, r {b['r_mm']:.0f} mm, z {b['z_mm']:.0f} mm, tilt {b['tilt']:.0f}): object "
              f"{b[f'object_seen_{k}']:.1f}% (seen at all in {b[f'object_any_{k}']:.0f}% of frames), fingertips {b[f'tips_seen_{k}']:.0f}% | "
              f"table: min {b['table_min_mm']:.0f} mm, under {TABLE_LIMIT * 1000:.0f} mm in {b['table_under_pct']:.1f}% of frames "
              f"({b['table_clips_under']} clips) | fingers min {b['fingers_min_mm']:.0f} mm | object min {b['object_min_mm']:.0f} mm")
        print("  table clearance per clip [mm]:", {n.replace("_anthony_real", ""): round(v) for n, v in b["table_min_per_clip_mm"].items()})
        return
    print(f"best per side by object_seen_{k} + tips_seen_{k} / 2, fingers and object clear:")
    for s in args.sides:
        ok = [r for r in rows if r["side"] == s and r["fingers_min_mm"] >= FINGER_LIMIT * 1000 and r["object_under_pct"] < 1]
        if not ok:
            print(f"  side {s:5.0f}: no candidate keeps clear of the fingers and the object")
            continue
        b = max(ok, key=lambda r: r[f"object_seen_{k}"] + r[f"tips_seen_{k}"] / 2)
        print(f"  side {s:5.0f} ({SIDES.get(int(s), '')}): r {b['r_mm']:.0f} z {b['z_mm']:.0f} tilt {b['tilt']:.0f} pan {b['pan']:.0f} | object "
              f"{b[f'object_seen_{k}']:.1f}% (any {b[f'object_any_{k}']:.0f}%), tips {b[f'tips_seen_{k}']:.0f}% | table min "
              f"{b['table_min_mm']:.0f} mm, under {b['table_under_pct']:.1f}% of frames, {b['table_clips_under']} clips | "
              f"fingers min {b['fingers_min_mm']:.0f} mm | object min {b['object_min_mm']:.0f} mm")


if __name__ == "__main__":
    main()
