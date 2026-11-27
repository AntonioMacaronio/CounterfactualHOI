"""Render the policy's two depth images for every step of a clip, from the camera definitions.

  python scripts/embodiment/render_depth.py --dataset DATA [--rollouts RUN/evals/step<N>] [--arm marvin_m6ccs]
        [--clips VID ...] [--stride 1] [--noise-seed 0] [--workers 10]
  -> $PROJECT_STORAGE_ROOT/runs/embodiment/depth/<robot>/<tag>/<clip>.npz, <clip>.png, manifest.json
     (venv: $PROJECT_STORAGE_ROOT/venvs/arm)

This is the re-rendering that rows 5 and 6 of "Data this stage has to hand to 1.4" ask for: it needs only the robot's
joint angles (row 1), the object's pose (row 3) and the scene (row 11). No GPU: rays are cast in MuJoCo.

Robot:
  (default)   FDR's floating-wrist hand. The wrist camera's image is the same as on any arm (the camera looks away
              from the arm). The exo image has no arm in it, only a generic forearm behind the wrist.
  --arm NAME  the arm model, with the joint angles ik.py wrote for the same dataset / rollouts (clips or frames
              the arm cannot reach are skipped and listed in the manifest), or with --ik-dir those of any IK
              result, e.g. ik_pyroki.py's fingertips mode (all frames, with its re-bent fingers).

Per clip, <clip>.npz holds, at the policy size (60 x 106):
  wrist_depth, exo_depth      float32 metres along the optical axis, 0 where invalid
  wrist_valid, exo_valid      bool: inside the depth range and seen by both imagers
  wrist_depth_noisy, ...      the same after the sensor model of cameras.yaml (with --noise-seed)
  wrist_T, exo_T              camera poses in the clip world (optical frames), frames (indices into the clip)
<clip>.png is a contact sheet: six moments of the clip, top to bottom; wrist ideal | wrist with noise | exo ideal |
exo with noise. Near is bright, invalid pixels are dark red.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

import common as C
from cameras import Cameras, colorize


def _worker(job):
    clip, src, arm, ikres, stride, seed, out = job
    from PIL import Image
    from scene import Scene
    cams = Cameras()
    sc = Scene(arm or "floating", clip, forearm=arm is None)
    Q, OP, OR = src["joint_pos"], src["obj_pos"], src["obj_rot"]
    n = min(len(Q), len(OP))
    frames = np.arange(0, n, stride)
    if arm and not ikres.get("keep_all"):
        ok = (ikres["pos_err_mm"] < 1.0) & (ikres["rot_err_deg"] < 0.5)
        frames = frames[ok[frames]]
    rng = np.random.default_rng(seed) if seed is not None else None
    res = {}
    for cam in (cams.wrist, cams.exo):
        key = cam.name.split("_")[0]
        ep = cam.new_episode(rng) if rng is not None else None
        D, V, Dn, Vn, Ts = [], [], [], [], []
        for t in frames:
            if arm:
                sc.set_arm(ikres["arm_q"][t], ikres["hand_q"][t])
            else:
                sc.set_floating(Q[t])
            sc.set_object(OP[t], OR[t])
            sc.fk()
            T = sc.cam_pose(cam)
            img = cam.render(sc.model, sc.data, T, bodyexclude=sc.cam_body(cam))
            d, v = cam.observe(img)
            D.append(d)
            V.append(v)
            Ts.append(T)
            if rng is not None:
                dn, vn = cam.observe(img, rng, ep)
                Dn.append(dn)
                Vn.append(vn)
        res[f"{key}_depth"], res[f"{key}_valid"], res[f"{key}_T"] = np.array(D, np.float32), np.array(V), np.array(Ts, np.float32)
        if rng is not None:
            res[f"{key}_depth_noisy"], res[f"{key}_valid_noisy"] = np.array(Dn, np.float32), np.array(Vn)
    res["frames"] = frames
    np.savez_compressed(out / f"{clip['name']}.npz", **res)
    # contact sheet: one row per moment; wrist ideal | wrist with noise | exo ideal | exo with noise
    if len(frames):
        pick = np.linspace(0, len(frames) - 1, 6).astype(int)
        cols = [(k, sfx, c) for k, c in (("wrist", cams.wrist), ("exo", cams.exo)) for sfx in (("", "_noisy") if rng is not None else ("",))]
        rows = []
        for i in pick:
            tiles = []
            for k, sfx, c in cols:
                tiles.append(np.kron(colorize(res[f"{k}_depth{sfx}"][i], res[f"{k}_valid{sfx}"][i], c.z_min, c.z_max), np.ones((3, 3, 1), np.uint8)))
                tiles.append(np.full((tiles[-1].shape[0], 8, 3), 255, np.uint8))
            rows.append(np.concatenate(tiles[:-1], 1))
            rows.append(np.full((8, rows[-1].shape[1], 3), 255, np.uint8))
        Image.fromarray(np.concatenate(rows[:-1], 0)).save(out / f"{clip['name']}.png")
    stats = {"frames": int(len(frames)), "of": int(len(np.arange(0, n, stride)))}
    for k in ("wrist", "exo"):
        if len(frames):
            stats[f"{k}_valid_pct"] = float(res[f"{k}_valid"].mean() * 100)
    return clip["name"], stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(C.STORAGE / "runs" / "flashdex" / "data_r10_flat"))
    ap.add_argument("--rollouts", default=None)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--ik-dir", default=None, help="with --arm: a result directory of ik.py or ik_pyroki.py (default: ik.py's for the same data)")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--noise-seed", type=int, default=0, help="-1: ideal sensor only")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    clips = C.load_pack(args.dataset)
    roll = C.load_rollouts(args.rollouts)[0] if args.rollouts else None
    tag = args.tag or (Path(args.dataset).name + ("_rollouts" if args.rollouts else "_reference"))
    out = C.RUNS / "depth" / (args.arm or "floating") / (Path(args.ik_dir).name if args.ik_dir else tag)
    out.mkdir(parents=True, exist_ok=True)
    jobs, skipped = [], {}
    for c in clips:
        if args.clips and c["name"] not in args.clips:
            continue
        src = roll.get(c["name"]) if roll is not None else c
        if src is None:
            continue
        ikres = None
        if args.arm:
            f = (Path(args.ik_dir) if args.ik_dir else C.RUNS / "ik" / args.arm / tag) / f"{c['name']}.npz"
            if not f.exists():
                skipped[c["name"]] = f"no IK result ({f}); run ik.py or ik_pyroki.py first"
                continue
            z = np.load(f)
            ikres = {k: z[k] for k in ("arm_q", "hand_q", "pos_err_mm", "rot_err_deg")}
            ikres["keep_all"] = bool(args.ik_dir)                # a chosen result is rendered as it is
        light = {k: c[k] for k in ("name", "mesh_dir", "mesh_scale")}
        jobs.append((light, {k: src[k] for k in ("joint_pos", "obj_pos", "obj_rot")}, args.arm, ikres, args.stride,
                     None if args.noise_seed < 0 else args.noise_seed + c["motion_id"], out))
    with mp.get_context("fork").Pool(max(1, min(args.workers, len(jobs)))) as pool:
        res = dict(pool.map(_worker, jobs))
    cams = C.load_cameras()
    json.dump({"dataset": args.dataset, "rollouts": args.rollouts, "robot": args.arm or "floating", "stride": args.stride,
               "cameras": {k: cams[k] for k in ("wrist", "exo")}, "clips": res, "skipped": skipped},
              open(out / "manifest.json", "w"), indent=1)
    for k, v in res.items():
        print(f"{k:22s} {v['frames']:4d} / {v['of']:4d} frames | valid pixels: wrist {v.get('wrist_valid_pct', 0):.0f}%, exo {v.get('exo_valid_pct', 0):.0f}%")
    for k, v in skipped.items():
        print(f"{k:22s} skipped: {v}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
