"""GenEnv clips (AdaDex): copies of every clip with a new object start and a new goal, for the hybrid policy.

  python -I scripts/flashdex/make_gen_clips.py SRC DST [--variants 2] [--seed 0]     (the flashdex env)

SRC: an imported dataset with tables (human_demo/*.npz after add_table.py, objects/). DST gets the original clips
(TrackEnv: follow the reference) plus --variants warped copies per clip, cfhoi_gen<k>_<vid> (GenEnv). A warp is a
planar rigid motion of the whole scene (hand, object; about the object's start, on the table) that changes over
the clip: Delta_start (yaw +-30 deg, shift +-5 cm) until the object is lifted, Delta_goal (yaw +-30 deg, shift +-8 cm)
from when it is put down, a smoothstep in between. So the object starts somewhere else and must be put somewhere
else, and the human hand reference moves with it (still a consistent grasp: a rigid motion about the vertical axis
leaves gravity and the table unchanged). The disk is re-centred under the new path. Then run FDR's kinematic
retargeting on DST (desk.mode=keep), which retargets every clip, warped ones included, to the Sharpa hand.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

FDR = os.environ.get("FDR", "/data/sky/projects/CounterfactualHOI/build/FlashDexRetarget")
sys.path.insert(0, os.path.join(FDR, "src"))
from retarget.human_demo import clips as C  # noqa: E402


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    out = np.zeros(np.shape(a) + (3, 3))
    out[..., 0, 0], out[..., 0, 1], out[..., 1, 0], out[..., 1, 1], out[..., 2, 2] = c, -s, s, c, 1
    return out


def warp(clip, rng, start_deg, start_cm, goal_deg, goal_cm, moved=0.01):
    pose = C.obj_pose(clip, "right").astype(np.float64)
    T = len(pose)
    p = pose[:, :3, 3]
    lift = np.nonzero(np.linalg.norm(p - p[0], axis=1) > moved)[0]
    place = np.nonzero(np.linalg.norm(p - p[-1], axis=1) > moved)[0]
    if not len(lift) or not len(place) or place[-1] <= lift[0]:
        return None
    t0, t1 = int(lift[0]), int(place[-1])
    u = np.clip((np.arange(T) - t0) / max(t1 - t0, 1), 0, 1)
    s = u * u * (3 - 2 * u)
    ds = np.r_[np.radians(rng.uniform(-start_deg, start_deg)), rng.uniform(-start_cm, start_cm, 2) / 100]
    dg = np.r_[np.radians(rng.uniform(-goal_deg, goal_deg)), rng.uniform(-goal_cm, goal_cm, 2) / 100]
    d = ds[None] + s[:, None] * (dg - ds)[None]                                 # (T, 3): yaw, dx, dy
    R = Rz(d[:, 0])
    c = np.r_[p[0, :2], 0.0]
    shift = np.c_[d[:, 1:], np.zeros(T)]
    pts = lambda x: np.einsum("tij,t...j->t...i", R, x - c) + c + (shift if x.ndim == 2 else shift[:, None])  # noqa: E731
    out = dict(clip)
    newpose = pose.copy()
    newpose[:, :3, :3] = R @ pose[:, :3, :3]
    newpose[:, :3, 3] = pts(p)
    out["obj_right_pose"] = newpose.astype(np.float32)
    out["mano_right_wrist_pos"] = pts(clip["mano_right_wrist_pos"].astype(np.float64)).astype(np.float32)
    out["mano_right_wrist_rot"] = (R @ clip["mano_right_wrist_rot"].astype(np.float64)).astype(np.float32)
    out["mano_right_joints"] = pts(clip["mano_right_joints"].astype(np.float64)).astype(np.float32)
    disks = np.array(clip["support_disks"], np.float32).copy()
    xy = newpose[:, :2, 3]
    disks[:, :2] = (xy.min(0) + xy.max(0)) / 2
    out["support_disks"] = disks
    info = {"lift_frame": t0, "place_frame": t1, "start": [round(float(np.degrees(ds[0])), 1), *np.round(100 * ds[1:], 1).tolist()],
            "goal": [round(float(np.degrees(dg[0])), 1), *np.round(100 * dg[1:], 1).tolist()],
            "max_reach_from_disk_m": round(float(np.linalg.norm(xy - disks[0, :2], axis=1).max()), 3)}
    return out, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-deg", type=float, default=30.0)
    ap.add_argument("--start-cm", type=float, default=5.0)
    ap.add_argument("--goal-deg", type=float, default=30.0)
    ap.add_argument("--goal-cm", type=float, default=8.0)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    (dst / "human_demo").mkdir(parents=True, exist_ok=True)
    if not (dst / "objects").exists():
        (dst / "objects").symlink_to((src / "objects").resolve())               # same meshes, convex parts, SDFs
    rng = np.random.default_rng(args.seed)
    log = {}
    for path in C.list_clips(src / "human_demo"):
        clip = C.load_clip(path)
        C.save_clip(dst / "human_demo" / path.name, clip)
        vid = path.stem.removeprefix("cfhoi_")
        for k in range(args.variants):
            r = warp(clip, rng, args.start_deg, args.start_cm, args.goal_deg, args.goal_cm)
            if r is None:
                log[f"gen{k}_{vid}"] = "skipped: no lift / place found"
                continue
            C.save_clip(dst / "human_demo" / f"cfhoi_gen{k}_{vid}.npz", r[0])
            log[f"gen{k}_{vid}"] = r[1]
            print(f"gen{k}_{vid}", r[1])
    json.dump(log, open(dst / "gen_clips.json", "w"), indent=1)


if __name__ == "__main__":
    main()
