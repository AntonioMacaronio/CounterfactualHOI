"""§1.4's "pick done" and "place done" on archived rollouts (antzhang-notes/project.md §1.4, Reward; §1.3 row 9).

  python -I scripts/flashdex/done_checks.py --dataset DATA --archive RUN/evals/step<N> [--clips VID ...]   (flashdex env)

Per rollout, from the archive's fields (scripts/flashdex/train_cfhoi.py: obj_pos, obj_quat, contact_force,
link_obj_dist) and the dataset (the object's mesh, the clip's table disk):
  pick done   the object's lowest point is at least 3 cm above the table and the thumb and at least one other
              finger touch it, for 0.2 s. The step at which that span is complete.
  place done  after pick done: the object rests on the table (lowest point within 1 cm, slower than 2 cm/s and
              10 deg/s), at least 20 cm across the table from where it started, upright (the object-frame direction
              that pointed up at the first step is within 20 deg of up again), no hand link touches it and the five
              fingertips are at least 3 cm from its surface. The first such step.
"Touch" = a contact force above --min-force on the link. The speed limits and --min-force are this script's choices;
the other numbers are the notes'. Rollouts archived before contact_force / link_obj_dist existed have no contact
information: their checks are reported as unknown (null), with the geometric parts alone in "geometry_only".
Writes ARCHIVE/done.json and prints one line per clip.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh

TIPS = [1, 2, 3, 4, 5]                                  # columns of link_obj_dist (0 = palm)


def contact_links(run_dir, side="right"):
    """Rows of contact_force that are thumb links and other-finger links: from the names in the run's config.yaml
    (the <r|l>_alllink_contact sensor's pattern). The palm (and anything else) is neither."""
    import yaml
    cfg = yaml.safe_load(open(Path(run_dir) / "config.yaml"))
    names = next(sn["primary"]["pattern"] for sn in cfg["scene"]["sensors"] if sn.get("name") == f"{side[0]}_alllink_contact")
    thumb = [i for i, n in enumerate(names) if "thumb" in n]
    fingers = [i for i, n in enumerate(names) if any(k in n for k in ("index", "middle", "mid_", "ring", "pinky"))]
    return thumb, fingers, len(names)


def rotmat(q):
    """(T, 4) quaternions (w, x, y, z) -> (T, 3, 3)."""
    w, x, y, z = (q / np.linalg.norm(q, axis=-1, keepdims=True)).T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def first_held(ok, n):
    """Index of the last step of the first run of n consecutive True, or None."""
    run = 0
    for i, v in enumerate(ok):
        run = run + 1 if v else 0
        if run >= n:
            return i
    return None


def check(roll, verts, z_top, fps, a, links):
    pos, R = roll["obj_pos"].reshape(-1, 3), rotmat(roll["obj_quat"].reshape(-1, 4))
    T = len(pos)
    low = (np.einsum("tij,vj->tvi", R, verts)[..., 2] + pos[:, None, 2]).min(1) - z_top      # lowest point above the table
    carried = np.linalg.norm(pos[:, :2] - pos[0, :2], axis=1)
    up0 = R[0].T @ np.array([0.0, 0.0, 1.0])                                                  # the object's vertical axis
    tilt = np.degrees(np.arccos(np.clip(np.einsum("tij,j->ti", R, up0)[:, 2], -1, 1)))
    speed = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps]
    dR = np.einsum("tji,tjk->tik", R[:-1], R[1:])
    spin = np.r_[0.0, np.degrees(np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1, 1))) * fps]
    lifted = low >= a.lift
    resting = (np.abs(low) <= a.rest) & (speed < a.max_speed) & (spin < a.max_spin) & (carried >= a.carry) & (tilt <= a.tilt)
    has_contact = "contact_force" in roll and "link_obj_dist" in roll
    out = {"frames": T, "lift_max_cm": float(100 * low.max()), "carried_end_cm": float(100 * carried[-1]),
           "tilt_end_deg": float(tilt[-1]), "low_end_cm": float(100 * low[-1]),
           "geometry_only": {"lifted_step": first_held(lifted, a.hold), "resting_step": None}}
    g = out["geometry_only"]
    if g["lifted_step"] is not None:
        r = first_held(resting[g["lifted_step"]:], 1)
        g["resting_step"] = None if r is None else g["lifted_step"] + r
    if not has_contact:
        return dict(out, pick_done=None, place_done=None, contacts=False)
    thumb, fingers, n_l = links
    f = np.linalg.norm(roll["contact_force"].reshape(T, -1, n_l, 3)[:, 0], axis=-1) > a.min_force   # (T, links), first side
    tips = roll["link_obj_dist"].reshape(T, -1, 6)[:, 0][:, TIPS]
    grip = f[:, thumb].any(1) & f[:, fingers].any(1)
    pick = first_held(lifted & grip, a.hold)
    place = None
    if pick is not None:
        ok = resting & ~f.any(1) & (tips.min(1) >= a.clear)
        r = first_held(ok[pick:], 1)
        place = None if r is None else pick + r
    return dict(out, pick_done=pick, place_done=place, contacts=True, grip_frames=float(grip.mean()),
                touch_end=bool(f[-1].any()), tip_clear_end_cm=float(100 * tips[-1].min()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="FDR dataset dir (motion_bimanual.pt or motion.pt, objects/)")
    ap.add_argument("--archive", required=True, help="RUN/evals/step<N> (manifest.json, rollouts/)")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--lift", type=float, default=0.03)
    ap.add_argument("--rest", type=float, default=0.01)
    ap.add_argument("--carry", type=float, default=0.20)
    ap.add_argument("--tilt", type=float, default=20.0)
    ap.add_argument("--clear", type=float, default=0.03)
    ap.add_argument("--hold-s", type=float, default=0.2, help="how long pick done has to hold")
    ap.add_argument("--max-speed", type=float, default=0.02, help="m/s: 'nearly still'")
    ap.add_argument("--max-spin", type=float, default=10.0, help="deg/s: 'nearly still'")
    ap.add_argument("--min-force", type=float, default=0.01, help="N: a link touches the object above this")
    a = ap.parse_args()
    a.hold = max(1, int(round(a.hold_s * a.fps)))
    D, A = Path(a.dataset), Path(a.archive)
    pack = D / "motion_bimanual.pt"
    m = torch.load(pack if pack.exists() else D / "motion.pt", map_location="cpu", weights_only=False)
    names = list(m["motion_filename"])
    side = "right" if "right_object_mesh_dirs" in m else "left"
    res = {}
    try:
        links = contact_links(A.parent.parent, side)
    except Exception as exc:  # noqa: BLE001 - old runs: no contact fields either
        print(f"no contact link names ({exc!r})")
        links = None
    for e in json.load(open(A / "manifest.json"))["saved"]:
        mid, vid = int(e["motion_id"]), str(e["name"]).replace("cfhoi_", "")
        if a.clips and vid not in a.clips:
            continue
        slot = int(m[f"motion_object_slot_{side}"][mid]) if f"motion_object_slot_{side}" in m else names.index(e["name"])
        mesh = trimesh.load(D / m[f"{side}_object_mesh_dirs"][slot] / "visual.obj", force="mesh", process=False)
        # the lowest point only needs the corners of the convex hull
        verts = np.asarray(trimesh.convex.convex_hull(mesh).vertices) * float(m[f"{side}_object_mesh_scales"][slot])
        z_top = float(m["support_disks"][mid][0][2]) if "support_disks" in m else 0.8
        roll = dict(np.load(A / e["rollout"]))
        if links is None:
            roll.pop("contact_force", None)
        res[vid] = r = check(roll, verts, z_top, a.fps, a, links)
        txt = (f"pick done {r['pick_done']}, place done {r['place_done']}" if r["contacts"] else
               f"no contact fields (geometry only: lifted at {r['geometry_only']['lifted_step']}, resting at {r['geometry_only']['resting_step']})")
        print(f"{vid}: {r['frames']} steps; {txt}; lift max {r['lift_max_cm']:.1f} cm, carried {r['carried_end_cm']:.1f} cm, "
              f"tilt at the end {r['tilt_end_deg']:.1f} deg, lowest point at the end {r['low_end_cm']:+.1f} cm")
    json.dump({"thresholds": {k: v for k, v in vars(a).items() if k not in ("dataset", "archive", "clips")}, "clips": res},
              open(A / "done.json", "w"), indent=1)
    print(f"wrote {A / 'done.json'}")


if __name__ == "__main__":
    main()
