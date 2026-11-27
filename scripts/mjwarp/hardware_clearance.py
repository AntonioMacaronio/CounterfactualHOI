"""How far the real robot's wrist hardware (arm link, mount, D405; scripts/mjwarp/hand_variant.py) would sit inside the
table along a retargeted clip: per frame, the lowest vertex of each hardware geom above the clip's table disk.

  python -I scripts/mjwarp/hardware_clearance.py DATASET [--arm vega_1p]       (any env with mujoco)

Works on any dataset retargeted to the right Sharpa hand (robot sharpa or sharpa_<arm>): the hardware is posed from
the retargeted joint values (retargeted/<clip>.npz qpos) with the variant's right.xml. Writes DATASET/hardware_clearance.json.
"""

import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np

STORAGE = Path(os.environ.get("PROJECT_STORAGE_ROOT", "/data/sky/projects/CounterfactualHOI"))
FDR = Path(os.environ.get("FDR", STORAGE / "build" / "FlashDexRetarget"))
NAMES = ("collision_hand_right_wrist_armlink", "collision_hand_right_wrist_mount", "collision_hand_right_wrist_camera")


def geom_points(m, gid):
    """Points of a geom in its own frame: mesh vertices, or a box's 8 corners."""
    if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
        k = m.geom_dataid[gid]
        a, n = m.mesh_vertadr[k], m.mesh_vertnum[k]
        return m.mesh_vert[a:a + n].astype(np.float64)
    s = m.geom_size[gid]
    return np.array([[x, y, z] for x in (-s[0], s[0]) for y in (-s[1], s[1]) for z in (-s[2], s[2])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--arm", default="vega_1p")
    a = ap.parse_args()
    D = Path(a.dataset)
    m = mujoco.MjModel.from_xml_path(str(FDR / "assets" / "robot" / f"sharpa_{a.arm}" / "right.xml"))
    d = mujoco.MjData(m)
    gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in NAMES]
    pts = [geom_points(m, g) for g in gids]
    out = {}
    for f in sorted((D / "retargeted").glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        q, top = z["qpos"], float(z["support_disks"][0][2])
        low = np.zeros((len(q), len(gids)))
        for t in range(len(q)):
            d.qpos[:q.shape[1]] = q[t]
            mujoco.mj_kinematics(m, d)
            for j, (g, p) in enumerate(zip(gids, pts)):
                low[t, j] = (p @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min() - top
        worst = low.min(1)
        out[f.stem] = {"lowest_cm": [round(100 * float(x), 2) for x in low.min(0)],
                       "frames_inside": float((worst < 0).mean()), "frames_within_8mm": float((worst < 0.008).mean())}
        print(f"{f.stem}: lowest above the table (arm link / mount / camera) "
              f"{', '.join('%+.1f' % (100 * x) for x in low.min(0))} cm; inside the table in {100 * (worst < 0).mean():.1f}% "
              f"of frames, within 8 mm in {100 * (worst < 0.008).mean():.1f}%")
    json.dump(out, open(D / "hardware_clearance.json", "w"), indent=1)


if __name__ == "__main__":
    main()
