"""A right-hand-only FDR pack -> the bimanual robot's joint layout, the left hand parked out of the way.

  python -I scripts/flashdex/pad_bimanual.py DATASET          (flashdex env) -> DATASET/motion_bimanual.pt

FDR retargets single-hand clips with the single-hand MJCF (assets/robot/sharpa/right.xml: 28 joints), but trains on
the bimanual robot, whose joint ids (MJCF order == .pt column order: the right hand's 28, then the left's) index
the reference joint arrays: a 28-column pack reads out of bounds at the first reset. This writes the same pack
with 56 columns: the right hand's reference, and the left hand parked at a fixed pose 2 m above the table and 1.2 m
to the side (fingers at 0, clipped to their limits; zero velocity). Train it with per_hand_actor_critic=false (the
per-hand reward split needs both hands' references).
"""

import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

FDR = Path(os.environ.get("FDR", "/data/sky/projects/CounterfactualHOI/build/FlashDexRetarget"))
PARK = {"L_forearm_pos_x_joint": 0.0, "L_forearm_pos_y_joint": 1.05, "L_forearm_pos_z_joint": 1.7}


def main():
    ds = Path(sys.argv[1])
    pt = torch.load(ds / "motion.pt", map_location="cpu", weights_only=False)
    m = mujoco.MjModel.from_xml_path(str(FDR / "assets" / "robot" / "sharpa" / "bimanual.xml"))
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    right = [n for n in names if n.startswith(("R_", "right_"))]
    rm = mujoco.MjModel.from_xml_path(str(FDR / "assets" / "robot" / "sharpa" / "right.xml"))
    rnames = [mujoco.mj_id2name(rm, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(rm.njnt)]
    assert rnames == right == names[:28], "right.xml joints must be bimanual.xml's first 28"
    jp = pt["joint_pos"]
    assert jp.shape[1] == 28, f"expected a 28-column right-hand pack, got {jp.shape[1]}"
    park = np.zeros(m.njnt - 28, np.float32)
    for k, n in enumerate(names[28:]):
        lo, hi = m.jnt_range[28 + k]
        park[k] = np.clip(PARK.get(n, 0.0), lo, hi)
    T = jp.shape[0]
    pt["joint_pos"] = torch.cat([jp, torch.from_numpy(park)[None].expand(T, -1)], 1).contiguous()
    pt["joint_vel"] = torch.cat([pt["joint_vel"], torch.zeros(T, m.njnt - 28, dtype=pt["joint_vel"].dtype)], 1)
    torch.save(pt, ds / "motion_bimanual.pt")
    print(f"wrote {ds / 'motion_bimanual.pt'}: joint_pos {tuple(pt['joint_pos'].shape)}, left parked at {PARK}")


if __name__ == "__main__":
    main()
