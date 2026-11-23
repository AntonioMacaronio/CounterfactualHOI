"""FlashDexRetarget import source for our reconstructions (linked into FlashDexRetarget as
src/retarget/human_demo/sources/cfhoi.py by scripts/flashdex/setup_cfhoi.sh).

  bash scripts/retarget/import_human_demo.sh source=cfhoi raw=$PROJECT_STORAGE_ROOT dataset=<out> sides=right \
      source.refine=refine_r6        (or refine_r5: the tracker's MANO hand + round-5 object poses)

One demo per clip (source.clips): the right hand as MANO parameters (flat-hand-mean, no PCA, like TACO) and the
object (one mesh per clip, the fprecon run's metric SAM3D mesh) in a z-up world built from the clip's table: z =
the table normal (export/table.json normal_up), x = the camera's right projected onto the table, the origin under the
object's first-frame centre, and the table at z = source.table_height. Hand parameters:
  * refine_r6 and later: export/mano_params.npz (global_orient, transl, hand_pose, betas; camera frame);
  * otherwise: scripts/refine/mano.tracker_params_camera (Dyn-HaMR, depth-corrected; the round-5 finger edits are
    mesh vertices, not MANO, so they are not used).
Object poses: <refine>/VID/export/object_poses.npy (object -> camera).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from retarget.human_demo.sources.base import Demo, Hand, Source as Base

# this repo: CFHOI_ROOT (scripts/env.sh), else from this file's real path (FDR imports it through a symlink)
CFHOI = os.environ.get("CFHOI_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.insert(0, os.path.join(CFHOI, "scripts", "refine"))
sys.path.insert(0, os.path.join(CFHOI, "scripts", "fprecon"))
import mano as M  # noqa: E402


def world_frame(rd, vertices):
    """camera -> world of one clip: (R 3x3, o 3) with p_world = R (p_cam - o) + [0, 0, table_height]. rd: the
    refine run dir (export/table.json, export/object_poses.npy); vertices: the object mesh's (object frame)."""
    rd = Path(rd)
    tj = json.load(open(rd / "export" / "table.json"))
    n = np.array(tj["normal_up"], float)
    n /= np.linalg.norm(n)
    x = np.array([1.0, 0, 0]) - n[0] * n
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    R = np.stack([x, y, n])
    poses = np.load(rd / "export" / "object_poses.npy")
    c0 = poses[0, :3, :3] @ np.asarray(vertices).mean(0) + poses[0, :3, 3]
    o = c0 - (n @ c0 - float(tj["offset"])) * n                         # first-frame centre, dropped onto the table
    return R, o


class Source(Base):
    mano = {"use_pca": False, "flat_hand_mean": True}

    def units(self) -> list[str]:
        return list(self.cfg.clips)

    def _paths(self, vid):
        st = self.raw
        return st / "build" / "CHOIR" / "output" / vid, st / "runs" / "fprecon" / vid, st / "runs" / self.cfg.refine / vid

    def world(self, vid):
        return world_frame(self._paths(vid)[2], self.mesh(vid).vertices)

    def demos(self, unit: str) -> list[Demo | str]:
        vid = unit
        vd, run, rd = self._paths(vid)
        mp = rd / "export" / "mano_params.npz"
        if mp.exists():
            z = np.load(mp, allow_pickle=True)
            P = {k: z[k] for k in ("global_orient", "transl", "hand_pose", "betas")}
            side = str(z["side"]) if "side" in z.files else "right"
        else:
            P = M.tracker_params_camera(str(vd), str(run))
            side = P["side"]
        if side != "right":
            return [f"{vid}: left hand (only right supported)"]
        poses = np.load(rd / "export" / "object_poses.npy").astype(np.float64)
        N = min(len(poses), len(P["global_orient"]))
        R, o = self.world(vid)
        lift = np.array([0.0, 0.0, float(self.cfg.table_height)])
        betas = np.asarray(P["betas"], np.float64)[:10]
        J0 = M.root_joint(betas, "right")
        go = Rotation.from_matrix(R @ Rotation.from_rotvec(P["global_orient"][:N]).as_matrix()).as_rotvec()
        tr = (R @ (J0 + P["transl"][:N] - o).T).T + lift - J0
        W = np.tile(np.eye(4), (N, 1, 1))
        W[:, :3, :3] = R @ poses[:N, :3, :3]
        W[:, :3, 3] = (R @ (poses[:N, :3, 3] - o).T).T + lift
        hand = Hand(global_orient=go, hand_pose=np.asarray(P["hand_pose"][:N], np.float64), betas=betas, transl=tr)
        return [Demo(f"cfhoi_{vid}", float(self.cfg.fps), {"right": hand}, {"right": vid}, {vid: W})]

    def load_mesh(self, obj_id: str) -> trimesh.Trimesh:
        _, run, _ = self._paths(obj_id)
        return trimesh.load(run / "object_mesh.obj", force="mesh", process=False, skip_materials=True)
