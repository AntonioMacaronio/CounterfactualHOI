"""MuJoCo scenes of a clip for the embodiment tools: the robot, the table and the clip's object.

Two robots share one interface:
  Scene("floating", clip)        FlashDexRetarget's floating-wrist Sharpa hand (no arm): what its rollouts contain
                                 (forearm=True adds a generic forearm behind the wrist, for occlusion studies)
  Scene("marvin_m6ccs", clip)    the arm model from build_model.py (scene.xml), in the clip world at its placement

Cameras see geom groups 1 (robot visuals) and 2 (table, floor, object). A `probe` box (group 5, invisible) can be
attached to the palm to measure clearances of a camera housing with mj_geomDistance before any CAD exists.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

import common as C


def _object_stl(clip):
    """The clip's object mesh as STL (MuJoCo does not need its texture); cached under runs/embodiment/cache."""
    import trimesh
    src = Path(clip["mesh_dir"]) / "visual.obj"
    dst = C.RUNS / "cache" / "objects" / f"{Path(clip['mesh_dir']).parent.parent.name}_{clip['name']}.stl"
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        m = trimesh.load(src, force="mesh", process=False, skip_materials=True)
        m = trimesh.Trimesh(np.asarray(m.vertices) * clip.get("mesh_scale", 1.0), np.asarray(m.faces), process=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        m.export(dst)
    return dst


class Scene:
    def __init__(self, kind, clip=None, probe=False, table=None, forearm=False):
        self.kind = kind
        cams = C.load_cameras()
        if kind == "floating":
            src = C.FDR / "assets" / "robot" / "sharpa" / "right.xml"
            root = ET.parse(src).getroot()
            root.find("compiler").set("meshdir", str(src.parent / "meshes"))
            w = root.find("worldbody")
            sc = table or cams["scene"]["table"]
            ET.SubElement(w, "geom", name="table", type="box", group="2", contype="0", conaffinity="0",
                          size=f"{sc['size'][0] / 2} {sc['size'][1] / 2} {sc['thickness'] / 2}",
                          pos=f"{sc['centre'][0]} {sc['centre'][1]} {C.TABLE_Z - sc['thickness'] / 2}")
        else:
            src = C.model_dir(kind) / "scene.xml"
            root = ET.parse(src).getroot()
            root.find("compiler").set("meshdir", str(src.parent / "meshes"))
            w = root.find("worldbody")
        if clip is not None:
            ET.SubElement(root.find("asset"), "mesh", name="clip_object", file=str(_object_stl(clip)))
            b = ET.SubElement(w, "body", name="clip_object", mocap="true")
            ET.SubElement(b, "geom", name="clip_object", type="mesh", mesh="clip_object", group="2", contype="0",
                          conaffinity="0", rgba="0.8 0.62 0.38 1")
        if forearm and kind == "floating":
            # a stand-in for whatever arm carries the hand: a 10 cm thick, 35 cm long forearm behind the wrist
            palm = next(b for b in root.iter("body") if b.get("name") == C.PALM_BODY)
            ET.SubElement(palm, "geom", name="forearm_proxy", type="capsule", fromto="0 0 -0.08 0 0 -0.43", size="0.05",
                          group="1", contype="0", conaffinity="0", density="0", rgba="0.6 0.6 0.65 1")
        if probe:
            palm = next(b for b in root.iter("body") if b.get("name") == C.PALM_BODY)
            ET.SubElement(palm, "geom", name="probe", type="box", size="0.021 0.021 0.0115", group="5", contype="0",
                          conaffinity="0", density="0", rgba="1 0 0 0.4")
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.palm_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, C.PALM_BODY)
        self.table_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
        self.object_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "clip_object") if clip is not None else -1
        self.probe_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "probe") if probe else -1
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
        self.geom_names = names
        # the hand's collision geoms, by part: fingers and thumb (everything but the palm), and the palm
        self.finger_geoms = [g for g, n in enumerate(names) if n.startswith("collision_hand_") and "palm" not in n]
        self.palm_geoms = [g for g, n in enumerate(names) if n.startswith("collision_hand_") and "palm" in n]
        self.thumb_geoms = [g for g in self.finger_geoms if "thumb" in names[g]]
        self.tip_sites = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"track_hand_right_{f}_tip") for f in C.FINGERS]
        if kind != "floating":
            self.arm_adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"arm_j{k + 1}")] for k in range(7)]
            self.hand_adr = [a for a in range(m.nq) if a not in self.arm_adr]
            self.arm_geoms = [g for g, n in enumerate(names) if n.startswith("arm_")]
        self.cams = cams

    # ---- state
    def set_floating(self, q):
        """FDR joint vector: 6 floating-wrist joints + 22 hand joints (right.xml order)."""
        assert self.kind == "floating"
        self.data.qpos[:] = q[: self.model.nq]

    def set_arm(self, q_arm, q_hand):
        self.data.qpos[self.arm_adr] = q_arm
        self.data.qpos[self.hand_adr] = q_hand

    def set_object(self, pos, R):
        self.data.mocap_pos[0] = pos
        self.data.mocap_quat[0] = C.mat2quat(R)

    def fk(self):
        mujoco.mj_kinematics(self.model, self.data)

    # ---- queries
    def palm(self):
        return C.make_T(self.data.xmat[self.palm_id].reshape(3, 3).copy(), self.data.xpos[self.palm_id].copy())

    def tips(self):
        return self.data.site_xpos[self.tip_sites].copy()

    def cam_pose(self, cam):
        return self.palm() @ cam.T_parent_cam if cam.parent == "palm" else cam.T_parent_cam.copy()

    def cam_body(self, cam):
        """The body that carries the camera's own housing in an arm model (-1 if there is none): pass it as
        `bodyexclude` when rendering, or the camera looks at the inside of its housing."""
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{cam.name}_optical")

    def set_probe(self, T_palm_box, size):
        """Move the probe box (pose of its centre in the palm frame, full sizes in metres)."""
        g = self.probe_id
        self.model.geom_sameframe[g] = 0                        # compiled at the body frame: MuJoCo would ignore geom_pos
        self.model.geom_pos[g] = T_palm_box[:3, 3]
        self.model.geom_quat[g] = C.mat2quat(T_palm_box[:3, :3])
        self.model.geom_size[g] = np.asarray(size) / 2
        self.model.geom_rbound[g] = np.linalg.norm(np.asarray(size) / 2)

    # Distances are computed from points on the geoms' surfaces, not with mj_geomDistance: its convex-hull GJK
    # returns 0 for some of the detailed vendor meshes (the Marvin's Link7, some object meshes) however far apart.
    def local_points(self, g, n=400):
        """Points on geom g's surface in the geom's frame: mesh vertices plus samples on its faces; box corners,
        edges and faces for boxes; cached."""
        cache = self.__dict__.setdefault("_pts", {})
        if (g, n) in cache:
            return cache[(g, n)]
        m = self.model
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            import trimesh
            i = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[i]: m.mesh_vertadr[i] + m.mesh_vertnum[i]].astype(np.float64)
            f = m.mesh_face[m.mesh_faceadr[i]: m.mesh_faceadr[i] + m.mesh_facenum[i]]
            tm = trimesh.Trimesh(v, f, process=False)
            sp, _ = trimesh.sample.sample_surface(tm, n, seed=0)
            keep = v if len(v) <= 4 * n else v[np.random.default_rng(0).choice(len(v), 4 * n, replace=False)]
            P = np.concatenate([keep, np.asarray(sp)])
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            t = np.linspace(-1, 1, 7)
            a, b, c = np.meshgrid(t, t, t, indexing="ij")
            P = np.stack([a, b, c], -1).reshape(-1, 3)
            P = P[np.abs(P).max(1) > 0.999] * m.geom_size[g]
        else:                                                    # cylinder / sphere / capsule: a coarse shell
            r, h = m.geom_size[g][0], (m.geom_size[g][1] if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_SPHERE else 0.0)
            th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            P = np.array([[r * np.cos(a), r * np.sin(a), z] for a in th for z in (-h, 0, h)])
        cache[(g, n)] = P
        return P

    def world_points(self, geoms, n=400):
        d = self.data
        return np.concatenate([self.local_points(g, n) @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] for g in geoms])

    def above_table(self, P):
        """Height of the lowest of the world points P above the table top, counting only points over the table's
        footprint (inf when none is)."""
        g = self.table_id
        c, h = self.data.geom_xpos[g], self.model.geom_size[g]
        over = (np.abs(P[:, 0] - c[0]) <= h[0]) & (np.abs(P[:, 1] - c[1]) <= h[1])
        return float(P[over, 2].min() - C.TABLE_Z) if over.any() else np.inf

    def table_clearance(self, geoms):
        return self.above_table(self.world_points(geoms)) if len(geoms) else np.inf

    @staticmethod
    def points_to_box(P, T_box, half):
        """Signed distance of the nearest of points P to a box (pose T_box, half sizes): negative inside."""
        q = np.abs((P - T_box[:3, 3]) @ T_box[:3, :3]) - half
        out = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
        return float(out.min())

    def table_box(self):
        g = self.table_id
        return C.make_T(self.data.geom_xmat[g].reshape(3, 3), self.data.geom_xpos[g]), self.model.geom_size[g].copy()

    def probe_corners(self):
        g = self.probe_id
        R, p, s = self.data.geom_xmat[g].reshape(3, 3), self.data.geom_xpos[g], self.model.geom_size[g]
        sg = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float) * s
        return p + sg @ R.T
