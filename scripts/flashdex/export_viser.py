"""Export FlashDexRetarget rollouts of our clips as self-contained viser HTMLs (like the reconstruction viewers),
with their tracking metrics.

  bash scripts/flashdex/export_viser.sh --dataset DATA --archive RUN/evals/step<N> [--out OUT] [--clips VID ...]
      (flashdex env + FDR's viser in third_party/viewer_site; defaults: OUT = outputs/flashdex_viser)

Per clip, OUT/VID.html plays the physics rollout of the policy: the Sharpa Wave hand (floating wrist) and the
simulated object (textured SAM3D mesh) on the table disk, with what it tracks: the reconstructed object
(translucent orange) and the human hand (orange skeleton: MANO joints). --ghost-robot adds the kinematic retarget
(translucent blue robot hand, before physics). The camera starts at the video's camera. OUT/VID.json: metrics.py's
numbers (object position / rotation error, AdaDex progress and success, FDR's success, wrist and fingertip errors)
and per-frame curves, plus the archive and policy step it came from.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
import trimesh
import viser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FDR = Path(os.environ.get("FDR", "/data/sky/projects/CounterfactualHOI/build/FlashDexRetarget"))
sys.path.insert(0, str(FDR / "src"))
STORAGE = Path(os.environ.get("PROJECT_STORAGE_ROOT", "/data/sky/projects/CounterfactualHOI"))
WORKDIR = Path(HERE).parents[2]

import metrics as MT  # noqa: E402

FINGERS = MT.FINGERS


def write_embed_html(path, data, cam):
    """viser's single-file client with the recording embedded and the initial camera (as scripts/daid/view_recon)."""
    import base64
    client_html = open(os.path.join(os.path.dirname(viser.__file__), "client", "build", "index.html")).read()
    params = {"initialCameraPosition": ",".join(f"{x:.4f}" for x in cam["position"]),
              "initialCameraLookAt": ",".join(f"{x:.4f}" for x in cam["look_at"]),
              "initialCameraUp": ",".join(f"{x:.4f}" for x in cam["up"])}
    inject = ("<script>"
              f'window.__VISER_EMBED_DATA__="{base64.b64encode(data).decode("ascii")}";'
              "window.__VISER_EMBED_CONFIG__={darkMode:false};"
              "(function(){var q=new URLSearchParams(location.search);"
              "if(!q.has('initialCameraPosition')){"
              f"var d={json.dumps(params)};for(var k in d)q.set(k,d[k]);"
              "try{history.replaceState(null,'',location.pathname+'?'+q.toString()+location.hash);}catch(e){}}})();"
              "</script>")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    open(path, "w").write(client_html.replace("</head>", inject + "</head>", 1))


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, np.float64).reshape(-1))
    return q


class Hand:
    """The right Sharpa hand's visual meshes (one viser frame per body), posed by MuJoCo forward kinematics."""

    def __init__(self, xml):
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        m = self.model
        name = lambda t, i: mujoco.mj_id2name(m, t, i) or ""  # noqa: E731
        self.bodies = [b for b in range(1, m.nbody) if name(mujoco.mjtObj.mjOBJ_BODY, b).startswith(("R_", "right_"))]
        self.tip_sites = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"track_hand_right_{f}_tip") for f in FINGERS]
        self.frames = {}

    def add(self, server, key, tint=None, opacity=None):
        m, frames = self.model, {}
        keep = set(self.bodies)
        for g in range(m.ngeom):
            gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            b = int(m.geom_bodyid[g])
            if (b not in keep or m.geom_group[g] >= 3 or m.geom_rgba[g, 3] <= 0 or gname.startswith("collision_")
                    or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH):
                continue
            mat = m.geom_matid[g]
            col = m.mat_rgba[mat] if mat >= 0 and np.allclose(m.geom_rgba[g], [0.5, 0.5, 0.5, 1]) else m.geom_rgba[g]
            col = np.asarray(col[:3], float)
            if tint is not None:
                col = 0.35 * col + 0.65 * np.asarray(tint)
            if b not in frames:
                frames[b] = server.scene.add_frame(f"/{key}/b{b}", show_axes=False)
            d = m.geom_dataid[g]
            v0, nv, f0, nf = m.mesh_vertadr[d], m.mesh_vertnum[d], m.mesh_faceadr[d], m.mesh_facenum[d]
            server.scene.add_mesh_simple(f"/{key}/b{b}/g{g}", m.mesh_vert[v0:v0 + nv].copy(), m.mesh_face[f0:f0 + nf].copy(),
                                         color=tuple(int(255 * c) for c in col), opacity=opacity,
                                         position=m.geom_pos[g].copy(), wxyz=m.geom_quat[g].copy())
        self.frames[key] = frames

    def fk(self, qpos):
        self.data.qpos[:] = qpos[:self.model.nq]
        mujoco.mj_kinematics(self.model, self.data)

    def pose(self, key, qpos):
        self.fk(qpos)
        for b, h in self.frames[key].items():
            h.position = self.data.xpos[b].copy()
            h.wxyz = self.data.xquat[b].copy()

    def tips(self, Q):
        out = np.zeros((len(Q), 5, 3))
        for t, q in enumerate(Q):
            self.fk(q)
            out[t] = self.data.site_xpos[self.tip_sites]
        return out


def textured_mesh(vid, visual_obj):
    """The SAM3D mesh with its texture when its vertices are FDR's visual.obj's (the import's mesh), else None."""
    tex = STORAGE / "runs" / "fprecon" / vid / "object_mesh.obj"
    if not tex.exists():
        return None
    m = trimesh.load(tex, force="mesh", process=False)
    v = trimesh.load(visual_obj, force="mesh", process=False, skip_materials=True).vertices
    return m if len(m.vertices) == len(v) and np.allclose(m.vertices, v, atol=1e-5) else None


def camera_world(vid, refine, table_height, verts):
    """The video camera's centre in the clip's world (cfhoi_source.world_frame) and its viewing direction."""
    from cfhoi_source import world_frame
    R, o = world_frame(STORAGE / "runs" / refine / vid, verts)
    lift = np.array([0, 0, table_height])
    return R @ (-o) + lift, R @ np.array([0.0, 0, 1])


def skeleton(wrist, joints, names):
    """(T,3), (T,20,3) -> (T, 20, 2, 3) segments wrist-proximal-intermediate-distal-tip per finger."""
    idx = {n: i for i, n in enumerate(names)}
    segs = []
    for f in FINGERS:
        chain = [None] + [idx[f"{f}_{p}"] for p in ("proximal", "intermediate", "distal", "tip")]
        for a, b in zip(chain[:-1], chain[1:]):
            pa = wrist if a is None else joints[:, a]
            segs.append(np.stack([pa, joints[:, b]], 1))
    return np.stack(segs, 1)


def export_clip(args, pt, hands, m, vid, entry, archive, server_port):
    a0, n = int(pt["length_starts"][m]), int(pt["motion_num_frames"][m])
    ref_q = pt["joint_pos"][a0:a0 + n].numpy().astype(np.float64)
    ref_p, ref_R = pt["obj_right_pos"][a0:a0 + n].numpy(), pt["obj_right_rotmat"][a0:a0 + n].numpy()
    names = [str(x) for x in pt["mano_right_joint_names"]]
    mj = pt["mano_right_joints"][a0:a0 + n].numpy()
    mw = pt["mano_right_wrist_pos"][a0:a0 + n].numpy()
    z = np.load(archive / entry["rollout"])
    q, op, oq = z["joint_pos"].astype(np.float64), z["obj_pos"], z["obj_quat"]
    if op.ndim == 3:
        op, oq = op[:, 0], oq[:, 0]
    T = min(len(q), n)
    hand, ref_hand = hands[q.shape[1]], hands[ref_q.shape[1]]      # right-only (28) or bimanual (56) joint vectors
    tips = hand.tips(q[:T])
    ref_tips = mj[:T][:, [names.index(f"{f}_tip") for f in FINGERS]]
    res, curves = MT.clip_metrics(op[:T], oq[:T], ref_p[:T], ref_R[:T], q[:T], ref_q[:T], tips, ref_tips)
    res.update(clip=vid, rollout_frames=int(len(q)), reference_frames=n, fps=args.fps, archive=str(archive),
               env_step=entry.get("first_saved_env_step"), source=args.refine)

    server = viser.ViserServer(port=server_port, verbose=False)
    server.scene.set_up_direction("+z")
    ser = server.get_scene_serializer()
    mesh_dir = Path(args.dataset) / str(pt["right_object_mesh_dirs"][int(pt["motion_object_slot_right"][m])])
    tm = textured_mesh(vid, mesh_dir / "visual.obj")
    vis = trimesh.load(mesh_dir / "visual.obj", force="mesh", process=False, skip_materials=True)
    if tm is not None:
        obj = server.scene.add_mesh_trimesh("/object", tm)
    else:
        obj = server.scene.add_mesh_simple("/object", np.asarray(vis.vertices, np.float32), np.asarray(vis.faces, np.int32),
                                           color=(204, 158, 97))
    ghost = server.scene.add_mesh_simple("/reference_object", np.asarray(vis.vertices, np.float32),
                                         np.asarray(vis.faces, np.int32), color=(255, 140, 40), opacity=0.3)
    disks = pt["support_disks"][m].numpy() if "support_disks" in pt else np.zeros((0, 5))
    for k, (x, y, z_top, r, h) in enumerate(disks):
        if r > 0:
            r_sim = float(np.asarray(args.disk_radii)[np.searchsorted(args.disk_radii, r + 0.1 - 1e-6).clip(max=len(args.disk_radii) - 1)])
            cyl = trimesh.creation.cylinder(radius=r_sim, height=0.01, sections=64)
            server.scene.add_mesh_simple(f"/table/d{k}", np.asarray(cyl.vertices, np.float32), np.asarray(cyl.faces, np.int32),
                                         color=(190, 186, 178), position=(float(x), float(y), float(z_top) - 0.005))
    hand.add(server, "robot")
    if args.ghost_robot:
        ref_hand.add(server, "kinematic", tint=(0.35, 0.6, 1.0), opacity=0.3)
    segs = skeleton(mw, mj, names)
    stride = max(1, int(round(args.fps / args.play_fps)))
    sk = None
    for t in range(0, T, stride):
        with server.atomic():
            hand.pose("robot", q[t])
            if args.ghost_robot:
                ref_hand.pose("kinematic", ref_q[t])
            obj.position, obj.wxyz = op[t].astype(float), oq[t].astype(float)
            ghost.position, ghost.wxyz = ref_p[t].astype(float), mat2quat(ref_R[t])
            if sk is None:
                sk = server.scene.add_line_segments("/human_hand", segs[t].astype(np.float32), colors=(255, 120, 20),
                                                    line_width=3.0)
            else:
                sk.points = segs[t].astype(np.float32)
        ser.insert_sleep(stride / args.fps)
    data = ser.serialize()
    cam_pos, fwd = camera_world(vid, args.refine, args.table_height, vis.vertices)
    look = ref_p[0].astype(float)
    d = cam_pos - look                                  # from the video camera's side, at most 0.9 m away
    cam_pos = look + d / max(np.linalg.norm(d), 1e-6) * min(float(np.linalg.norm(d)), 0.9)
    write_embed_html(str(Path(args.out) / f"{vid}.html"), data,
                     {"position": cam_pos.tolist(), "look_at": look.tolist(), "up": [0.0, 0.0, 1.0]})
    res["curves"] = curves
    json.dump(res, open(Path(args.out) / f"{vid}.json", "w"), indent=1)
    server.stop()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="FDR dataset dir (motion.pt, objects/)")
    ap.add_argument("--archive", required=True, help="a rollout archive: RUN/evals/step<N> or RUN/success")
    ap.add_argument("--out", default=str(WORKDIR / "outputs" / "flashdex_viser"))
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--refine", default="refine_r5", help="the refine round the dataset was imported from")
    ap.add_argument("--table-height", type=float, default=0.8)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--play-fps", type=float, default=30.0)
    ap.add_argument("--ghost-robot", action="store_true")
    ap.add_argument("--disk-radii", type=float, nargs="*", default=[0.08, 0.15, 0.25, 0.4])
    ap.add_argument("--port", type=int, default=8431)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pt = torch.load(Path(args.dataset) / "motion.pt", map_location="cpu", weights_only=False)
    names = [str(x) for x in pt["motion_filename"]]
    archive = Path(args.archive)
    man = json.load(open(archive / "manifest.json"))
    robot_dir = FDR / "assets" / "robot" / "sharpa"
    hands = {h.model.nq: h for h in (Hand(robot_dir / "right.xml"), Hand(robot_dir / "bimanual.xml"))}
    summary = {}
    for e in man["saved"]:
        m = int(e["motion_id"])
        vid = names[m].split("#")[-1].removeprefix("cfhoi_")
        if args.clips and vid not in args.clips:
            continue
        r = export_clip(args, pt, hands, m, vid, e, archive, args.port)
        summary[vid] = {k: v for k, v in r.items() if k != "curves"}
        print(f"{vid}: progress {r['progress']:.2f}, obj {r['obj_pos_err_cm_mean']:.1f} cm / {r['obj_rot_err_deg_mean']:.0f} deg"
              f" mean, success {r['success']} (FDR {r['fdr_success']}) -> {args.out}/{vid}.html "
              f"({os.path.getsize(Path(args.out) / f'{vid}.html') / 1e6:.1f} MB)", flush=True)
    json.dump(summary, open(Path(args.out) / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
