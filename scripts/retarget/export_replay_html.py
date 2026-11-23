"""Export a retargeting result (Sharpa hand in MuJoCo) as a self-contained viser HTML, like our reconstruction viewers.

  python scripts/retarget/export_replay_html.py --run-dir RETARGET/sharpa/right/TASK/0 --export-html OUT.html
      (retargeting env; run from anywhere)

Records the retargeted robot (trajectory_mjwp.npz in scene.xml) and the human-hand reference it tracked (orange:
the reconstruction, from trajectory_keypoints.npz) at the reference frame rate (one frame per video frame, warmup
skipped), reusing third_party/do-as-i-do/retargeting/replay_viser.py's scene building, and embeds the recording in
viser's client (scripts/daid/view_recon.write_embed_html). Also writes OUT.json: frames, fps, and the object's
tracking error (simulated object vs the reconstructed one: position cm and rotation deg, median / max over frames,
and the share of frames within 2 cm), i.e. whether the robot really carried the object along the reconstruction.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RT = os.path.join(HERE, "..", "..", "third_party", "do-as-i-do", "retargeting")
sys.path.insert(0, RT)
sys.path.insert(0, os.path.join(HERE, "..", "daid"))
os.chdir(RT)                                                    # the pipeline resolves its assets relative to here

import mujoco  # noqa: E402
import viser  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import replay_viser as rv  # noqa: E402
from retargeting.utils import viser_viewer  # noqa: E402


def write_embed_html(path, data, cam):
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


def object_body(model):
    """The free-floating object body: the body owning a free joint that is not part of the robot."""
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return model.jnt_bodyid[j], model.jnt_qposadr[j]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--export-html", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    scene_path, traj_path = run_dir / "scene.xml", run_dir / "trajectory_mjwp.npz"
    cfg = rv._load_run_config(run_dir / "config.yaml")
    warmup = int(cfg.get("warmup_steps", 0) or 0)
    sim_dt, ref_dt = float(cfg.get("sim_dt", 0.005)), float(cfg.get("ref_dt", 0.0333))
    ref_steps = max(1, int(round(ref_dt / sim_dt)))
    spec = mujoco.MjSpec.from_file(str(scene_path))
    model = spec.compile()
    data = mujoco.MjData(model)
    qpos = rv.load_qpos(traj_path, model.nq)
    data_id, task, emb = run_dir.name, run_dir.parent.name, run_dir.parent.parent.name
    outputs_root = run_dir.parents[3]
    kp_path = outputs_root / "mano" / emb / task / data_id / "trajectory_keypoints.npz"

    viser_viewer.init_viser(app_name="retargeting-export", port=rv_free_port())
    server = viser_viewer._get_server()
    serializer = server.get_scene_serializer()
    body_ids = viser_viewer.build_and_log_scene_from_spec(spec, model, xml_path=scene_path, build_ref=False, build_gui=False)
    mano = rv.ManoOverlay(kp_path, outputs_root, task)
    if mano.available:
        mano.build(server)
    else:
        mano = None

    sim_frames = list(range(warmup, len(qpos), ref_steps))
    ob, adr = object_body(model)
    sim_obj = []
    for k, si in enumerate(sim_frames):
        data.qpos[:] = qpos[si]
        mujoco.mj_kinematics(model, data)
        viser_viewer.log_frame(data, sim_time=0.0, viewer_body_entity_and_ids=body_ids, data_ref=None, record=False)
        if mano is not None:
            mano.show_frame(k)
        if adr is not None:                         # the object's free-joint state: position + quaternion (w, x, y, z)
            sim_obj.append((qpos[si][adr:adr + 3].copy(), qpos[si][adr + 3:adr + 7].copy()))
        serializer.insert_sleep(ref_dt)
    rec = serializer.serialize()

    # default camera: from the front-right, above the object's start position (MuJoCo world: z up)
    c = sim_obj[0][0] if sim_obj else np.zeros(3)
    cam = {"look_at": c.tolist(), "position": (c + np.array([0.45, -0.55, 0.40])).tolist(), "up": [0.0, 0.0, 1.0]}
    write_embed_html(args.export_html, rec, cam)

    info = {"frames": len(sim_frames), "fps": 1.0 / ref_dt, "task": task}
    if sim_obj and kp_path.exists():                  # tracking error vs the reconstructed object trajectory
        kp = np.load(str(kp_path), allow_pickle=True)
        key = next((k for k in ("qpos_obj_right", "qpos_obj_left") if k in kp.files and len(kp[k])), None)
        if key is not None:
            ref = np.asarray(kp[key])
            n = min(len(ref), len(sim_obj))
            e = 100 * np.linalg.norm(np.array([p for p, _ in sim_obj[:n]]) - ref[:n, :3], axis=1)
            sq, rq = np.array([q for _, q in sim_obj[:n]]), ref[:n, 3:7]
            r = np.degrees((Rotation.from_quat(sq[:, [1, 2, 3, 0]]) * Rotation.from_quat(rq[:, [1, 2, 3, 0]]).inv()).magnitude())
            info.update(obj_pos_err_cm_median=float(np.median(e)), obj_pos_err_cm_max=float(e.max()),
                        obj_rot_err_deg_median=float(np.median(r)), obj_rot_err_deg_max=float(r.max()),
                        frames_within_2cm=float(np.mean(e < 2.0)), frames_compared=int(n),
                        obj_pos_err_cm=[round(float(x), 2) for x in e])
    json.dump(info, open(os.path.splitext(args.export_html)[0] + ".json", "w"), indent=1)
    try:
        server.stop()
    except RuntimeError:
        pass
    print(f"wrote {args.export_html} ({os.path.getsize(args.export_html) / 1e6:.1f} MB, {len(sim_frames)} frames); {info}")


def rv_free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    main()
