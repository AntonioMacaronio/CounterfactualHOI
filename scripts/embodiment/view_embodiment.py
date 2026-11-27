"""Interactive 3D view of the embodiment playing a clip: the arm with the Sharpa hand, the D405 on its mount, the
table, the object, both cameras with what they see.

  bash scripts/embodiment/view_embodiment.sh --arm vega_1p [--clips VID ...] [--ik-dir DIR] [--out OUT]
      (the flashdex env with FDR's viser, like scripts/flashdex/export_viser.sh; no GPU)
  -> OUT/<clip>.html   self-contained: open it in a browser, drag to orbit, scroll to zoom, it plays in a loop
     OUT/<clip>.png    three stills at the middle of the carry: overview, the wrist from its palm side, the scene
                       from where the exo camera stands
     OUT/index.html    links to the clips
     OUT defaults to $PROJECT_STORAGE_ROOT/runs/embodiment/viewer/<arm>/<name of the IK result>

What is shown
  * the robot from build_model.py at its placement (for the Vega: the whole body, frozen, and the moving right arm),
    with the mount plate and the D405 under the wrist, in the vendor's colours; its meshes are the thinned copies
    from viewer_meshes.py (within 1 mm of the CAD for what moves, 2 mm for the frozen body);
  * the joint angles of an IK result on the dataset's reference motions (FDR's kinematic retarget of each clip): by
    default ik_pyroki.py's fingertips mode, else ik.py's; the object follows the reference's poses;
  * the reference's floating hand as a translucent orange ghost (--no-ghost to hide): where the two differ, the
    solver lifted or tilted the palm to keep the wrist hardware off the table;
  * the wrist camera (moves with the hand) and the exo camera as frustums carrying their current depth image at the
    policy's resolution: near is bright, dark red is invalid (out of range, or seen by one imager only). The exo
    frustum is drawn for the window of the stream that the policy sees, centred on the optical axis;
  * the robot's own view, for a robot with a head camera (the Vega's ZED): a green frustum carrying a shaded colour
    view from there (nominal 90° x 60° field of view; not a policy input). Checkboxes at the top right hide each
    frustum (and its label); the choice carries over to the other clips' viewers, and ?hide=wrist,exo,head in the URL
    starts with them hidden.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import viser

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "flashdex"))

import common as C  # noqa: E402
from cameras import Cameras, colorize  # noqa: E402
from export_viser import textured_mesh  # noqa: E402  (scripts/flashdex)
from preview import look_at, render_scene  # noqa: E402
from scene import Scene  # noqa: E402


OVERVIEW = ((-1.35, 0.75, 1.65), (0.2, -0.15, 0.95))      # eye and target of the first view: from the table's far left corner
# The Vega's head camera (Dexmate's model: a ZED stereo camera, the left lens' optical frame `zed_left_camera` in the
# head). Its model and lens are not given there, so the field of view is nominal. Not a policy input; shown as the
# robot's point of view.
HEAD_CAMERA = {"body": "zed_left_camera", "shell": "head_l3", "fov_deg": (90.0, 60.0), "image": (168, 96)}
_CLIENT = None


def client_page():
    """viser's single-file client with one line added: its scene tree (the store of every scene node and the actions
    that change them, the ones the client's own "Scene tree" panel uses) is put on window.__cfhoiSceneTree, so that a
    script in the page can hide and show nodes while the recording plays. The client's code is zstd-compressed inside
    the page; it is decompressed, patched and compressed again. (The client's own window.__viserSceneTree is a
    read-only test hook without the actions.)"""
    global _CLIENT
    if _CLIENT is None:
        import base64
        import re
        import zstandard
        page = (Path(viser.__file__).parent / "client" / "build" / "index.html").read_text()
        c = re.search(r'data-c="([^"]+)"', page).group(1)
        cs = int(re.search(r'data-cs="(\d+)"', page).group(1))
        code = zstandard.ZstdDecompressor().decompress(base64.b64decode(c), max_output_size=cs).decode()
        hook = re.compile(r"sceneTreeActions:([A-Za-z_$][\w$]*)\.actions")
        if len(hook.findall(code)) != 1:
            raise RuntimeError(f"viser {viser.__version__}: cannot find where the scene tree is created; update client_page()")
        code = hook.sub(lambda mm: f"sceneTreeActions:(window.__cfhoiSceneTree={mm.group(1)},{mm.group(1)}.actions)", code)
        raw = code.encode()
        c2 = base64.b64encode(zstandard.ZstdCompressor(level=19).compress(raw)).decode()
        _CLIENT = page.replace(f'data-c="{c}"', f'data-c="{c2}"', 1).replace(f'data-cs="{cs}"', f'data-cs="{len(raw)}"', 1)
    return _CLIENT


TOGGLE_JS = """
(function(){
  var boxes=function(){return Array.prototype.slice.call(document.querySelectorAll('#cfhoi-toggles input'));};
  var q=new URLSearchParams(location.search), hide=(q.get('hide')||'').split(',');
  boxes().forEach(function(b){
    var k='cfhoi-show-'+b.dataset.key, v=null;
    try{v=localStorage.getItem(k);}catch(e){}
    if(hide.indexOf(b.dataset.key)>=0) b.checked=false; else if(v!==null) b.checked=(v==='1');
    b.addEventListener('change',function(){try{localStorage.setItem(k,b.checked?'1':'0');}catch(e){} apply();});
  });
  function apply(){
    var st=window.__cfhoiSceneTree; if(!st) return;
    boxes().forEach(function(b){ JSON.parse(b.dataset.nodes).forEach(function(n){
      var node=st.store.get(n);
      if(node && node.overrideVisibility!==b.checked) st.actions.updateNodeAttributes(n,{overrideVisibility:b.checked});
    });});
  }
  setInterval(apply,250);       /* nodes appear as the recording plays, and again when it loops */
})();
"""


def write_viewer_html(path, data, cam, toggles):
    """The recording in one HTML file (as scripts/flashdex/export_viser.write_embed_html) on the patched client, with
    a panel of checkboxes at the top right that hide and show groups of scene nodes. toggles: (key, label, node
    names). A choice is remembered for the other viewers of the site (localStorage); ?hide=key1,key2 starts hidden."""
    import base64
    import html as H
    import json
    params = {"initialCameraPosition": ",".join(f"{x:.4f}" for x in cam["position"]),
              "initialCameraLookAt": ",".join(f"{x:.4f}" for x in cam["look_at"]),
              "initialCameraUp": ",".join(f"{x:.4f}" for x in cam["up"])}
    head = ("<script>"
            f'window.__VISER_EMBED_DATA__="{base64.b64encode(data).decode("ascii")}";'
            "window.__VISER_EMBED_CONFIG__={darkMode:false};"
            "(function(){var q=new URLSearchParams(location.search);"
            "if(!q.has('initialCameraPosition')){"
            f"var d={json.dumps(params)};for(var k in d)q.set(k,d[k]);"
            "try{history.replaceState(null,'',location.pathname+'?'+q.toString()+location.hash);}catch(e){}}})();"
            "</script>")
    boxes = "".join(f'<label style="display:block;cursor:pointer"><input type="checkbox" checked data-key="{H.escape(k)}" '
                    f"data-nodes='{H.escape(json.dumps(nodes))}'> {H.escape(label)}</label>" for k, label, nodes in toggles)
    panel = ('<div id="cfhoi-toggles" style="position:fixed;top:10px;right:10px;z-index:1000;background:rgba(255,255,255,0.9);'
             'border:1px solid #ccc;border-radius:6px;padding:5px 10px;font:13px -apple-system,system-ui,sans-serif;color:#222">'
             f"{boxes}</div><script>{TOGGLE_JS}</script>")
    page = client_page().replace("</head>", head + "</head>", 1).replace("</body>", panel + "</body>", 1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(page)


def geom_mesh(m, g, cache=None):
    """A visible geom as (vertices, faces) in the geom's frame. Meshes come from `cache` (viewer_meshes.py: thinned
    within 1-2 mm of the original, split at sharp edges) when it has them, else at full resolution."""
    t = m.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        i = m.geom_dataid[g]
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, i)
        f = cache / f"{name}.npz" if cache else None
        if f is not None and f.exists():
            z = np.load(f)
            v = (z["vertices"].astype(np.float64) - m.mesh_pos[i]) @ C.quat2mat(m.mesh_quat[i])   # file -> this model's mesh frame
            return v.astype(np.float32), z["faces"].astype(np.int32)
        v = m.mesh_vert[m.mesh_vertadr[i]: m.mesh_vertadr[i] + m.mesh_vertnum[i]].copy()
        fc = m.mesh_face[m.mesh_faceadr[i]: m.mesh_faceadr[i] + m.mesh_facenum[i]].copy()
        return v.astype(np.float32), fc.astype(np.int32)
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        tm = trimesh.creation.box(extents=2 * m.geom_size[g])
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        tm = trimesh.creation.cylinder(radius=m.geom_size[g][0], height=2 * m.geom_size[g][1], sections=24)
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        tm = trimesh.creation.capsule(radius=m.geom_size[g][0], height=2 * m.geom_size[g][1])
    else:
        return None
    return np.asarray(tm.vertices, np.float32), np.asarray(tm.faces, np.int32)


class Bodies:
    """A MuJoCo model's visible geoms in viser: one frame per body, its geoms as children."""

    def __init__(self, server, model, key, skip=(), cache=None, tint=None, opacity=None, only_prefix=None):
        self.m, self.frames, self.last = model, {}, {}
        m = model
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            b = int(m.geom_bodyid[g])
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "world"
            if (m.geom_group[g] not in (1, 2) or m.geom_rgba[g, 3] <= 0 or name.startswith("collision_") or name in skip
                    or m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE):
                continue
            if only_prefix and not bname.startswith(only_prefix):
                continue
            vf = geom_mesh(m, g, cache)
            if vf is None:
                continue
            col = np.asarray(m.geom_rgba[g, :3], float)
            if tint is not None:
                col = 0.3 * col + 0.7 * np.asarray(tint)
            if b not in self.frames:
                self.frames[b] = server.scene.add_frame(f"/{key}/b{b}", show_axes=False)
            server.scene.add_mesh_simple(f"/{key}/b{b}/g{g}", vf[0], vf[1], color=tuple(int(255 * c) for c in col), opacity=opacity,
                                         position=m.geom_pos[g].copy(), wxyz=m.geom_quat[g].copy())

    def pose(self, data):
        """Send the bodies that moved since the last call (the Vega's frozen body is sent once)."""
        for b, h in self.frames.items():
            p, q = data.xpos[b].copy(), data.xquat[b].copy()
            lp = self.last.get(b)
            if lp is None or np.abs(lp[0] - p).max() > 1e-6 or np.abs(lp[1] - q).max() > 1e-6:
                h.position, h.wxyz = p, q
                self.last[b] = (p, q)


def default_ik_dir(arm, tag):
    for d in (C.RUNS / "ik_pyroki" / arm / f"{tag}_fingertips", C.RUNS / "ik" / arm / tag):
        if d.exists() and any(d.glob("*.npz")):
            return d
    raise SystemExit(f"no IK result for {arm}: run ik_pyroki.py or ik.py first")


def export_clip(args, clip, ik_dir, out, cams, port):
    from PIL import Image
    z = np.load(ik_dir / f"{clip['name']}.npz")
    arm_q, hand_q = z["arm_q"].astype(float), z["hand_q"].astype(float)
    T = min(len(arm_q), clip["frames"])
    sc = Scene(args.arm, clip)
    m, d = sc.model, sc.data
    palm = sc.palm_id
    server = viser.ViserServer(port=port, verbose=False)
    server.scene.set_up_direction("+z")
    ser = server.get_scene_serializer()
    server.scene.add_grid("/floor", width=4.0, height=4.0, position=(0.3, -0.3, 0.0), cell_size=0.25, section_size=1.0)
    cache = C.model_dir(args.arm) / "viewer_meshes"
    if not (cache / "report.json").exists():
        print(f"[view] no thinned meshes in {cache} (run viewer_meshes.py): the robot is sent at full resolution")
    robot = Bodies(server, m, "robot", skip=("clip_object",), cache=cache)
    # the object: the textured SAM3D mesh when it is the one the dataset was built from
    visual = Path(clip["mesh_dir"]) / "visual.obj"
    tm = textured_mesh(clip["name"], visual)
    if tm is not None:
        obj = server.scene.add_mesh_trimesh("/object", tm)
    else:
        vis = trimesh.load(visual, force="mesh", process=False, skip_materials=True)
        obj = server.scene.add_mesh_simple("/object", np.asarray(vis.vertices, np.float32), np.asarray(vis.faces, np.int32), color=(204, 158, 97))
    # FDR's floating hand, for comparison
    ghost = None
    if args.ghost:
        fl = Scene("floating")
        ghost = (fl, Bodies(server, fl.model, "fdr_hand", tint=(1.0, 0.55, 0.15), opacity=0.3, cache=cache, only_prefix=("R_", "right_")))
    # cameras: frustums in the optical convention (z forward, y down), each with its depth image
    frusta = {}
    for cam, parent, scale, colour in ((cams.wrist, f"/robot/b{palm}", 0.07, (30, 110, 220)), (cams.exo, "", 0.16, (220, 90, 30))):
        w, h = cam.policy["width"], cam.policy["height"]
        fy = cam.intrinsics()[1]
        Tc = cam.T_parent_cam
        frusta[cam.name] = server.scene.add_camera_frustum(
            f"{parent}/{cam.name}", fov=float(2 * np.arctan(h / 2 / fy)), aspect=w / h, scale=scale, color=colour, line_width=2.0,
            position=Tc[:3, 3].copy(), wxyz=C.mat2quat(Tc[:3, :3]), format="png")
    server.scene.add_label("/labels/exo", "exo D435", position=cams.exo.T_parent_cam[:3, 3] + np.array([0, 0, 0.06]))
    server.scene.add_label(f"/robot/b{palm}/label", "wrist D405", position=cams.wrist.T_parent_cam[:3, 3] + np.array([0.06, 0, 0]))
    # the robot's own point of view: a head camera, if the model has one (the Vega's ZED, left lens). Its head is frozen,
    # so the frustum is placed once; it carries a shaded colour view of the scene from there.
    head = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, HEAD_CAMERA["body"])
    if head >= 0:
        sc.set_arm(arm_q[0], hand_q[0])
        sc.fk()
        T_head = C.make_T(d.xmat[head].reshape(3, 3).copy(), d.xpos[head].copy())
        head_shell = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, HEAD_CAMERA["shell"])      # the lens sits inside it
        fh, fv = HEAD_CAMERA["fov_deg"]
        hw, hh = HEAD_CAMERA["image"]
        frusta["head_cam"] = server.scene.add_camera_frustum(
            "/head_cam", fov=float(np.radians(fv)), aspect=hw / hh, scale=0.12, color=(40, 150, 70), line_width=2.0,
            position=T_head[:3, 3], wxyz=C.mat2quat(T_head[:3, :3]), format="jpeg", jpeg_quality=80)
        server.scene.add_label("/labels/head", "head camera", position=T_head[:3, 3] + np.array([0, 0, 0.07]))

    stride = max(1, int(round(args.fps / args.play_fps)))
    still_t = int(np.argmin(np.linalg.norm(clip["obj_pos"][:T] - (clip["obj_pos"][0] + clip["obj_pos"][T - 1]) / 2, axis=1)))   # mid-carry
    for t in range(0, T, stride):
        sc.set_arm(arm_q[t], hand_q[t])
        sc.set_object(clip["obj_pos"][t], clip["obj_rot"][t])
        sc.fk()
        with server.atomic():
            robot.pose(d)
            obj.position, obj.wxyz = clip["obj_pos"][t].copy(), C.mat2quat(clip["obj_rot"][t])
            if ghost:
                ghost[0].set_floating(clip["joint_pos"][t])
                ghost[0].fk()
                ghost[1].pose(ghost[0].data)
            if args.depth:
                for cam in (cams.wrist, cams.exo):
                    img = cam.render(m, d, sc.cam_pose(cam), bodyexclude=sc.cam_body(cam))
                    dep, val = cam.observe(img)
                    frusta[cam.name].image = colorize(dep, val, cam.z_min, cam.z_max)      # the policy's resolution
            if head >= 0:
                frusta["head_cam"].image = render_scene(m, d, T_head, hw, hh, fh, bodyexclude=head_shell)
        ser.insert_sleep(stride / args.fps)
        if abs(t - still_t) < stride:                            # stills at the middle of the carry
            Tp = sc.palm()
            side = np.cross(Tp[:3, 2], [0, 0, 1.0])
            side /= np.linalg.norm(side)
            wrist = Tp[:3, 3] - 0.03 * Tp[:3, 2]
            # look at the wrist from its palm side (where the camera sits), from whichever side keeps the eye above the table
            eyes = [wrist + 0.42 * v / np.linalg.norm(v) + [0, 0, 0.10] for v in (Tp[:3, 0] + 0.6 * side, Tp[:3, 0] - 0.6 * side, side, -side)]
            eye = next((e for e in eyes if e[2] > C.TABLE_Z + 0.12), eyes[2])
            views = [render_scene(m, d, look_at(OVERVIEW[0], OVERVIEW[1]), 640, 480, 48.0),
                     render_scene(m, d, look_at(eye, wrist), 640, 480, 40.0),
                     render_scene(m, d, cams.exo.T_parent_cam, 640, 480, cams.exo.fov_deg()[0], bodyexclude=sc.cam_body(cams.exo))]
            Image.fromarray(np.concatenate(views, 1)).save(out / f"{clip['name']}.png")
    toggles = [("wrist", "wrist camera frustum", [f"/robot/b{palm}/{cams.wrist.name}", f"/robot/b{palm}/label"]),
               ("exo", "exo camera frustum", [f"/{cams.exo.name}", "/labels/exo"])]
    if head >= 0:
        toggles.append(("head", "head camera frustum (robot's view)", ["/head_cam", "/labels/head"]))
    write_viewer_html(out / f"{clip['name']}.html", ser.serialize(),
                      {"position": list(OVERVIEW[0]), "look_at": list(OVERVIEW[1]), "up": [0.0, 0.0, 1.0]}, toggles)
    server.stop()
    return os.path.getsize(out / f"{clip['name']}.html") / 1e6, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="vega_1p")
    ap.add_argument("--dataset", default=str(C.STORAGE / "runs" / "flashdex" / "data_r10_flat"))
    ap.add_argument("--ik-dir", default=None, help="an IK result directory (default: ik_pyroki fingertips on the reference clips, else ik.py)")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--play-fps", type=float, default=30.0)
    ap.add_argument("--no-depth", dest="depth", action="store_false")
    ap.add_argument("--no-ghost", dest="ghost", action="store_false")
    ap.add_argument("--port", type=int, default=8437)
    args = ap.parse_args()
    ik_dir = Path(args.ik_dir) if args.ik_dir else default_ik_dir(args.arm, Path(args.dataset).name + "_reference")
    if "rollouts" in ik_dir.name:                                # object and ghost would come from the reference, not the rollout
        raise SystemExit("IK results on rollouts are not supported yet: the viewer takes object poses from the dataset")
    out = Path(args.out) if args.out else C.RUNS / "viewer" / args.arm / ik_dir.name
    out.mkdir(parents=True, exist_ok=True)
    cams = Cameras()
    rows = []
    for clip in C.load_pack(args.dataset):
        if (args.clips and clip["name"] not in args.clips) or not (ik_dir / f"{clip['name']}.npz").exists():
            continue
        mb, T = export_clip(args, clip, ik_dir, out, cams, args.port)
        rows.append((clip["name"], mb, T))
        print(f"{clip['name']}: {T} frames -> {out}/{clip['name']}.html ({mb:.1f} MB)", flush=True)
    cfg = C.load_arm(args.arm)
    links = "\n".join(f'<li><a href="{n}.html">{n}</a> ({T / args.fps:.1f} s) &nbsp; <a href="{n}.png">stills</a></li>' for n, _, T in rows)
    (out / "index.html").write_text(
        f"<!doctype html><meta charset='utf-8'><title>{args.arm} embodiment</title><body style='font-family:sans-serif;max-width:60em;margin:2em auto'>"
        f"<h2>{cfg['description']} + Sharpa Wave + wrist D405</h2><p>IK result: <code>{ik_dir}</code>. Orange ghost: FlashDexRetarget's "
        f"floating hand. Blue frustum: wrist D405, orange frustum: exo D435, each with its depth image (near = bright, dark red = invalid).</p>"
        f"<ul>{links}</ul></body>\n")
    print(f"-> {out}/index.html")


if __name__ == "__main__":
    main()
