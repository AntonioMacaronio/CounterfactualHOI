"""Animated viser view of one Do-as-I-Do reconstruction: hand + object + table, in the metric camera frame.

  # live viewer (sam3d env)
  python scripts/daid/view_recon.py --video-dir RUN/ --object-id red_apple --port 8081 [--title "apple"]
  # self-contained HTML (recorded animation, no server needed)
  python scripts/daid/view_recon.py --video-dir RUN/ --object-id red_apple --export-html out/apple.html

Like reconstruction/scripts/visualize_3d.py (same layout/hand conventions), plus the table from
reconstruct_table.py, auto-play, and paths/scale inferred from the run directory. The object mesh is
uploaded once and moved per frame; the hand mesh is uploaded once and only its vertices change per frame.

Camera: "Save camera angle" in the live viewer writes RUN/viewer_camera.json (position, look-at, up, fov in
viser world coordinates: the camera frame rotated so the chosen up is +z). It becomes the default view for new browser tabs and for --export-html, which
passes it to the viser client as initialCamera* URL parameters (see scripts/daid/build_recon_html.py).
"""

import argparse
import base64
import glob
import json
import os
import socket
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation

CAMERA_FILE = "viewer_camera.json"


def load_run(args):
    vd = args.video_dir
    cv_dir = os.path.join(vd, "obj_tracking_out", args.object_id, "combined_visualization")
    lay_path = os.path.join(cv_dir, "layout_camera_frame_filled.json")      # fill_unanchored.py output
    if not os.path.exists(lay_path):
        lay_path = os.path.join(cv_dir, "layout_camera_frame_optimized.json")
    layout = json.load(open(lay_path))
    poses = {}
    for o in layout["objects"]:
        fidx = o.get("frame_index", o.get("frame_idx"))   # same keys as upstream visualize_3d.load_layout
        if fidx is None:
            continue
        lts = o["local_to_scene"]
        poses[int(fidx)] = (np.array(lts["quat_wxyz_camera_frame"], float),
                            np.array(lts["translation_camera_frame"], float))
    frames = sorted(poses)

    mesh_frame = json.load(open(os.path.join(vd, "config.json")))["frame_number"]
    mesh = trimesh.load(os.path.join(vd, "video_segmentation", "masks", f"frame_{mesh_frame:06d}_masks",
                                     args.object_id, f"{args.object_id}.obj"), force="mesh")  # SAM3D, textured
    mesh.apply_scale(layout["translation_scale_optimization"]["mesh_scale"])

    hands = np.load(glob.glob(os.path.join(vd, "*", "all_hand_meshes.npz"))[0])
    hv = hands[f"{args.anchor_hand}_vertices"]
    hf = hands[f"{args.anchor_hand}_faces"].astype(np.uint32)
    hvalid = hands[f"{args.anchor_hand}_valid"] if f"{args.anchor_hand}_valid" in hands.files else np.ones(len(hv), bool)

    table = json.load(open(os.path.join(vd, "table", "table.json")))
    table_mesh = trimesh.load(os.path.join(vd, "table", "table_mesh.obj"), force="mesh")
    table_pts = trimesh.load(os.path.join(vd, "table", "table_points.ply"))

    # Viewer "up". GeoCalib's vec3d is the up direction in the camera frame (x right, y down, z fwd); it can
    # be far off on textureless scenes (26-33 deg vs the table on 2 of 4 IMG_5710 clips), hence --up table.
    if args.up == "gravity":
        up = np.array(json.load(open(os.path.join(vd, "gravity.json")))["vec3d"], float)
    else:
        up = np.array(table["plane_normal_up"], float)
    up /= np.linalg.norm(up)
    tilt = float(np.degrees(np.arccos(np.clip(up @ np.array(table["plane_normal_up"]), -1, 1))))
    # Everything lives under a /world node that rotates the camera frame so `up` becomes viser's default +z.
    # (scene.set_up_direction would also work live, but in embedded playback the client applies the
    # initialCamera* URL params before replaying that message, so the start view would come out rotated.)
    R_world = Rotation.align_vectors([[0.0, 0.0, 1.0]], [up])[0]

    frames_dir = os.path.join(vd, "all_frames")
    H, W = cv2.imread(os.path.join(frames_dir, f"{frames[0]:06d}.png")).shape[:2]
    # get_pointmap_dir.py saves 3x3 intrinsics denormalized to pixels of the frame size.
    intr = np.load(os.path.join(frames_dir, f"{frames[0]:06d}_intrinsics.npy"))
    return dict(vd=vd, poses=poses, frames=frames, mesh=mesh, hv=hv, hf=hf, hvalid=hvalid, table=table,
                table_mesh=table_mesh, table_pts=table_pts, up=up, tilt=tilt, R_world=R_world,
                frames_dir=frames_dir, H=H, W=W,
                fov_y=float(2 * np.arctan(H / (2 * intr[1, 1]))))


def default_camera(d):
    """Three-quarter view from the side and above, looking at the middle of the action (between the camera
    and the object), so hand, object, table and the real camera's frustum are all visible."""
    up = d["up"]
    side = np.array([1.0, 0, 0]) - up * up[0]      # camera x, made horizontal w.r.t. the chosen up
    side /= np.linalg.norm(side)
    obj = np.mean([d["poses"][f][1] for f in d["frames"]], axis=0)
    dist = float(np.linalg.norm(obj))
    target = 0.6 * obj
    to_world = d["R_world"].apply                  # camera frame -> viser world (up = +z)
    return {"position": to_world(target + dist * (0.9 * side + 0.7 * up)).tolist(),
            "look_at": to_world(target).tolist(), "up": [0.0, 0.0, 1.0], "fov": None}


def saved_or_default_camera(d):
    p = os.path.join(d["vd"], CAMERA_FILE)
    return json.load(open(p)) if os.path.exists(p) else default_camera(d)


def apply_camera(client, cam):
    client.camera.up_direction = np.array(cam["up"])
    client.camera.position = np.array(cam["position"])
    client.camera.look_at = np.array(cam["look_at"])
    if cam.get("fov"):
        client.camera.fov = cam["fov"]


def build_scene(server, d, args, image_width, jpeg_quality):
    """Static scene + a per-frame update function (shared by the live viewer and the HTML export)."""
    q = d["R_world"].as_quat()                     # xyzw
    server.scene.add_frame("/world", wxyz=(q[3], q[0], q[1], q[2]), show_axes=False)
    server.scene.add_light_ambient("/lights/ambient", intensity=0.7)
    server.scene.add_light_directional("/lights/key", intensity=1.5, wxyz=(0.92, -0.38, 0, 0))
    tm, tp = d["table_mesh"], d["table_pts"]
    server.scene.add_mesh_simple("/world/table/surface", vertices=tm.vertices.astype(np.float32),
                                 faces=tm.faces.astype(np.uint32), color=(170, 140, 100), opacity=0.55)
    server.scene.add_point_cloud("/world/table/points", points=np.asarray(tp.vertices, np.float32),
                                 colors=np.asarray(tp.colors)[:, :3].astype(np.uint8), point_size=0.003)
    obj_frame = server.scene.add_frame("/world/object", show_axes=False)
    server.scene.add_mesh_trimesh("/world/object/mesh", mesh=d["mesh"])     # uploaded once; /object moves per frame
    hand = server.scene.add_mesh_simple(f"/world/hand_{args.anchor_hand}", vertices=d["hv"][0].astype(np.float32),
                                        faces=d["hf"], color=(120, 140, 220), side="double")
    H, W = d["H"], d["W"]

    def update(fidx, show_image=True):
        if fidx in d["poses"]:
            q, t = d["poses"][fidx]
            obj_frame.wxyz, obj_frame.position = q, t
        hi = min(fidx, len(d["hv"]) - 1)
        hand.vertices = d["hv"][hi].astype(np.float32)   # only vertices are re-sent; faces stay
        hand.visible = bool(d["hvalid"][hi])             # HaWoR marks frames with no detected hand
        kw = dict(fov=d["fov_y"], aspect=W / H, scale=args.frustum_scale, color=(40, 40, 40))
        img = cv2.imread(os.path.join(d["frames_dir"], f"{fidx:06d}.png")) if show_image else None
        if img is not None:
            img = cv2.cvtColor(cv2.resize(img, (image_width, int(image_width * H / W))), cv2.COLOR_BGR2RGB)
            server.scene.add_camera_frustum("/world/camera", image=img, format="jpeg", jpeg_quality=jpeg_quality, **kw)
        else:
            server.scene.add_camera_frustum("/world/camera", **kw)

    return update


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def export_html(d, args, fps):
    """Record the animation (one loop) and write a self-contained HTML file using viser's embed mode."""
    server = viser.ViserServer(host="127.0.0.1", port=free_port(), verbose=False)
    serializer = server.get_scene_serializer()          # create first so the static scene is recorded too
    update = build_scene(server, d, args, image_width=args.export_image_width, jpeg_quality=70)
    for f in d["frames"]:
        update(f)
        serializer.insert_sleep(1.0 / fps)
    data = serializer.serialize()
    server.stop()
    write_embed_html(args.export_html, data, saved_or_default_camera(d))
    print(f"wrote {args.export_html} ({os.path.getsize(args.export_html) / 1e6:.1f} MB, "
          f"recording {len(data) / 1e6:.1f} MB, {len(d['frames'])} frames @ {fps} fps, "
          f"camera: {'saved' if os.path.exists(os.path.join(d['vd'], CAMERA_FILE)) else 'default'})")


def write_embed_html(path, data, cam):
    """Write viser's client with a serialized recording embedded and `cam` as the default start view."""
    client_html = open(os.path.join(os.path.dirname(viser.__file__), "client", "build", "index.html")).read()
    params = {"initialCameraPosition": ",".join(f"{x:.4f}" for x in cam["position"]),
              "initialCameraLookAt": ",".join(f"{x:.4f}" for x in cam["look_at"]),
              "initialCameraUp": ",".join(f"{x:.4f}" for x in cam["up"])}
    if cam.get("fov"):
        params["initialCameraFov"] = f"{cam['fov']:.4f}"
    # The client reads the recording from window.__VISER_EMBED_DATA__ and the start camera from URL params.
    # Set the saved camera as the default unless the embedding page already passes initialCamera* params.
    inject = (
        "<script>"
        f'window.__VISER_EMBED_DATA__="{base64.b64encode(data).decode("ascii")}";'
        "window.__VISER_EMBED_CONFIG__={darkMode:false};"
        "(function(){var q=new URLSearchParams(location.search);"
        "if(!q.has('initialCameraPosition')){"
        f"var d={json.dumps(params)};for(var k in d)q.set(k,d[k]);"
        "try{history.replaceState(null,'',location.pathname+'?'+q.toString()+location.hash);}catch(e){}}})();"
        "</script>"
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    open(path, "w").write(client_html.replace("</head>", inject + "</head>", 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--object-id", required=True)
    ap.add_argument("--anchor-hand", default="right")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--title", default=None)
    ap.add_argument("--fps", type=float, default=None, help="default: 24")
    ap.add_argument("--image-width", type=int, default=480, help="frustum image width (smaller = faster)")
    ap.add_argument("--frustum-scale", type=float, default=0.06, help="size of the camera frustum (m)")
    ap.add_argument("--up", choices=["gravity", "table"], default="gravity",
                    help="viewer up direction: GeoCalib gravity (gravity.json) or the reconstructed table normal")
    ap.add_argument("--export-html", default=None, help="write a self-contained HTML recording and exit")
    ap.add_argument("--export-image-width", type=int, default=320, help="frustum image width in the export")
    ap.add_argument("--html-out", default=None,
                    help="live mode: re-export this HTML (in the background) whenever the camera angle is saved")
    args = ap.parse_args()

    d = load_run(args)
    fps = args.fps or 24.0
    if args.export_html:
        export_html(d, args, fps)
        return

    frames = d["frames"]
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.gui.configure_theme(control_layout="collapsible")
    server.gui.add_markdown(
        f"**{args.title or os.path.basename(d['vd'])}**  \nobject: `{args.object_id}` · "
        f"table {d['table']['surface_size_uv_m'][0]:.2f}×{d['table']['surface_size_uv_m'][1]:.2f} m  \n"
        f"up: {'GeoCalib gravity' if args.up == 'gravity' else 'table normal'} ({d['tilt']:.1f}° from the table normal)")
    update = build_scene(server, d, args, image_width=args.image_width, jpeg_quality=90)

    slider = server.gui.add_slider("Frame", min=frames[0], max=frames[-1], step=1, initial_value=frames[0])
    playing = server.gui.add_checkbox("Play", initial_value=True)
    show_img = server.gui.add_checkbox("Show video frame", initial_value=True)
    speed = server.gui.add_slider("Speed", min=0.1, max=2.0, step=0.1, initial_value=1.0)
    save_btn = server.gui.add_button("Save camera angle", icon=viser.Icon.CAMERA)
    reset_btn = server.gui.add_button("Reset camera")

    @slider.on_update
    def _(_):
        update(int(slider.value), show_img.value)

    @save_btn.on_click
    def _(event):
        cam = event.client.camera
        saved = {"position": cam.position.tolist(), "look_at": cam.look_at.tolist(),
                 "up": cam.up_direction.tolist(), "fov": float(cam.fov)}
        json.dump(saved, open(os.path.join(d["vd"], CAMERA_FILE), "w"), indent=2)
        print(f"saved camera -> {os.path.join(d['vd'], CAMERA_FILE)}: {saved}", flush=True)
        msg = "Used as the default view for new tabs"
        if args.html_out:
            # Re-export with the new default camera (~6 s); the live viewer keeps running.
            subprocess.Popen([sys.executable, os.path.abspath(__file__), "--video-dir", d["vd"],
                              "--object-id", args.object_id, "--anchor-hand", args.anchor_hand, "--fps", str(fps),
                              "--up", args.up, "--frustum-scale", str(args.frustum_scale),
                              "--export-html", args.html_out])
            msg += f" and re-exporting {os.path.basename(args.html_out)}"
        event.client.add_notification("Camera angle saved", msg + ".", auto_close_seconds=4.0)

    @reset_btn.on_click
    def _(event):
        apply_camera(event.client, saved_or_default_camera(d))

    @server.on_client_connect
    def _(client):
        apply_camera(client, saved_or_default_camera(d))

    def loop():
        while True:
            if playing.value:
                nxt = int(slider.value) + 1
                slider.value = frames[0] if nxt > frames[-1] else nxt
            time.sleep(1.0 / (fps * speed.value))

    update(frames[0])
    threading.Thread(target=loop, daemon=True).start()
    print(f"viewer ready on port {args.port}", flush=True)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
