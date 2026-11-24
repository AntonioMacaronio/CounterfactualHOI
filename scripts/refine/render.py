"""Images the refinement agent looks at (OpenCV BGR uint8 arrays; encode() turns them into API image blocks).

overlay(): the real frame with the rendered object (orange outline + light fill, or the textured render blended in),
the rendered hand (blue) and the SAM3 object mask (green outline), the detected 2D hand keypoints (white dots) joined
to MANO's projected joints (blue dots), plus the object's axes (long = red, mid = green, short = blue), cropped to the
hand-object region. view(): a gravity-aligned virtual camera ("side": horizontal, from
the real camera's right; "top": looking straight down, forward = up in the image) showing object, hand and table,
for contact and height. view_orbit(): the same scene from any azimuth / elevation around the action or zoomed onto
the contact region. In the virtual views the hand is blue, red where it crosses the object's surface or is inside
it, green where it touches it (< 3 mm). Also contact sheets, zoom crops and metric plots.
"""

import base64
import io

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from clip import Clip, State, phase_of  # noqa: E402
from Utils import make_mesh_tensors, nvdiffrast_render  # noqa: E402

ORANGE, BLUE, GREEN = (40, 150, 240), (235, 140, 90), (60, 220, 60)
AXIS_COLORS = [(40, 40, 230), (40, 200, 40), (230, 90, 40)]       # BGR: long red, mid green, short blue


def encode(img, max_edge=1400, quality=88):
    h, w = img.shape[:2]
    s = min(1.0, max_edge / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.standard_b64encode(buf.tobytes()).decode()}}


def _render(clip, mesh_tensors, T, K, H, W, light=True):
    col, dep, _ = nvdiffrast_render(K=K, H=H, W=W, glctx=clip.glctx, mesh_tensors=mesh_tensors, use_light=light,
                                    ob_in_cams=torch.as_tensor(T[None], device="cuda", dtype=torch.float))
    c = (col[0].cpu().numpy()[..., ::-1] * 255).clip(0, 255).astype(np.uint8)
    return c, dep[0].cpu().numpy()


def render_frame(clip, state, i):
    """Full-resolution object and hand renders in the real camera: (obj_color, obj_depth, hand_color, hand_depth)."""
    clip.use(state)
    oc, od = _render(clip, clip.mt, state.obj[i], clip.K, clip.H, clip.W)
    hc, hd = _render(clip, clip.hand_tensors(clip.hand_verts(state, i)), np.eye(4), clip.K, clip.H, clip.W)
    return oc, od, hc, hd


def action_box(clip, state, i, pad=0.45, min_size=260):
    """Pixel box around the object render, the hand and the object mask."""
    clip.use(state)
    pts = []
    T = state.obj[i]
    for v in (clip.sub_v @ T[:3, :3].T + T[:3, 3], clip.hand_verts(state, i)):
        p = (clip.K @ v.T).T
        pts.append(p[:, :2] / p[:, 2:3])
    om, _ = clip.masks_full(i)
    ys, xs = np.nonzero(om)
    if len(xs):
        pts.append(np.stack([xs, ys], 1)[:: max(1, len(xs) // 500)])
    pts = np.concatenate(pts)
    x0, y0 = np.percentile(pts, 1, 0)
    x1, y1 = np.percentile(pts, 99, 0)
    w, h = max(x1 - x0, min_size), max(y1 - y0, min_size)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = max(w, h * 16 / 9) * (1 + pad)
    x0, x1 = cx - s / 2, cx + s / 2
    y0, y1 = cy - s * 9 / 32, cy + s * 9 / 32
    return [int(max(0, x0)), int(max(0, y0)), int(min(clip.W, x1)), int(min(clip.H, y1))]


def overlay(clip, state, i, mode="outline", box="auto", axes=True, label=None):
    clip.use(state)
    img = clip.rgb(i).copy()
    oc, od, hc, hd = render_frame(clip, state, i)
    so, sh = od > 0, hd > 0
    vis_o = so & ~(sh & (hd < od))                     # object pixels not covered by the hand
    if mode == "blend":
        img[vis_o] = (0.35 * img[vis_o] + 0.65 * oc[vis_o]).astype(np.uint8)
        img[sh] = (0.6 * img[sh] + 0.4 * np.array(BLUE)).astype(np.uint8)
    else:
        img[so] = (0.75 * img[so] + 0.25 * np.array(ORANGE)).astype(np.uint8)
    om, _ = clip.masks_full(i)
    for m, col, th in ((om, GREEN, 2), (so, ORANGE, 2), (sh, BLUE, 2)):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, col, th)
    kp = clip.kp2d[i]
    if np.isfinite(kp[:, 0]).sum() >= 8:                 # detected keypoints (white) vs MANO joints (blue)
        mp = clip.hand_kp_proj(state, i)
        for a, b in zip(kp, mp):
            if np.isfinite(a[0]):
                cv2.line(img, tuple(int(x) for x in a), tuple(int(x) for x in b), (200, 200, 200), 1)
                cv2.circle(img, tuple(int(x) for x in a), 4, (255, 255, 255), -1)
            cv2.circle(img, tuple(int(x) for x in b), 4, BLUE, -1)
    if axes:
        T = state.obj[i]
        c = clip.obj_center(state, i)
        pc = clip.K @ c
        for k in range(3):
            e = c + T[:3, :3] @ clip.axes_obj[:, k] * 0.5 * max(clip.extent[k], 0.03)
            pe = clip.K @ e
            cv2.arrowedLine(img, tuple(int(x) for x in pc[:2] / pc[2]), tuple(int(x) for x in pe[:2] / pe[2]),
                            AXIS_COLORS[k], 3, tipLength=0.15)
    if box == "auto":
        box = action_box(clip, state, i)
    if box:
        x0, y0, x1, y1 = box
        img = img[y0:y1, x0:x1]
    h = img.shape[0]
    txt = label if label is not None else f"frame {i}" + (f" ({phase_of(state, i)})" if phase_of(state, i) else "")
    cv2.putText(img, txt, (10, max(28, h // 18)), cv2.FONT_HERSHEY_SIMPLEX, max(0.7, h / 700), (255, 255, 255), 4)
    cv2.putText(img, txt, (10, max(28, h // 18)), cv2.FONT_HERSHEY_SIMPLEX, max(0.7, h / 700), (20, 20, 20), 2)
    return img


def view(clip, state, i, which="side", size=(720, 540), dist=None, label=None):
    """Gravity-aligned virtual view of object + hand + table around the action at frame i."""
    clip.use(state)
    up, right, fwd = clip.up, clip.right, clip.fwd
    c_obj = clip.obj_center(state, i)
    c = 0.5 * (c_obj + clip.hand_verts(state, i).mean(0)) if clip.hand_valid[i] else c_obj
    dist = dist or max(0.45, 3.0 * float(clip.extent.max()))
    if which == "top":
        z_axis, y_axis = -up, -fwd                     # look down; image up = away from the real camera
    elif which == "front":
        z_axis, y_axis = fwd, -up                      # the real camera's direction, but level
    else:
        z_axis, y_axis = -right, -up                   # from the real camera's right, level
    x_axis = np.cross(y_axis, z_axis)
    R_vc = np.stack([x_axis, y_axis, z_axis])          # camera-frame vector -> virtual camera
    t_vc = -R_vc @ (c - dist * z_axis)
    T_vc = np.eye(4)
    T_vc[:3, :3], T_vc[:3, 3] = R_vc, t_vc
    W, H = size
    f = 0.9 * W
    Kv = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    img = np.full((H, W, 3), 245, np.uint8)
    # table: shaded quad + 5 cm grid around the action, on the fitted plane
    n, p0 = clip.table_n, clip.table_p
    cp = c - ((c - p0) @ n) * n
    e1 = right - (right @ n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    half = 0.4
    quad = [cp + a * half * e1 + b * half * e2 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    proj = lambda P: (Kv @ (R_vc @ np.asarray(P).T + t_vc[:, None])).T
    q = proj(quad)
    if (q[:, 2] > 0.01).all():
        cv2.fillPoly(img, [np.int32(q[:, :2] / q[:, 2:3])], (205, 200, 190))
        for s in np.arange(-half, half + 1e-6, 0.05):
            for a, b in (((s, -half), (s, half)), ((-half, s), (half, s))):
                P = proj([cp + a[0] * e1 + a[1] * e2, cp + b[0] * e1 + b[1] * e2])
                if (P[:, 2] > 0.01).all():
                    cv2.line(img, tuple(np.int32(P[0, :2] / P[0, 2])), tuple(np.int32(P[1, :2] / P[1, 2])), (175, 170, 160), 1)
    oc, od = _render(clip, clip.mt, T_vc @ state.obj[i], Kv, H, W)
    hv = clip.hand_verts(state, i)
    hc, hd = _render(clip, clip.hand_tensors(hv, clip.hand_colors(state, i, hv)), T_vc, Kv, H, W)
    hd[~(hd > 0)] = np.inf
    od[~(od > 0)] = np.inf
    so, sh = np.isfinite(od) & (od <= hd), np.isfinite(hd) & (hd < od)
    img[so] = oc[so]
    img[sh] = hc[sh]
    txt = label if label is not None else f"frame {i} {which} view"
    cv2.putText(img, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    return img


AZ_NAMES = {0: "from the camera's right", 45: "right-behind", 90: "from behind", 135: "left-behind",
            180: "from the camera's left", 225: "left-front", 270: "from the camera's side", 315: "right-front"}


def view_orbit(clip, state, i, az=0.0, el=25.0, zoom="action", size=(560, 440), label=None, center=None, dist=None):
    """Object + hand + table from azimuth `az` (degrees about gravity: 0 = from the real camera's right, 90 = from
    behind the object looking back at the camera, 180 = from the left, 270 = from the camera's side) and elevation
    `el` (0 = level, 90 = straight down); zoom "action" frames hand and object, "contact" the region where they meet."""
    clip.use(state)
    up, right, fwd = clip.up, clip.right, clip.fwd
    hv = clip.hand_verts(state, i)
    T = state.obj[i]
    c_obj = clip.obj_center(state, i)
    sd = clip.sdf_at((hv - T[:3, 3]) @ T[:3, :3])
    if center is not None:                             # fixed camera (filmstrips): same view for every frame
        c, dist = np.asarray(center, float), dist or max(0.40, 2.6 * float(clip.extent.max()))
    elif zoom == "contact" and clip.hand_valid[i]:
        near = hv[sd < max(0.01, np.sort(sd)[min(20, len(sd) - 1)])]
        c, dist = near.mean(0), 0.16
    else:
        c = 0.5 * (c_obj + hv.mean(0)) if clip.hand_valid[i] else c_obj
        dist = max(0.40, 2.6 * float(clip.extent.max()))
    a, e = np.radians(az), np.radians(el)
    eye = np.cos(e) * (np.cos(a) * right + np.sin(a) * fwd) + np.sin(e) * up
    z = -eye
    y = -up - (-up @ z) * z
    if np.linalg.norm(y) < 1e-6:                       # straight down: image up = away from the camera
        y = -fwd - (-fwd @ z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    Rv = np.stack([x, y, z])
    t = -Rv @ (c + dist * eye)
    Tv = np.eye(4)
    Tv[:3, :3], Tv[:3, 3] = Rv, t
    W, H = size
    f = 0.9 * W
    Kv = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    img = np.full((H, W, 3), 245, np.uint8)
    n, p0 = clip.table_n, clip.table_p
    cp = c - ((c - p0) @ n) * n
    e1 = right - (right @ n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    proj = lambda P: (Kv @ (Rv @ np.asarray(P).T + t[:, None])).T
    for s_ in np.arange(-0.4, 0.4001, 0.05):            # table: 5 cm grid on the fitted plane
        for a0, b0 in (((s_, -0.4), (s_, 0.4)), ((-0.4, s_), (0.4, s_))):
            P = proj([cp + a0[0] * e1 + a0[1] * e2, cp + b0[0] * e1 + b0[1] * e2])
            if (P[:, 2] > 0.01).all():
                cv2.line(img, tuple(np.int32(P[0, :2] / P[0, 2])), tuple(np.int32(P[1, :2] / P[1, 2])), (190, 185, 175), 1)
    oc, od = _render(clip, clip.mt, Tv @ T, Kv, H, W)
    hc, hd = _render(clip, clip.hand_tensors(hv, clip.hand_colors(state, i, hv)), Tv, Kv, H, W)
    hd[~(hd > 0)] = np.inf
    od[~(od > 0)] = np.inf
    so, sh = np.isfinite(od) & (od <= hd), np.isfinite(hd) & (hd < od)
    img[so] = oc[so]
    img[sh] = hc[sh]
    name = AZ_NAMES.get(int(round(az)) % 360, f"az {az:.0f}")
    txt = label if label is not None else f"frame {i} {name}" + (f", el {el:.0f}" if el != 25 else "") + (" (contact)" if zoom == "contact" else "")
    cv2.putText(img, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
    return img


def _orbit_cam(clip, c, dist, az, el, W, H):
    up, right, fwd = clip.up, clip.right, clip.fwd
    a, e = np.radians(az), np.radians(el)
    eye = np.cos(e) * (np.cos(a) * right + np.sin(a) * fwd) + np.sin(e) * up
    z = -eye
    y = -up - (-up @ z) * z
    if np.linalg.norm(y) < 1e-6:
        y = -fwd - (-fwd @ z) * z
    y /= np.linalg.norm(y)
    Rv = np.stack([np.cross(y, z), y, z])
    t = -Rv @ (c + dist * eye)
    Kv = np.array([[0.9 * W, 0, W / 2], [0, 0.9 * W, H / 2], [0, 0, 1.0]])
    return Rv, t, Kv


TRACE_COLORS = [(40, 40, 40), (200, 60, 200), (40, 40, 230), (40, 160, 40), (230, 120, 30), (30, 180, 230)]  # BGR


def motion_center(clip, state, frames):
    clip.use(state)
    pts = [0.5 * (clip.obj_center(state, i) + clip.hand_verts(state, i).mean(0)) for i in frames]
    return np.mean(pts, 0)


def trace(clip, state, start, end, az=0.0, el=25.0, size=(720, 560), label=None):
    """One fixed 3D view of [start, end]: the scene at the middle frame and the paths of the wrist (dark) and the five
    fingertips (thumb magenta, index red, middle green, ring orange, pinky cyan) over the range, one dot per frame;
    real motion draws smooth curves, jitter draws zig-zags."""
    clip.use(state)
    from clip import TIPS, FINGERS
    fr = list(range(start, end + 1))
    mid = fr[len(fr) // 2]
    hvs = {i: clip.hand_verts(state, i) for i in fr}
    pts = np.concatenate([hvs[i][[TIPS[f] for f in FINGERS]] for i in fr])
    c = motion_center(clip, state, fr)
    dist = max(0.40, 2.6 * float(clip.extent.max()), 2.2 * float(np.ptp(pts, 0).max()))
    img = view_orbit(clip, state, mid, az, el, size=size, center=c, dist=dist, label="")
    img = (0.55 * img + 0.45 * 255).astype(np.uint8)              # fade the scene so the paths stand out
    W, H = size
    Rv, t, Kv = _orbit_cam(clip, c, dist, az, el, W, H)
    proj = lambda P: (lambda q: q[:, :2] / q[:, 2:3])((Kv @ (Rv @ np.asarray(P).T + t[:, None])).T)
    idx = [None] + [TIPS[f] for f in FINGERS]
    for k, vi in enumerate(idx):
        P = np.array([(clip.J16 @ hvs[i].astype(np.float64))[0] if vi is None else hvs[i][vi] for i in fr])
        q = np.int32(proj(P))
        ok = [clip.hand_valid[i] for i in fr]
        for a in range(len(q) - 1):
            if ok[a] and ok[a + 1]:
                cv2.line(img, tuple(q[a]), tuple(q[a + 1]), TRACE_COLORS[k], 2)
        for a in range(len(q)):
            if ok[a]:
                cv2.circle(img, tuple(q[a]), 2, TRACE_COLORS[k], -1)
    txt = label if label is not None else f"paths of wrist + fingertips, frames {start}-{end}, {AZ_NAMES.get(int(round(az)) % 360, f'az {az:.0f}')}"
    cv2.putText(img, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
    return img


def filmstrip(clip, state, start, count=8, az=0.0, el=25.0, zoom="action", tile=(360, 300)):
    """`count` consecutive frames from one fixed viewpoint (and, with zoom="contact", close to where hand and object
    meet), so frame-to-frame jitter is visible."""
    clip.use(state)
    fr = [i for i in range(start, min(clip.N, start + count))]
    c = motion_center(clip, state, fr)
    dist = None
    if zoom == "contact":
        hv = clip.hand_verts(state, fr[len(fr) // 2])
        T = state.obj[fr[len(fr) // 2]]
        sd = clip.sdf_at((hv - T[:3, 3]) @ T[:3, :3])
        c, dist = hv[sd < max(0.01, np.sort(sd)[min(20, len(sd) - 1)])].mean(0), 0.18
    return sheet([view_orbit(clip, state, i, az, el, size=tile, center=c, dist=dist, label=f"frame {i}") for i in fr],
                 ncols=min(4, len(fr)), tile_w=tile[0])


def sheet(imgs, ncols=3, tile_w=640):
    tiles = []
    for im in imgs:
        h, w = im.shape[:2]
        tiles.append(cv2.resize(im, (tile_w, int(h * tile_w / w))))
    th = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, th - t.shape[0]), (0, 0), (0, 0)), constant_values=255) for t in tiles]
    while len(tiles) % ncols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[k:k + ncols]) for k in range(0, len(tiles), ncols)]
    return np.vstack(rows)


def crop(clip, state, i, box, mode="outline"):
    clip.use(state)
    img = overlay(clip, state, i, mode=mode, box=None, label=f"frame {i} crop {box}")
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(clip.W, x1), min(clip.H, y1)
    c = img[y0:y1, x0:x1]
    s = 1100 / max(c.shape[:2])
    return cv2.resize(c, (int(c.shape[1] * s), int(c.shape[0] * s)), interpolation=cv2.INTER_CUBIC)


MIN_SPAN = {"contact_cm": 2.0, "penetration_cm": 1.0, "obj_height_cm": 2.0, "hand_height_cm": 2.0,
            "speed_cm_per_frame": 2.0, "ang_speed_deg_per_frame": 5.0, "hand_kp_px": 20.0, "depth_res_cm": 1.0,
            "n_cross": 50.0, "hand_accel_mm": 10.0, "hand_rel_accel_mm": 10.0, "finger_accel_deg": 10.0, "tip_accel_mm": 15.0,
            "obj_accel_cm": 1.0, "obj_ang_accel_deg": 5.0}


def plot(clip, rows_by_name, keys, start=0, end=None, phases=()):
    """Line plot of per-frame metrics; rows_by_name: {"current": rows, "initial": rows, ...}."""
    end = clip.N - 1 if end is None else end
    fig, axs = plt.subplots(len(keys), 1, figsize=(10, 1.9 * len(keys)), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, k in zip(axs, keys):
        for j, (name, rows) in enumerate(rows_by_name.items()):     # dashed / dotted so coinciding lines both show
            fr = [r["frame"] for r in rows if start <= r["frame"] <= end]
            y = [r.get(k) if r.get(k) is not None else np.nan for r in rows if start <= r["frame"] <= end]
            ax.plot(fr, y, label=name, lw=1.6, ls=["-", "--", ":"][j % 3])
        for p in phases:
            ax.axvspan(p["start"], p["end"], alpha=0.08, color="C%d" % (hash(p["label"]) % 9))
        if k in MIN_SPAN:                   # keep millimetre noise looking like noise (matplotlib autoscales to it)
            lo, hi = ax.get_ylim()
            if hi - lo < MIN_SPAN[k]:
                lo = min(lo, 0.0) if k != "obj_height_cm" else lo
                ax.set_ylim(lo, lo + MIN_SPAN[k])
        ax.set_ylabel(k, fontsize=8)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8, loc="upper right")
    for p in phases:
        axs[0].text((p["start"] + p["end"]) / 2, axs[0].get_ylim()[1], p["label"], ha="center", va="bottom", fontsize=8)
    axs[-1].set_xlabel("frame")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    return cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
