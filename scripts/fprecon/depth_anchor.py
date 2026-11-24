"""Table-anchored depth: one table plane for the whole clip (numpy + cv2 only; used by the refinement agent and the viewer).

Depth Pro 2's metric scale is not stable over a clip. With a static camera (background points move < 4 px over our
clips) the table is flat in every frame's depth map, to ~0.5 cm, but its distance from the camera changes from frame
to frame by up to +-12% (2-6 cm in height), most while the hand is in the scene; the frames' planes stay within
1.5 deg of each other. The depth on the object changes with the table's: while the object rests, the depth map's
value on it follows the table plane's with slope 1 (it is static, yet "moves" up to 9 cm along the view). The table
does not move, so it calibrates every frame.

On a plane, inverse depth is affine in the normalised image coordinates: 1/z = co . (1, x, y). Per frame we fit the
table's co; the frame's depth along a ray r = (1, x, y) is then multiplied by (co_i . r) / (co_ref . r), the ratio of
the reference plane's depth to the frame's own plane's depth on that ray: one scale per frame, plus the small tilt
between the frames' planes. After that the table is the same plane, co_ref, in every frame.
"""

import json

import cv2
import numpy as np

VERSION = 1


def rays(K, h, w):
    """(h, w, 3): (1, x, y) per pixel, with x and y the normalised image coordinates."""
    ys, xs = np.mgrid[0:h, 0:w]
    return np.stack([np.ones((h, w)), (xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1]], -1)


def plane_co(normal, point):
    """A plane (normal, a point on it) as the coefficients of its inverse depth, 1/z = co . (1, x, y)."""
    n = np.asarray(normal, float)
    return np.array([n[2], n[0], n[1]]) / float(n @ np.asarray(point, float))


def co_plane(co):
    """Inverse-depth coefficients -> (unit normal pointing away from the camera, distance of the plane from it)."""
    c = 1.0 / np.linalg.norm(co)
    return np.array([co[1], co[2], co[0]]) * c, c


def factor(co_i, ref, p):
    """Depth factor of a frame along the ray(s) through the camera-frame point(s) p (..., 3)."""
    p = np.asarray(p, float)
    r = np.stack([np.ones(p.shape[:-1]), p[..., 0] / p[..., 2], p[..., 1] / p[..., 2]], -1)
    return (r @ co_i) / (r @ ref)


def fit_planes(depth, fg, K, start, bands=(0.05, 0.025, 0.012, 0.012), min_points=3000, max_deg=6.0, max_ratio=1.25):
    """The table's plane in every frame of `depth` (N, h, w; 0 = invalid), as inverse-depth coefficients (N, 3):
    least squares over the frame's points outside `fg` (object and hand masks, dilated here) that lie within a band
    of the previous frame's plane, the band narrowing from 5 to 1.2 cm around each new fit. `start`: frame 0's plane
    (plane_co). A frame with fewer than `min_points` table points, or whose plane is more than `max_deg` or a factor
    `max_ratio` from the start plane, takes its neighbours' coefficients (interpolated in time).
    Returns (co, rms of the table points off their frame's plane in m, per frame; NaN where interpolated), or
    (None, None) when no frame has a usable table."""
    N, h, w = depth.shape
    A = rays(K, h, w)
    kern = np.ones((7, 7), np.uint8)
    n_s, c_s = co_plane(start)
    co, rms, cur = np.full((N, 3), np.nan), np.full(N, np.nan), np.asarray(start, float)
    for i in range(N):
        d = depth[i]
        ok = (d > 0.05) & (d < 8) & ~(cv2.dilate(fg[i].astype(np.uint8), kern) > 0)
        wv = 1.0 / np.maximum(d, 1e-3)
        c, good = cur, True
        for band in bands:
            hgt = d * (wv - A @ c) / np.linalg.norm(c)             # height off the current plane
            sel = ok & (np.abs(hgt) < band)
            if sel.sum() < min_points:
                good = False
                break
            c = np.linalg.lstsq(A[sel], wv[sel], rcond=None)[0]
        if good:
            n, off = co_plane(c)
            good = (np.degrees(np.arccos(np.clip(n @ n_s, -1, 1))) < max_deg) and (1 / max_ratio < off / c_s < max_ratio)
        if good:
            co[i], cur = c, c
            rms[i] = float(np.sqrt((hgt[sel] ** 2).mean()))
    ok = np.isfinite(co[:, 0])
    if not ok.any():
        return None, None
    idx = np.nonzero(ok)[0]
    for k in range(3):
        co[:, k] = np.interp(np.arange(N), idx, co[idx, k])
    return co, rms


def save(path, co, ref, mesh_scale, info):
    json.dump({"version": VERSION, "co": np.asarray(co).tolist(), "ref": np.asarray(ref).tolist(),
               "mesh_scale": float(mesh_scale), "info": info}, open(path, "w"), indent=1)


def load(path):
    a = json.load(open(path))
    a["co"], a["ref"] = np.array(a["co"], float), np.array(a["ref"], float)
    return a


def apply_frame(depth_i, K, co_i, ref):
    """One full-resolution depth map scaled onto the reference table plane."""
    h, w = depth_i.shape
    A = rays(K, h, w)
    return depth_i * ((A @ co_i) / (A @ ref)).astype(depth_i.dtype)
