"""Depth cameras of the §1.4 policy: the definitions in configs/embodiment/cameras.yaml as objects, a reference depth
renderer and the sensor model (range, stereo validity, noise).

The renderer casts rays in MuJoCo (mj_multiRay): no OpenGL and no GPU, exact geometry, surface normals for free. It
is the reference for what the definitions mean; a faster renderer used for training has to reproduce its output.

A stereo depth camera measures a point only if both imagers see it. So next to the depth along the optical axis of
the left imager (the depth origin) the renderer reports, per pixel, whether the right imager, baseline_m to the
right, sees the same point: unobstructed and inside its image. That gives the occlusion shadows beside near objects
and the invalid band on the image's left that real D400 cameras have.

    cams = Cameras()                                   # cameras.yaml
    img = cams.wrist.render(model, data, T_world_cam)  # {'depth', 'valid', 'geom', 'normal_z', ...} at policy size
    obs = cams.wrist.observe(img, rng)                 # noisy depth + validity mask, as the policy gets them
"""

from __future__ import annotations

import numpy as np

import common as C

GROUPS_VISIBLE = np.array([0, 1, 1, 0, 0, 0], np.uint8)        # geom groups a camera sees: 1 robot visuals, 2 scene


class Camera:
    def __init__(self, name, cfg, sensor):
        self.name, self.cfg, self.sensor = name, cfg, sensor
        self.parent = cfg["parent"]
        self.T_parent_cam = C.cfg_T(cfg["pose"])
        s, k = sensor["stream"], sensor["intrinsics"]
        self.W, self.H = int(s["width"]), int(s["height"])
        self.fx, self.fy, self.cx, self.cy = (float(k[x]) for x in ("fx", "fy", "cx", "cy"))
        self.baseline = float(sensor["baseline_m"])
        self.z_min, self.z_max = (float(x) for x in cfg["depth_range_m"])
        self.policy = dict(cfg.get("policy_image", {"width": self.W // 8, "height": self.H // 8}))
        # the policy image shows this window of the stream (x0, y0, width, height in stream pixels; default: all of it)
        self.roi = tuple(self.policy.get("roi", (0, 0, self.W, self.H)))
        self.noise = dict(sensor.get("noise", {}))
        self.noise.update(cfg.get("noise", {}))

    # ---- geometry
    def fov_deg(self):
        return (float(np.degrees(2 * np.arctan(self.W / 2 / self.fx))), float(np.degrees(2 * np.arctan(self.H / 2 / self.fy))))

    def fov_policy_deg(self):
        fx, fy, cx, cy = self.intrinsics()
        w, h = self.policy["width"], self.policy["height"]
        return (float(np.degrees(np.arctan(cx / fx) + np.arctan((w - cx) / fx))),
                float(np.degrees(np.arctan(cy / fy) + np.arctan((h - cy) / fy))))

    def intrinsics(self, width=None, height=None):
        """(fx, fy, cx, cy) of the policy image's window of the stream, sampled at width x height pixels (pixel
        centres at integer + 0.5)."""
        width, height = width or self.policy["width"], height or self.policy["height"]
        x0, y0, rw, rh = self.roi
        sx, sy = width / rw, height / rh
        return self.fx * sx, self.fy * sy, (self.cx - x0) * sx, (self.cy - y0) * sy

    def rays(self, width, height):
        fx, fy, cx, cy = self.intrinsics(width, height)
        u = (np.arange(width) + 0.5 - cx) / fx
        v = (np.arange(height) + 0.5 - cy) / fy
        d = np.stack(np.broadcast_arrays(u[None, :], v[:, None], 1.0), -1).reshape(-1, 3)
        return d                                                # z = 1: range along the ray * |d| ... see render()

    def project(self, P_cam):
        """Points in the optical frame -> (u, v) in stream pixels and z."""
        z = P_cam[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * P_cam[..., 0] / z + self.cx
            v = self.fy * P_cam[..., 1] / z + self.cy
        return u, v, z

    def sees(self, P_cam, margin_px=0.0):
        """True where a point is in front of the camera, inside the image of BOTH imagers and inside the depth
        range (occlusion not tested)."""
        u, v, z = self.project(P_cam)
        ur = u - self.fx * self.baseline / np.where(z > 1e-6, z, np.inf)
        x0, y0, rw, rh = self.roi
        ok = (z >= self.z_min) & (z <= self.z_max) & (v >= y0 + margin_px) & (v <= y0 + rh - margin_px)
        return ok & (u >= x0 + margin_px) & (u <= x0 + rw - margin_px) & (ur >= margin_px)

    def measures(self, model, data, T_world_cam, P, z_min=None, tol=0.003):
        """Which of the world points P give a valid stereo depth: inside the policy image's window for the left
        imager and inside the right imager's image, inside the depth range, and hidden from neither imager.
        `tol` (scalar or per point) is how far in front of a point a surface may be hit and still count as the
        point itself: 3 mm for points on a surface, about a fingertip's radius for a fingertip's centre."""
        tol = np.broadcast_to(np.asarray(tol, float), (len(P),))
        import mujoco
        R, o = T_world_cam[:3, :3], T_world_cam[:3, 3]
        keep = self.z_min
        if z_min is not None:
            self.z_min = z_min
        ok = self.sees((P - o) @ R)
        self.z_min = keep
        idx = np.flatnonzero(ok)
        for org in (o, o + R[:, 0] * self.baseline):              # left, then right imager
            if not len(idx):
                break
            v = P[idx] - org
            L = np.linalg.norm(v, axis=1)
            v = np.ascontiguousarray(v / L[:, None])
            n = len(idx)
            geom, dist = np.zeros(n, np.int32), np.zeros(n)
            mujoco.mj_multiRay(model, data, org.copy(), v.reshape(-1), GROUPS_VISIBLE, 1, -1, geom, dist, None, n, self.z_max * 2 + 1)
            free = (dist < 0) | (dist >= L - tol[idx])
            ok[idx[~free]] = False
            idx = idx[free]
        return ok

    # ---- rendering
    def render(self, model, data, T_world_cam, width=None, height=None, supersample=2, bodyexclude=-1):
        """Ray-cast depth at width x height (default: the policy image), `supersample`^2 rays per pixel.

        Returns depth [m] along the optical axis (0 where nothing is hit), `stereo` (the right imager sees the
        point), `geom` (id of the geom hit at the pixel's centre ray, -1 none), `cos` (cosine between the surface
        normal and the ray, 1 = facing the camera) and `frac` (share of the pixel's rays that hit something)."""
        import mujoco
        width, height = width or self.policy["width"], height or self.policy["height"]
        s = int(supersample)
        w, h = width * s, height * s
        d_cam = self.rays(w, h)
        norm = np.linalg.norm(d_cam, axis=1)
        d_unit = d_cam / norm[:, None]
        R, o = T_world_cam[:3, :3], T_world_cam[:3, 3]
        vec = np.ascontiguousarray(d_unit @ R.T)
        n = len(vec)
        geom = np.full(n, -1, np.int32)
        dist = np.full(n, -1.0)
        normal = np.zeros((n, 3))
        cutoff = self.z_max * 1.6 + 0.05
        mujoco.mj_multiRay(model, data, o.copy(), vec.reshape(-1), GROUPS_VISIBLE, 1, bodyexclude, geom, dist,
                           normal.reshape(-1), n, cutoff)
        hit = dist >= 0
        z = np.where(hit, dist / norm, 0.0)                     # range -> depth along the optical axis
        cosang = np.where(hit, np.abs(np.einsum("ij,ij->i", normal, vec)), 0.0)
        # stereo: is the hit point visible from the right imager?
        stereo = np.zeros(n, bool)
        if hit.any():
            P = o + vec[hit] * dist[hit, None]
            o_r = o + R[:, 0] * self.baseline
            v_r = P - o_r
            len_r = np.linalg.norm(v_r, axis=1)
            v_r = np.ascontiguousarray(v_r / len_r[:, None])
            g2 = np.full(len(P), -1, np.int32)
            d2 = np.full(len(P), -1.0)
            mujoco.mj_multiRay(model, data, o_r.copy(), v_r.reshape(-1), GROUPS_VISIBLE, 1, bodyexclude, g2, d2, None,
                               len(P), cutoff * 1.2)
            tol = 0.002 + 0.01 * len_r                         # the same surface point, up to grazing-angle slack
            vis = (d2 < 0) | (d2 >= len_r - tol)
            u = self.roi[0] + (np.tile(np.arange(w), h) + 0.5)[hit] * self.roi[2] / w      # column in the stream
            in_right = u - self.fx * self.baseline / z[hit] >= 0                           # inside the right imager's image
            stereo[hit] = vis & in_right
        out = {"depth_ss": z.reshape(h, w), "stereo_ss": stereo.reshape(h, w), "hit_ss": hit.reshape(h, w)}
        c = s // 2
        out["geom"] = geom.reshape(h, w)[c::s, c::s]
        out["cos"] = cosang.reshape(h, w)[c::s, c::s]
        zz = out["depth_ss"].reshape(height, s, width, s).transpose(0, 2, 1, 3).reshape(height, width, s * s)
        ok = (out["hit_ss"] & out["stereo_ss"]).reshape(height, s, width, s).transpose(0, 2, 1, 3).reshape(height, width, s * s)
        hh = out["hit_ss"].reshape(height, s, width, s).transpose(0, 2, 1, 3).reshape(height, width, s * s)
        with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            out["depth"] = np.nan_to_num(np.nanmedian(np.where(hh, zz, np.nan), axis=2))          # geometric depth
            out["depth_stereo"] = np.nan_to_num(np.nanmedian(np.where(ok, zz, np.nan), axis=2))    # of the measurable rays
        out["frac"] = hh.mean(2)
        out["stereo"] = ok.mean(2) >= 0.5
        return out

    # ---- sensor model
    def observe(self, img, rng=None, episode=None):
        """What the policy gets from a rendered image: (depth [m], valid mask). With rng: the noise model of
        cameras.yaml; without: the ideal sensor (range and stereo validity only).

        `episode` holds what stays fixed over an episode (this camera's depth scale error): make one with
        new_episode(rng) and pass it to every frame of the episode."""
        nz = self.noise
        z = img["depth_stereo"].copy()
        valid = img["stereo"] & (z >= self.z_min) & (z <= self.z_max)
        if rng is not None and nz:
            fb = self.fx * self.baseline                        # disparity [stream px] = fb / z
            H, W = z.shape
            with np.errstate(divide="ignore"):
                disp = np.where(z > 0, fb / np.maximum(z, 1e-6), 0.0)
            sig = nz.get("disparity_sigma_px", 0.0)
            if sig:
                disp = disp + rng.normal(0, sig, disp.shape)
            lf = nz.get("disparity_lowfreq_px", 0.0)
            if lf:                                              # smooth field: bilinear upsample of a coarse grid
                g = rng.normal(0, lf, (4, 5))
                yy = np.linspace(0, 3, H)[:, None]
                xx = np.linspace(0, 4, W)[None, :]
                y0, x0 = np.clip(yy.astype(int), 0, 2), np.clip(xx.astype(int), 0, 3)
                ty, tx = yy - y0, xx - x0
                disp = disp + ((1 - ty) * (1 - tx) * g[y0, x0] + (1 - ty) * tx * g[y0, x0 + 1]
                               + ty * (1 - tx) * g[y0 + 1, x0] + ty * tx * g[y0 + 1, x0 + 1])
            q = nz.get("disparity_quant_px", 0.0)
            if q:
                disp = np.round(disp / q) * q
            scale = (episode or {}).get("scale", 1.0)
            with np.errstate(divide="ignore"):
                z = np.where(disp > 1e-6, fb / np.maximum(disp, 1e-6), 0.0) * scale
            # dropouts
            p = nz.get("dropout_pixel", 0.0)
            if p:
                valid &= rng.random(z.shape) >= p
            ge = nz.get("grazing_deg", 0.0)
            if ge:                                              # surfaces seen at more than grazing_deg from the normal
                gp = nz.get("grazing_dropout", 1.0)
                valid &= ~((img["cos"] < np.cos(np.radians(ge))) & (rng.random(z.shape) < gp))
            ed = nz.get("edge_dropout", 0.0)
            if ed:                                              # pixels straddling a depth edge (mixed rays)
                edge = (img["frac"] < 0.999) | (_discontinuity(img["depth"]) > nz.get("edge_jump_m", 0.02))
                valid &= ~(edge & (rng.random(z.shape) < ed))
            nh = nz.get("holes_per_frame", 0.0)
            if nh:
                yy, xx = np.mgrid[0:H, 0:W]
                for _ in range(rng.poisson(nh)):
                    cy, cx = rng.uniform(0, H), rng.uniform(0, W)
                    r = rng.uniform(*nz.get("hole_radius_px", [1.0, 4.0]))
                    valid &= (yy - cy) ** 2 + (xx - cx) ** 2 > r * r
            valid &= (z >= self.z_min) & (z <= self.z_max)
        u = self.sensor.get("depth_unit_m", 0.0)
        if u:
            z = np.round(z / u) * u
        return np.where(valid, z, 0.0).astype(np.float32), valid

    def new_episode(self, rng):
        s = self.noise.get("scale_sigma", 0.0)
        return {"scale": float(1.0 + np.clip(rng.normal(0, s), -3 * s, 3 * s)) if s else 1.0}

    def encode(self, depth, valid):
        """The policy's input: 2 channels, depth mapped to [0, 1] over the camera's range (0 where invalid) and
        the validity mask."""
        d = np.clip((depth - self.z_min) / (self.z_max - self.z_min), 0, 1) * valid
        return np.stack([d, valid.astype(np.float32)]).astype(np.float32)


def _discontinuity(z):
    """How far each pixel's depth is from the straight line through its two neighbours, the largest over the four
    directions through the pixel [m]. Zero on a plane however steeply it is seen, large at a depth edge; a neighbour
    without a hit counts as an edge."""
    zz = np.where(z > 0, z, np.nan)
    p = np.pad(zz, 1, mode="edge")
    H, W = z.shape
    c = p[1:-1, 1:-1]
    out = np.zeros_like(z, dtype=float)
    for (a0, a1), (b0, b1) in (((0, 1), (2, 1)), ((1, 0), (1, 2)), ((0, 0), (2, 2)), ((0, 2), (2, 0))):
        a, b = p[a0:a0 + H, a1:a1 + W], p[b0:b0 + H, b1:b1 + W]
        d = np.abs(a + b - 2 * c)
        out = np.maximum(out, np.where(np.isnan(d), np.inf, d))
    return np.where(z > 0, out, 0.0)


class Cameras:
    def __init__(self, cfg=None):
        cfg = cfg or C.load_cameras()
        self.cfg = cfg
        self.wrist = Camera("wrist_cam", cfg["wrist"], cfg["sensors"][cfg["wrist"]["sensor"]])
        self.exo = Camera("exo_cam", cfg["exo"], cfg["sensors"][cfg["exo"]["sensor"]])


def colorize(depth, valid, z_min, z_max):
    """Depth image -> RGB uint8 (near = bright), invalid pixels dark red."""
    t = np.clip((depth - z_min) / (z_max - z_min), 0, 1)
    v = (1 - t)
    rgb = np.stack([0.15 + 0.85 * v, 0.2 + 0.75 * v ** 1.3, 0.35 + 0.6 * v ** 2], -1)
    rgb[~valid] = (0.35, 0.05, 0.05)
    return (rgb * 255).astype(np.uint8)
