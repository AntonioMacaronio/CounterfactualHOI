"""The D405 wrist mount as CAD: one plate between the arm's tool flange and the Sharpa Wave's flange, with a tab on
the palm side that carries the camera.

  python scripts/embodiment/d405_mount.py --arm marvin_m6ccs      (venv: $PROJECT_STORAGE_ROOT/venvs/arm, CadQuery)
  -> $PROJECT_STORAGE_ROOT/build/embodiment/mount/<arm>/d405_mount.step   millimetres, for machining / printing
                                                         d405_mount.stl    metres, in the hand_mount frame (build_model.py)
                                                         d405_mount.json   dimensions, hardware list, clearance checks
                                                         d405_mount_{top,side,iso}.png

Everything is in the Sharpa flange's frame F: origin at the centre of the face that meets the plate, +z towards the
hand, +x out of the palm, +y towards the thumb (the palm frame is F + 29.5 mm in z). The plate fills z = -t .. 0.

  * Arm side (z = -t): the arm's bolt circle, from the arm's config.
      Marvin: ISO 9409-1-50-4-M6. Four 6.6 mm holes at 45 deg: the M6 screws pass through the Sharpa flange and the
      plate into the arm (M6 x 18). A 6 mm dowel through the plate keys both flanges; a 31.4 mm boss centres it.
      Vega: four M4 counterbored holes on the 49 mm circle (four of the wrist's six holes) hold the plate, and four
      M6 threads take the Sharpa flange (M6 x 10). No dowel: it would break into an M4 counterbore.
  * Hand side (z = 0): a 31.6 mm recess for the Sharpa flange's centring boss.
  * Camera tab: the D405 is screwed to the plate's hand-side face through its two rear M3 threads (20 mm apart,
    4 mm deep), from the arm side, with the screw heads sunk. The camera's pose comes from cameras.yaml; with the
    default pose (looking straight along the fingers, its back flush with the plate) the whole mount is flat and
    can be cut from sheet. A tilted camera gets a wedge-shaped tab instead.
"""

from __future__ import annotations

import argparse
import json

import cadquery as cq
import numpy as np

import common as C

SHARPA_FLANGE_TO_PALM = 29.5          # mm
SHARPA = {"boss_d": 31.5, "boss_h": 2.0, "pcd": 50.0, "hole_d": 6.6, "pin_d": 6.1, "flange_r": 31.5, "flange_h": 8.0,
          "wrist_r": 24.0}
D405 = {"size": (42.0, 42.0, 23.0), "centre": (9.0, 0.0, -7.8), "back_z": -19.3, "m3_spacing": 20.0, "m3_insert": 4.0,
        "usb_from_back": 6.82}
AL_DENSITY = 2.70e-3                  # g / mm^3 (6061)


def cyl(d, p0, p1):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    v = p1 - p0
    return cq.Solid.makeCylinder(d / 2, float(np.linalg.norm(v)), cq.Vector(*p0), cq.Vector(*(v / np.linalg.norm(v))))


def oriented_box(size, T):
    """Box of full sizes `size` centred at T's origin with T's axes."""
    b = cq.Workplane("XY").box(*size).val()
    rv = C.rot_log(T[:3, :3])
    ang = float(np.linalg.norm(rv))
    axis = rv / ang if ang > 1e-9 else np.array([0, 0, 1.0])
    return b.moved(cq.Location(cq.Vector(*map(float, T[:3, 3])), cq.Vector(*map(float, axis)), float(np.degrees(ang))))


def hull2d(pts):
    pts = sorted(set(map(tuple, np.round(pts, 6))))
    def half(points):
        h = []
        for p in points:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= 0:
                h.pop()
            h.append(p)
        return h
    lo, up = half(pts), half(pts[::-1])
    return lo[:-1] + up[:-1]


def build(arm):
    cfg = C.load_arm(arm)
    cams = C.load_cameras()
    t = float(cfg["hand_mount"]["plate_mm"])
    yaw = float(cfg["hand_mount"]["yaw_deg"])
    iface = cfg["flange"]["interface"]
    T_F_cam = C.make_T(None, [0, 0, SHARPA_FLANGE_TO_PALM / 1000]) @ C.cfg_T(cams["wrist"]["pose"])
    R, o = T_F_cam[:3, :3], T_F_cam[:3, 3] * 1000                         # mm from here on
    body_r = iface["centering"]["body_diameter_mm"] / 2
    plate_r = max(body_r, SHARPA["flange_r"] + 0.5) if "ISO" in iface["standard"] else max(body_r + 2.5, SHARPA["flange_r"] + 0.5)
    notes, hardware, checks = [], [], {}

    # ---- the disc
    part = cq.Workplane("XY").circle(plate_r).extrude(t).translate((0, 0, -t))

    # ---- the camera tab and its bridge to the disc
    cam_centre = o + R @ np.array(D405["centre"])
    back = o + R @ np.array([D405["centre"][0], 0, D405["back_z"]])           # centre of the camera's back face
    view, down, right = R[:, 2], R[:, 1], R[:, 0]
    flat = abs(view[2] - 1) < 1e-6 and abs(back[2]) < 0.05                     # camera square to the plate, back flush
    usb_ear = 9.0                                                 # extra width on the camera's USB side for the cable ties
    tab_w, tab_h = D405["size"][0] + 2.0 + usb_ear, D405["size"][1] + 2.0
    tab_t = t if flat else 6.0
    T_tab = C.make_T(R, back - view * tab_t / 2 - right * (usb_ear - 2.0) / 2)
    tab = oriented_box((tab_w, tab_h, tab_t), T_tab)
    part = part.union(cq.Workplane(obj=tab))
    # bridge: the convex hull, in the side view (x, z), of the tab's section and a strip of the disc, extruded in y
    corners = [T_tab[:3, 3] + down * a * tab_h / 2 + view * b * tab_t / 2 for a in (-1, 1) for b in (-1, 1)]
    strip = [(plate_r * 0.55, -t), (plate_r * 0.55, 0.0), (plate_r * 0.95, -t), (plate_r * 0.95, 0.0)]
    prof = hull2d([(c[0], c[2]) for c in corners] + strip)
    bridge_w = min(tab_w, 2 * np.sqrt(max(plate_r ** 2 - (plate_r * 0.55) ** 2, 1.0)))
    bridge = cq.Workplane("XZ").polyline(prof).close().extrude(bridge_w / 2, both=True).translate((0, float(T_tab[1, 3]), 0))
    part = part.union(bridge)
    # keep clear of the Sharpa flange and wrist above the plate, and of the camera body itself
    keep_out = cq.Workplane("XY").circle(SHARPA["flange_r"] + 1.0).extrude(SHARPA["flange_h"] + 6).union(
        cq.Workplane("XY").circle(SHARPA["wrist_r"] + 1.5).extrude(60))
    part = part.cut(keep_out)
    T_cam_body = C.make_T(R, cam_centre)
    part = part.cut(cq.Workplane(obj=oriented_box(tuple(s + 0.4 for s in D405["size"]), T_cam_body)))

    # ---- hand side: recess for the Sharpa boss, the Sharpa bolt circle
    part = part.cut(cq.Workplane("XY").circle(SHARPA["boss_d"] / 2 + 0.05).extrude(SHARPA["boss_h"] + 0.5).translate((0, 0, -(SHARPA["boss_h"] + 0.5))))
    sharpa_angles = [45, 135, 225, 315]
    r_pcd = SHARPA["pcd"] / 2
    arm_bolts = iface["bolts"]
    arm_angles = [(arm_bolts["first_deg"] + k * 360.0 / arm_bolts["count"] - yaw) % 360 for k in range(arm_bolts["count"])]
    if "ISO" in iface["standard"]:
        # the two bolt circles coincide: clearance holes, the hand's screws go through into the arm
        mis = min(min(abs((a - b + 180) % 360 - 180) for b in sharpa_angles) for a in arm_angles)
        assert mis < 0.5, f"hand yaw {yaw} does not line the Sharpa holes up with the arm's ({mis:.1f} deg off)"
        for a in sharpa_angles:
            x, y = r_pcd * np.cos(np.radians(a)), r_pcd * np.sin(np.radians(a))
            part = part.cut(cq.Workplane(obj=cyl(SHARPA["hole_d"], (x, y, -t - 5), (x, y, 5))))
        length = 4 + t + 8
        hardware.append(f"4x M6 x {int(round(length))} socket head cap screw (through Sharpa flange 4 mm + plate {t:g} mm, "
                        f"{length - 4 - t:.0f} mm into the arm's M6 x {arm_bolts['depth_mm']} threads)")
        pin = iface.get("pin")
        if pin:
            a = pin["at_deg"] - yaw
            x, y = r_pcd * np.cos(np.radians(a)), r_pcd * np.sin(np.radians(a))
            part = part.cut(cq.Workplane(obj=cyl(pin["diameter_mm"], (x, y, -t - 5), (x, y, 5))))
            checks["pin_vs_sharpa_pin_hole_mm"] = float(np.hypot(x + r_pcd, y))          # Sharpa's pin hole is at (-25, 0)
            hardware.append(f"1x dowel pin 6 x {int(pin['depth_mm'] + t + 4)} (m6), through the plate (6 H7) into both flanges")
        cb = iface["centering"]
        part = part.union(cq.Workplane("XY").circle(cb["recess_diameter_mm"] / 2 - 0.05).extrude(1.5).translate((0, 0, -t - 1.5)))
        notes.append(f"arm side: {cb['recess_diameter_mm'] - 0.1:g} mm boss, 1.5 mm high, into the flange's {cb['recess_diameter_mm']} mm recess")
    else:
        # different bolt circles: use the arm's holes that keep clear of the Sharpa M6 threads
        used = [a for a in arm_angles if min(abs((a - b + 180) % 360 - 180) for b in sharpa_angles) > 20]
        r_arm = arm_bolts["pcd_mm"] / 2
        for a in used:
            x, y = r_arm * np.cos(np.radians(a)), r_arm * np.sin(np.radians(a))
            part = part.cut(cq.Workplane(obj=cyl(4.5, (x, y, -t - 5), (x, y, 5))))
            part = part.cut(cq.Workplane(obj=cyl(8.0, (x, y, -4.6), (x, y, 5))))            # counterbore for the head
        for a in sharpa_angles:
            x, y = r_pcd * np.cos(np.radians(a)), r_pcd * np.sin(np.radians(a))
            part = part.cut(cq.Workplane(obj=cyl(5.0, (x, y, -t - 5), (x, y, 5))))            # M6 tap drill
        gap = min(2 * np.sin(np.radians(abs((a - b + 180) % 360 - 180)) / 2) * (r_arm + r_pcd) / 2 for a in used for b in sharpa_angles)
        checks["m4_counterbore_to_m6_thread_wall_mm"] = float(gap - 4.0 - 3.0)
        checks["arm_holes_used"] = f"{len(used)} of {arm_bolts['count']}"
        hardware.append(f"{len(used)}x M4 x {int(t - 4.6 + 6)} socket head cap screw, head sunk 4.6 mm (plate to the arm's wrist)")
        hardware.append("4x M6 x 10 socket head cap screw (Sharpa flange 4 mm + 6 mm into the plate's M6 threads; tap the four 5.0 mm holes M6)")
        notes.append("no dowel pin: the Sharpa flange's pin hole would break into an M4 counterbore")

    # ---- camera screws: 2 x M3 from the back of the tab, heads sunk so that 3 mm of thread enters the camera
    insert, head_h = 3.0, 3.0
    for sgn in (-1, 1):
        p = back + right * sgn * D405["m3_spacing"] / 2
        part = part.cut(cq.Workplane(obj=cyl(3.4, p - view * (tab_t + 3), p + view * 3)))
        part = part.cut(cq.Workplane(obj=cyl(6.2, p - view * (tab_t + 3), p - view * (tab_t - head_h))))
    web = tab_t - head_h
    screw_len = web + insert
    hardware.append(f"2x M3 x {screw_len:g} socket head cap screw into the D405's rear threads (head sunk {head_h:g} mm; "
                    f"{insert:g} mm of thread in the camera, its limit is {D405['m3_insert']:g} mm)")
    # ---- cable ties on the tab's edge that faces the hand's thumb side (the D405's USB socket is on that side)
    for dz in (-1, 1):
        p = back - right * (D405["size"][0] / 2 + 4.5) + down * dz * 13.0
        part = part.cut(cq.Workplane(obj=cyl(3.2, p - view * (tab_t + 3), p + view * 3)))
    notes.append("the D405's USB 3 Micro-B socket faces the thumb side: use a right-angle plug with its two M2 locking screws "
                 "and tie the cable to the two 3.2 mm holes beside it")

    solid = part.val()
    vol = solid.Volume()
    bb = solid.BoundingBox()
    info = {
        "arm": arm, "frame": "Sharpa flange frame F (z towards the hand, x out of the palm); palm frame = F + 29.5 mm in z",
        "plate_thickness_mm": t, "plate_diameter_mm": 2 * plate_r, "flat": bool(flat), "tab_mm": [tab_w, tab_h, tab_t],
        "bounding_box_mm": [[bb.xmin, bb.ymin, bb.zmin], [bb.xmax, bb.ymax, bb.zmax]],
        "volume_cm3": vol / 1000, "mass_kg": vol * AL_DENSITY / 1000, "material": "aluminium 6061 (or SLS PA12, about 0.37 x the mass)",
        "camera_optical_in_F_mm": {"pos": o.round(3).tolist(), "x_axis": right.round(6).tolist(), "z_axis": view.round(6).tolist()},
        "camera_housing_centre_in_F_mm": cam_centre.round(3).tolist(),
        "camera_lowest_point_from_hand_axis_mm": float(max(abs((cam_centre + R @ (np.array([a, b, c]) * np.array(D405['size']) / 2))[0])
                                                          for a in (-1, 1) for b in (-1, 1) for c in (-1, 1))),
        "hardware": hardware, "notes": notes, "checks": checks,
    }
    return part, info


def clearances(info):
    """Camera housing against the Sharpa flange, wrist and palm meshes (their own frames -> F)."""
    import trimesh
    from scene import Scene
    up = C.VENDOR / "sharpa_upstream" / "right_sharpa_wave" / "meshes"
    R = C.frame_from_axes(x=info["camera_optical_in_F_mm"]["x_axis"], z=info["camera_optical_in_F_mm"]["z_axis"])
    T = C.make_T(R, np.array(info["camera_housing_centre_in_F_mm"]))
    half = np.array(D405["size"]) / 2
    out = {}
    for name, f, dz in (("sharpa_flange", "flange_B.STL", 0.0), ("sharpa_wrist", "wrist_B.STL", 0.5),
                        ("palm", "right_hand_C_MC_visual_.STL", SHARPA_FLANGE_TO_PALM)):
        m = trimesh.load(up / f, process=False)
        P, _ = trimesh.sample.sample_surface(m, 20000, seed=0)
        P = np.concatenate([np.asarray(P), np.asarray(m.vertices)]) * 1000 + [0, 0, dz]
        out[f"camera_to_{name}_mm"] = Scene.points_to_box(P, T, half)
    return out


def export(part, info, out):
    out.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(part, str(out / "d405_mount.step"))
    cq.exporters.export(cq.Workplane(obj=part.val().scale(0.001)), str(out / "d405_mount.stl"), tolerance=1e-5, angularTolerance=0.1)
    import pymupdf
    for name, d in (("top", (0, 0, 1)), ("side", (0, -1, 0)), ("iso", (1.0, -1.2, 0.9))):
        svg = out / f"d405_mount_{name}.svg"
        cq.exporters.export(part, str(svg), opt={"projectionDir": d, "showAxes": True, "width": 900, "height": 700,
                                                  "marginLeft": 40, "marginTop": 40, "showHidden": True, "strokeWidth": 0.35})
        try:
            pymupdf.open(str(svg))[0].get_pixmap(dpi=110).save(str(out / f"d405_mount_{name}.png"))
        except Exception as e:                                   # the drawing is a convenience only
            print(f"[mount] no PNG for {name}: {e}")
    json.dump(info, open(out / "d405_mount.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    args = ap.parse_args()
    part, info = build(args.arm)
    info["checks"].update(clearances(info))
    out = C.BUILD / "mount" / args.arm
    export(part, info, out)
    print(f"[mount] {args.arm}: {out}/d405_mount.step, .stl, .json")
    print(f"  plate {info['plate_diameter_mm']:.0f} mm x {info['plate_thickness_mm']:g} mm, tab {info['tab_mm']}, flat: {info['flat']}, "
          f"{info['volume_cm3']:.1f} cm3, {info['mass_kg'] * 1000:.0f} g in aluminium")
    print(f"  camera's far edge is {info['camera_lowest_point_from_hand_axis_mm']:.1f} mm from the hand's axis")
    for k, v in info["checks"].items():
        print(f"  {k}: {v if isinstance(v, str) else round(v, 2)}")
    for h in info["hardware"]:
        print("  -", h)


if __name__ == "__main__":
    main()
