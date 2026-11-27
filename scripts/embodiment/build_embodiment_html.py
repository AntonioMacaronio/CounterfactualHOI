"""Page of the robot embodiment (website tab 6): the arm + Sharpa Wave hand + wrist D405 playing every clip.

  python scripts/embodiment/build_embodiment_html.py [--arm vega_1p] [--ik-dir DIR] [--out DIR]
      (venv: $PROJECT_STORAGE_ROOT/venvs/arm; run view_embodiment.sh first, it writes the viewers into OUT)

OUT (default: view_embodiment's output for the same IK result) gets
  index.html     pp / prp tabs; per clip the viewer, its numbers and three stills; a table of every clip; the
                 embodiment's numbers (arm, hand, cameras) and the D405 mount (drawings, STEP file, hardware)
  summary.json   the headline numbers, read by scripts/site/build_site.py for the home page
  mount/         the mount's STEP file and drawings, copied from $PROJECT_STORAGE_ROOT/build/embodiment/mount/<arm>/
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

import numpy as np

import common as C

OBJECTS = ["small metal case", "blue tape roll", "black spatula", "gray mug", "dish soap bottle"]
SETS = {"pp": "pick and place", "prp": "pick, rotate in the hand, place"}


def load(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


def caption(v):
    if not v or "usable_pct" not in v:
        return "no IK result"
    return (f"<b>{v['usable_pct']:.1f}%</b> of the frames usable. Palm moved from the reference's by "
            f"{v['palm_dev_mm']['median']:.1f} mm median, {v['palm_dev_mm']['max']:.0f} mm / {v['palm_dev_deg']['max']:.0f}° at most; "
            f"fingertips within {v['tip_err_mm']['p95']:.1f} mm of the reference's in 95% of frames ({v['tip_err_mm']['max']:.1f} mm at most); "
            f"fingers re-bent by up to {v['finger_change_deg']['max']:.0f}°. Lowest point above the table: arm {v['arm_table_min_mm']:.0f} mm, "
            f"wrist hardware (last arm links, mount, camera) {v['hardware_table_min_mm']:.0f} mm. Hand inside the object at most "
            f"{v['hand_in_object_max_mm']:.1f} mm (the reference's own hand: {v['hand_in_object_max_fdr_mm']:.1f} mm). Fastest arm joint "
            f"{v['speed_ratio_max']:.2f}× its speed limit.")


def table(clips):
    head = ("<tr><th>clip</th><th>usable frames</th><th>palm moved, max</th><th>fingertips off, p95 / max</th>"
            "<th>fingers re-bent, max</th><th>wrist hardware above table, min</th><th>arm above table, min</th></tr>")
    rows = []
    for vid, v in clips.items():
        if not v or "usable_pct" not in v:
            continue
        rows.append(f"<tr><td>{vid.replace('_anthony_real', '')}</td><td>{v['usable_pct']:.1f}%</td>"
                    f"<td>{v['palm_dev_mm']['max']:.0f} mm, {v['palm_dev_deg']['max']:.0f}°</td>"
                    f"<td>{v['tip_err_mm']['p95']:.1f} / {v['tip_err_mm']['max']:.1f} mm</td><td>{v['finger_change_deg']['max']:.0f}°</td>"
                    f"<td>{v['hardware_table_min_mm']:.0f} mm</td><td>{v['arm_table_min_mm']:.0f} mm</td></tr>")
    return f'<table class="sum">{head}{"".join(rows)}</table>'


def spec(arm, cfg, cams, mount):
    w, e = cams["wrist"], cams["exo"]
    ws, es = cams["sensors"][w["sensor"]], cams["sensors"][e["sensor"]]
    T = C.cfg_T(w["pose"])
    centre = (T[:3, 3] + T[:3, :3] @ (np.array(ws["housing"]["centre_mm"]) / 1000)) * 1000
    ep = e["pose"]["pos"]
    roi = e["policy_image"].get("roi")
    rows = [
        ("Arm", f"{html.escape(cfg['description'])}: 7 joints move; chassis, torso, head and left arm are held at one posture"
                if "fixed_joints" in cfg and cfg["fixed_joints"] else html.escape(cfg["description"])),
        ("Hand", "Sharpa Wave, right, 22 joints, on the arm's flange through the camera mount plate "
                 f"(clocked {cfg['hand_mount']['yaw_deg']:.0f}° about the tool axis)"),
        ("Controlled joints", "7 + 22 = 29"),
        ("Wrist camera", f"{ws['name'].split(' (')[0]} on the palm side of the wrist, looking along the fingers (housing centre "
                         f"{centre[0]:.0f} mm below the hand's axis, {-centre[2]:.0f} mm behind the palm's wrist end); "
                         f"{ws['stream']['width']}×{ws['stream']['height']} at {ws['stream']['fps']} fps, depth "
                         f"{w['depth_range_m'][0]:.2f}–{w['depth_range_m'][1]:.2f} m, policy image {w['policy_image']['width']}×{w['policy_image']['height']}"),
        ("Exo camera", f"{es['name'].split(' (')[0]} on a stand at the table's left-front corner, ({ep[0]:.2f}, {ep[1]:.2f}, {ep[2]:.2f}) m, "
                       f"0.90 m from the workspace, 25° down; depth {e['depth_range_m'][0]:.2f}–{e['depth_range_m'][1]:.2f} m, policy image "
                       f"{e['policy_image']['width']}×{e['policy_image']['height']}" + (f" of a {roi[2]}×{roi[3]} window of the stream" if roi else "")),
    ]
    if mount:
        rows.append(("Camera mount", f"one flat plate, {mount['plate_diameter_mm']:.0f} mm × {mount['plate_thickness_mm']:g} mm with a "
                                     f"{mount['tab_mm'][0]:.0f} × {mount['tab_mm'][1]:.0f} mm tab for the camera, "
                                     f"{mount['mass_kg'] * 1000:.0f} g in aluminium"))
    return '<table class="spec">' + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows) + "</table>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="vega_1p")
    ap.add_argument("--dataset", default=str(C.STORAGE / "runs" / "flashdex" / "data_r10_flat"))
    ap.add_argument("--ik-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tag = Path(args.dataset).name + "_reference"
    ik_dir = Path(args.ik_dir) if args.ik_dir else C.RUNS / "ik_pyroki" / args.arm / f"{tag}_fingertips"
    out = Path(args.out) if args.out else C.RUNS / "viewer" / args.arm / ik_dir.name
    cfg, cams = C.load_arm(args.arm), C.load_cameras()
    rep = load(ik_dir / "report.json") or {"clips": {}}
    clips = rep["clips"]
    palm = load(C.RUNS / "ik_pyroki" / args.arm / f"{tag}_palm" / "report.json")               # the same clips, palm held
    roll = load(C.RUNS / "ik_pyroki" / args.arm / f"{tag.replace('_reference', '_rollouts')}_fingertips" / "report.json")
    # the mount
    md = C.BUILD / "mount" / args.arm
    mount = load(md / "d405_mount.json")
    if mount:
        (out / "mount").mkdir(parents=True, exist_ok=True)
        for f in ("d405_mount.step", "d405_mount_iso.png", "d405_mount_top.png", "d405_mount_side.png"):
            if (md / f).exists():
                shutil.copyfile(md / f, out / "mount" / f)
    data, sections = {}, []
    for s, title in SETS.items():
        buttons = []
        for k, obj in enumerate(OBJECTS):
            vid = f"{s}_anthony_real_{k + 1}"
            done = (out / f"{vid}.html").exists()
            data[vid] = {"src": f"{vid}.html" if done else "", "still": f"{vid}.png" if (out / f"{vid}.png").exists() else "",
                         "cap": caption(clips.get(vid)) if done else "not exported"}
            buttons.append(f'<button class="clip{"" if done else " pending"}" data-vid="{vid}" data-set="{s}">'
                           f'{html.escape(vid.replace("_anthony_real", ""))} · {html.escape(obj)}</button>')
        sections.append(f"""<section class="cmp" id="tab-{s}">
<div class="clips">{" ".join(buttons)}</div>
<figure><iframe title="embodiment"></iframe><figcaption class="cap"></figcaption></figure>
<figure class="stills"><img alt="stills"><figcaption>At the middle of the carry: the whole scene, the wrist from its palm
side (the D405 under the wrist, the mount plate between arm and hand), and the scene from where the exo camera stands.</figcaption></figure>
</section>""")
    tabs = "\n".join(f'  <button class="tab" data-tab="{s}">{s} · {t}</button>' for s, t in SETS.items())
    n_full = sum(1 for v in clips.values() if v.get("usable_pct", 0) >= 100)
    frames = rep.get("frames_usable_pct")
    summary = {"arm": args.arm, "description": cfg["description"], "ik_result": ik_dir.name, "clips": len(clips),
               "clips_fully_usable": n_full, "frames_usable_pct": frames,
               "palm_mode": {"clips_fully_usable": palm["clips_fully_usable"], "frames_usable_pct": palm["frames_usable_pct"]} if palm else None,
               "rollouts": {"clips_fully_usable": roll["clips_fully_usable"], "frames_usable_pct": roll["frames_usable_pct"]} if roll else None}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    status = (f"With the palm held exactly where the reference has it, {palm['clips_fully_usable']} of 10 clips are usable in every "
              f"frame ({palm['frames_usable_pct']:.1f}% of the frames): in the others the last arm links, the mount or the camera would be "
              "inside the table. Anchoring the fingertips instead and letting the palm lift and tilt makes "
              f"<b>{n_full} of 10</b> clips usable in every frame (<b>{frames:.1f}%</b> of the frames)") if palm and frames is not None else ""
    if status and roll:
        status += (f". On the rollouts of tab 5's RL policy (the round-10 run at 110M steps, physically simulated, so the wrist "
                   f"jitters and the object can slip) the same solver gives {roll['clips_fully_usable']} of 10 clips "
                   f"({roll['frames_usable_pct']:.1f}% of the frames); those are not shown here")
    status += "." if status else ""
    hw = "".join(f"<li>{html.escape(h)}</li>" for h in (mount or {}).get("hardware", []))
    notes = "".join(f"<li>{html.escape(h)}</li>" for h in (mount or {}).get("notes", []))
    iso = "ISO" in cfg["flange"]["interface"]["standard"]
    adapter = ("The arm's flange is ISO 9409-1-50-4-M6 like the Sharpa's, so the hand's four screws go through the plate "
               "into the arm, and a dowel pin keys both flanges." if iso else
               "On this arm the plate is also the adapter from the wrist's six-hole pattern to the Sharpa's ISO 9409-1-50-4-M6 "
               "flange. The wrist's hole pattern is measured on the vendor's model, not on a drawing: check it on the robot "
               "before machining.")
    mount_html = "" if not mount else f"""
<h2>The wrist camera mount</h2>
<p class="sub">One flat plate between the arm's wrist and the Sharpa flange, with a tab on the palm side; the D405 is
screwed to the tab through its two rear M3 threads, so the whole mount can be cut from {mount['plate_thickness_mm']:g} mm
aluminium sheet. {adapter} <a href="mount/d405_mount.step">STEP file</a> (mm).</p>
<div class="mount"><figure><img src="mount/d405_mount_iso.png" alt="mount, isometric"><figcaption>Isometric (hidden lines grey)</figcaption></figure>
<figure><img src="mount/d405_mount_top.png" alt="mount, top"><figcaption>From the hand's side: the bolt holes to the arm and to the Sharpa flange, and the camera's two M3 screws on the tab</figcaption></figure></div>
<ul>{hw}{notes}</ul>"""
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CounterfactualHOI: robot embodiment ({html.escape(cfg['description'])})</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; color: #222; background: #fafafa; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }} h2 {{ font-size: 18px; margin: 26px 0 6px; }}
  p.sub {{ margin: 0 0 12px; color: #444; font-size: 14px; max-width: 1150px; line-height: 1.5; }}
  p.status {{ font-size: 14px; color: #333; background: #fff7e6; border: 1px solid #f0d9a8; border-radius: 6px; padding: 8px 12px; max-width: 1150px; }}
  .tabs {{ display: flex; gap: 6px; margin: 14px 0; }}
  .tab {{ font: inherit; font-size: 15px; padding: 6px 22px; border: 1px solid #bbb; border-radius: 6px; background: #fff; cursor: pointer; }}
  .tab.active {{ background: #2d5be3; border-color: #2d5be3; color: #fff; }}
  .cmp {{ display: none; }} .cmp.active {{ display: block; }}
  .clips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }}
  .clip {{ font: inherit; font-size: 13px; padding: 4px 12px; border: 1px solid #bbb; border-radius: 14px; background: #fff; cursor: pointer; }}
  .clip.active {{ background: #1f2937; border-color: #1f2937; color: #fff; }} .clip.pending {{ color: #999; }}
  figure {{ margin: 0 0 12px; background: #fff; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; max-width: 1400px; }}
  iframe {{ width: 100%; height: 640px; border: 0; display: block; background: #f3f3f3; }}
  figure img {{ width: 100%; display: block; }}
  figcaption {{ padding: 8px 12px 10px; font-size: 13px; color: #444; border-top: 1px solid #eee; line-height: 1.5; }}
  table {{ border-collapse: collapse; font-size: 13px; background: #fff; margin: 8px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 4px 10px; text-align: left; vertical-align: top; }}
  table.spec th {{ background: #f1f5f9; white-space: nowrap; }} table.sum th {{ background: #f1f5f9; }}
  table.sum td {{ text-align: right; }} table.sum td:first-child {{ text-align: left; }}
  .mount {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; max-width: 1150px; }}
  ul {{ font-size: 14px; max-width: 1150px; }}
  @media (max-width: 1000px) {{ .mount {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>Robot embodiment: {html.escape(cfg['description'].split(',')[0])} arm + Sharpa Wave hand + wrist D405</h1>
<p class="sub">The robot the §1.4 policy controls, playing each clip. Each clip's motion is FlashDexRetarget's
kinematic reference: the Sharpa hand on a floating wrist following the round-10 reconstruction, the motion that tab 5's
RL policy is trained to track. The arm's 7 joint angles come from an inverse kinematics step with collision avoidance
(<a href="https://github.com/chungmin99/pyroki">PyRoki</a>): it keeps the five fingertips where the reference has
them and keeps the arm, the camera mount and the camera off the table and off the robot's own body, within the arm's
joint and speed limits. Where the hardware behind the palm would be inside the table, it lifts and tilts the palm and
re-bends the fingers. <b>Orange ghost</b>: the reference's floating hand; where it differs from the robot's hand, the
solver moved the palm. <b>Blue frustum</b>: the wrist D405, <b>orange frustum</b>: the exo
D435, each with the depth image the policy gets at that step (near is bright, dark red is invalid: out of range or seen
by one imager only). <b>Green frustum</b>: the robot's own point of view, its head camera (the Vega's ZED, nominal
90° × 60° field of view), with a shaded view of the scene from there; it is not a policy input. The checkboxes at the
top right of the viewer hide and show each frustum; the choice carries over when you switch clips. This is kinematic: the changed hand pose has not been simulated again, so whether the object is
still held is not checked here. The robot is the vendors' CAD in their colours, thinned for the browser to within 1 mm
(arm, hand, mount) and 2 mm (the frozen body). Drag to orbit, scroll to zoom; the viewer plays in a loop.</p>
<p class="status">{status}</p>
{spec(args.arm, cfg, cams, mount)}
<div class="tabs">
{tabs}
</div>
{chr(10).join(sections)}
<h2>Every clip</h2>
<p class="sub">A frame is usable when the fingertips are within 3 mm of the reference's, the arm and the wrist
hardware are above the table, the fingers are no deeper in the table and the hand no deeper in the object than the
reference's own (2 mm slack), and no arm joint is faster than its limit. Measured on the exact meshes.</p>
{table(clips)}
{mount_html}
<script>
  const D = {json.dumps(data)};
  const cur = {{}};
  function load(set, on) {{
    const sec = document.getElementById('tab-' + set);
    const f = sec.querySelector('iframe'), im = sec.querySelector('.stills img');
    if (!on) {{ f.removeAttribute('src'); return; }}
    const vid = cur[set], d = D[vid];
    if (d.src) f.src = d.src; else f.removeAttribute('src');
    if (d.still) {{ im.src = d.still; im.parentElement.style.display = ''; }} else im.parentElement.style.display = 'none';
    sec.querySelector('.cap').innerHTML = '<b>' + vid + '</b>: ' + d.cap;
    sec.querySelectorAll('.clip').forEach(c => c.classList.toggle('active', c.dataset.vid === vid));
  }}
  document.querySelectorAll('.clip').forEach(c => c.addEventListener('click', () => {{ cur[c.dataset.set] = c.dataset.vid; load(c.dataset.set, true); }}));
  function show(name) {{
    document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
    document.querySelectorAll('.cmp').forEach(s => {{
      const set = s.id.slice(4), on = set === name;
      s.classList.toggle('active', on);
      if (!cur[set]) cur[set] = Object.keys(D).find(v => v.startsWith(set + '_') && D[v].src) || Object.keys(D).find(v => v.startsWith(set + '_'));
      load(set, on);
    }});
    history.replaceState(null, '', '#' + name);
  }}
  document.querySelectorAll('.tab').forEach(b => b.addEventListener('click', () => show(b.dataset.tab)));
  show(location.hash.slice(1) === 'prp' ? 'prp' : 'pp');
</script>
</body>
</html>
"""
    (out / "index.html").write_text(page)
    print(f"wrote {out / 'index.html'} ({sum(bool(d['src']) for d in data.values())}/10 clips); {n_full}/10 clips fully usable")


if __name__ == "__main__":
    main()
