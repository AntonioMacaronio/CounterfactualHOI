"""Build one HTML page with offline viser viewers of the hand-object reconstructions of the real videos.

  python scripts/choir/build_choir_html.py [--method choir|fprecon|refine] [--out DIR] [--index-only]   (sam3d env)

--method choir (default, -> ../outputs/choir_viser): Open-CHOIR's final output (scripts/choir/view_choir.py).
--method fprecon (-> ../outputs/fprecon_viser): FoundationPose on hand-scaled Depth Pro 2 depth
(scripts/fprecon/view_fprecon.py, runs in $PROJECT_STORAGE_ROOT/runs/fprecon/<VID>).
--method refine (-> ../outputs/refine_viser): the fprecon result after the agentic refinement (scripts/refine, runs
in $PROJECT_STORAGE_ROOT/runs/refine/<VID>, or runs/refine_<REFINE_TAG> -> refine_viser_<REFINE_TAG>); captions
from its summary.json (scripts/refine/summarize.py), with its before/after QC sheet and the agent's report.
Writes OUT/<VID>.html per clip (hand + object + frame-0 scene points, gravity aligned; works from file://, no
server), OUT/<VID>.json (QC numbers) and OUT/index.html with two tabs, "pp" and "prp", each a grid of that set's
five clips. Only the active tab's iframes are loaded (each is a WebGL context). Clips without a finished
reconstruction get a placeholder panel; re-run once they finish.
"""

import argparse
import html
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
CHOIR_ROOT = os.environ.get("CHOIR_ROOT", "/data/sky/projects/CounterfactualHOI/build/CHOIR")
FPRECON_RUNS = os.path.join(os.environ.get("PROJECT_STORAGE_ROOT", "/data/sky/projects/CounterfactualHOI"), "runs", "fprecon")
REFINE_TAG = "_" + os.environ["REFINE_TAG"] if os.environ.get("REFINE_TAG") else ""
REFINE_RUNS = os.path.join(os.path.dirname(FPRECON_RUNS), "refine" + REFINE_TAG)
sys.path.insert(0, os.path.join(HERE, "..", "daid"))
from build_recon_html import camera_query  # noqa: E402

OBJECTS = ["small metal case", "blue tape roll", "black spatula", "gray mug", "dish soap bottle"]
SETS = ["pp", "prp"]
INTRO = "One tab per recording set; the same five objects in each (small metal case, blue tape roll, black spatula, gray mug, dish soap bottle). "
METHODS = {
    "choir": dict(
        title="Open-CHOIR 4D hand–object reconstructions of the real videos",
        sub=INTRO + """Each viewer shows CHOIR's final contact-optimized MANO hand (blue) and
SAM3D object (orange), the frame-0 depth points of the scene (hand/object pixels removed) and the real camera's
frustum with the current video frame. The scene is rotated so GeoCalib's gravity points down (grid = table height);
the depth points are rescaled so the object rests on the table in the first frame, so where it ends up relative to
the table at the end shows CHOIR's depth drift."""),
    "fprecon": dict(
        title="FoundationPose + Depth Pro 2 4D hand–object reconstructions of the real videos",
        sub=INTRO + """Pipeline: Depth Pro 2 metric video depth, scaled so it agrees with the
Dyn-HaMR MANO hand (the hand's known size is the metric reference); per-frame SAM3 masks of the object and hand
(text prompts); the object's textured SAM3D mesh (from the first frame), sized and tracked through the video by
FoundationPose on that depth; the hand moved per frame onto the depth map. Each viewer shows the hand (blue), the
textured object, the frame-0 depth points of the scene (hand/object pixels removed) and the real camera's frustum
with the current video frame. Hand, object and table come from the same metric depth; the scene is rotated so
GeoCalib's gravity points down (grid = table height). "QC frames"
shows 12 video frames per clip with the tracked object rendered (orange), its SAM3 mask (green) and the hand (blue)."""),
    "refine": dict(
        title="Agent-refined 4D hand–object reconstructions of the real videos",
        sub=INTRO + """The FoundationPose + Depth Pro 2 reconstructions after agentic refinement.
An automatic gradient-free fit (hand to its keypoints, mask and depth; object to its mask and depth; each finger's
flexion with no hand triangle through the object's surface) is followed by Claude (Opus 5.5), which looks at
overlays on the video, the 3D scene from any side (hand red where it passes through the object, green where it
touches it) and motion plots, labels the motion phases, and fixes what a per-frame fit cannot: contact while the
object is held, fingers through it, a still object at rest, jumps. Every edit is a new version, checked before /
after. A blind A/B verifier then compares each changed range with the automatic fit (overlays, 3D views from three
sides, motion plot) and keeps the fit where the agent's edit loses. Captions: medians before → after. "QC frames"
shows, per phase, the initial and refined overlays and the 3D scene from behind and from the left; "report" is
the agent's own account of what it changed."""),
}


def clips(s):
    return [(f"{s}_anthony_real_{i + 1}", obj) for i, obj in enumerate(OBJECTS)]


def panel(out_dir, vid, obj, missing="no reconstruction yet"):
    title = f"{vid} · {obj}"
    viewer = os.path.join(out_dir, f"{vid}.html")
    if not os.path.exists(viewer):
        return f"""    <figure class="missing"><div class="ph">{missing}</div>
      <figcaption><b>{html.escape(title)}</b></figcaption></figure>"""
    qc = json.load(open(os.path.join(out_dir, f"{vid}.json")))
    tilt = qc["table_tilt_vs_gravity_deg"]
    cap = (f"{qc['frames']} frames @ {qc['fps']:.0f} fps · up = GeoCalib gravity (camera pitch "
           f"{qc['gravity_pitch_deg']:.0f}°), " + ("no table plane found" if tilt is None else
           f"{tilt:.1f}° from the table plane in the depth points{' ⚠' if tilt > 10 else ''}"))
    if qc.get("object_height_above_table_first_last_m"):
        h0, h1 = (100 * h for h in qc["object_height_above_table_first_last_m"])
        if qc.get("method") == "foundationpose+depthpro2":
            cap += f" · object bottom vs table: start {h0:+.1f} cm, end {h1:+.1f} cm{' ⚠' if max(abs(h0), abs(h1)) > 3 else ''}"
        else:
            cap += f" · object ends {h1:.0f} cm {'above' if h1 >= 0 else 'below'} the table{' ⚠' if abs(h1) > 5 else ''}"
    if qc.get("iou_first_last"):
        cap += f" · mask IoU first/last {qc['iou_first_last'][0]:.2f}/{qc['iou_first_last'][1]:.2f}"
    summ = os.path.join(out_dir, "summary", f"{vid}.json")
    if os.path.exists(summ):                      # refine: medians before -> after
        sm = json.load(open(summ))
        ba = lambda part, k, d=2: (f"{sm[part][k][0]:.{d}f} → {sm[part][k][1]:.{d}f}" if None not in sm[part][k] else "—")
        cap = (f"{sm['frames']} frames · hand keypoint error {ba('all', 'hand_kp_px', 1)} px · object mask IoU "
               f"{ba('all', 'iou')} (held {ba('held', 'iou')}) · penetration while held {ba('held', 'penetration_cm')} cm · "
               + (f"hand triangles through the object while held {ba('held', 'n_cross', 0)} · " if "n_cross" in sm["held"] else "") +
               f"flagged frames {sm['flagged_frames'][0]} → {sm['flagged_frames'][1]} · verifier kept "
               f"{len(sm['verifier']) - sm['reverted']}/{len(sm['verifier'])} chunks · {sm['n_versions']} versions in "
               f"{sm['agent_minutes']:.0f} min")
        if os.path.exists(os.path.join(out_dir, "reports", f"{vid}.html")):
            cap += f' · <a href="reports/{vid}.html" target="_blank">report</a>'
    if os.path.exists(os.path.join(out_dir, "qc", f"{vid}.jpg")):
        cap += f' · <a href="qc/{vid}.jpg" target="_blank">QC frames</a>'
    src = f"{vid}.html" + camera_query(viewer)
    return f"""    <figure>
      <iframe data-src="{html.escape(src)}" title="{html.escape(title)}"></iframe>
      <figcaption><b>{html.escape(title)}</b><br>{cap}</figcaption>
    </figure>"""


def write_index(out_dir, method="choir"):
    meta = METHODS[method]
    tabs = "\n".join(f'  <button class="tab" data-tab="{s}">{s}</button>' for s in SETS)
    bodies = "\n".join(f'<section class="grid" id="tab-{s}">\n' + "\n".join(panel(out_dir, v, o) for v, o in clips(s))
                       + "\n</section>" for s in SETS)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CounterfactualHOI: {meta["title"]}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; color: #222; background: #fafafa; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  p.sub {{ margin: 0 0 14px; color: #555; font-size: 14px; max-width: 1150px; }}
  .tabs {{ display: flex; gap: 6px; margin-bottom: 14px; }}
  .tab {{ font: inherit; font-size: 15px; padding: 6px 22px; border: 1px solid #bbb; border-radius: 6px;
          background: #fff; cursor: pointer; }}
  .tab.active {{ background: #2d5be3; border-color: #2d5be3; color: #fff; }}
  .grid {{ display: none; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
  .grid.active {{ display: grid; }}
  figure {{ margin: 0; background: #fff; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }}
  iframe {{ width: 100%; height: 480px; border: 0; display: block; }}
  .ph {{ height: 480px; display: flex; align-items: center; justify-content: center; color: #999; }}
  figcaption {{ padding: 8px 12px 10px; font-size: 13px; color: #444; border-top: 1px solid #eee; }}
  @media (max-width: 1000px) {{ .grid.active {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>{meta["title"]}</h1>
<p class="sub">{meta["sub"]}
Drag to orbit, scroll to zoom; the playback bar is at the bottom of each viewer.</p>
<div class="tabs">
{tabs}
</div>
{bodies}
<script>
  function show(name) {{
    document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
    document.querySelectorAll('.grid').forEach(g => {{
      const on = g.id === 'tab-' + name;
      g.classList.toggle('active', on);
      // Load this tab's viewers, unload the other tab's (keeps the number of live WebGL contexts at 5).
      g.querySelectorAll('iframe').forEach(f => {{
        if (on && !f.getAttribute('src')) f.src = f.dataset.src;
        if (!on && f.getAttribute('src')) f.removeAttribute('src');
      }});
    }});
    history.replaceState(null, '', '#' + name);
  }}
  document.querySelectorAll('.tab').forEach(b => b.addEventListener('click', () => show(b.dataset.tab)));
  show(location.hash.slice(1) === 'prp' ? 'prp' : 'pp');
</script>
</body>
</html>
"""
    open(os.path.join(out_dir, "index.html"), "w").write(page)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=sorted(METHODS), default="choir")
    ap.add_argument("--out", default=None, help="default: ../outputs/{choir,fprecon,refine}_viser")
    ap.add_argument("--index-only", action="store_true", help="only rewrite index.html")
    args = ap.parse_args()
    args.out = args.out or os.path.join(os.path.dirname(PROJECT_ROOT), "outputs", f"{args.method}_viser"
                                        + (REFINE_TAG if args.method == "refine" else ""))
    os.makedirs(args.out, exist_ok=True)
    if not args.index_only:
        for s in SETS:
            for vid, _obj in clips(s):
                vd = os.path.join(CHOIR_ROOT, "output", vid)
                out_html = os.path.join(args.out, f"{vid}.html")
                if args.method == "choir":
                    ready = os.path.isdir(os.path.join(vd, "stage3", "final", "optimized_meshes", "hand"))
                    cmd = [os.path.join(HERE, "view_choir.py"), "--video-dir", vd, "--export-html", out_html]
                elif args.method == "refine":
                    rd = os.path.join(REFINE_RUNS, vid, "export")
                    ready = os.path.exists(os.path.join(rd, "hand_rigid.npz"))
                    cmd = [os.path.join(HERE, "..", "fprecon", "view_fprecon.py"), "--video-dir", vd, "--run-dir", rd,
                           "--export-html", out_html]
                else:
                    rd = os.path.join(FPRECON_RUNS, vid)
                    ready = os.path.exists(os.path.join(rd, "fpose_qc.json"))
                    cmd = [os.path.join(HERE, "..", "fprecon", "view_fprecon.py"), "--video-dir", vd, "--run-dir", rd,
                           "--export-html", out_html]
                if not ready:
                    print(f"{vid}: no {args.method} reconstruction, skipping")
                    continue
                subprocess.run([sys.executable] + cmd, check=True)
    if args.method == "fprecon":                  # 12-frame overlays (render / masks / hand) next to the viewers (also with --index-only)
        os.makedirs(os.path.join(args.out, "qc"), exist_ok=True)
        for s in SETS:
            for vid, _obj in clips(s):
                src = os.path.join(FPRECON_RUNS, vid, "overlay.jpg")
                if os.path.exists(src):
                    shutil.copyfile(src, os.path.join(args.out, "qc", f"{vid}.jpg"))
    if args.method == "refine":                   # summary numbers, QC sheet and the agent's report per clip
        for sub in ("qc", "summary", "reports"):
            os.makedirs(os.path.join(args.out, sub), exist_ok=True)
        for s in SETS:
            for vid, _obj in clips(s):
                for src, dst in (("qc.jpg", f"qc/{vid}.jpg"), ("summary.json", f"summary/{vid}.json")):
                    if os.path.exists(os.path.join(REFINE_RUNS, vid, src)):
                        shutil.copyfile(os.path.join(REFINE_RUNS, vid, src), os.path.join(args.out, dst))
                rep = os.path.join(REFINE_RUNS, vid, "report.md")
                if os.path.exists(rep):           # as plain text in a page (static hosts serve .md as a download)
                    open(os.path.join(args.out, "reports", f"{vid}.html"), "w").write(
                        f'<!doctype html><meta charset="utf-8"><title>{vid} refinement report</title>'
                        f'<pre style="white-space: pre-wrap; font: 14px/1.45 system-ui, sans-serif; max-width: 1000px; '
                        f'margin: 24px auto">{html.escape(open(rep).read())}</pre>')
    write_index(args.out, args.method)
    print(f"wrote {os.path.join(args.out, 'index.html')}")


if __name__ == "__main__":
    main()
