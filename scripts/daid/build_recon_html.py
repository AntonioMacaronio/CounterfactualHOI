"""Build an HTML page embedding the IMG_5710 Do-as-I-Do reconstructions as offline viser viewers.

  python scripts/daid/build_recon_html.py [--out ../outputs/recon_viser]   (sam3d env)

Writes OUT/<name>.html (self-contained viser playback of hand + object + table, ~10 MB each; works from
file://, no server) and OUT/index.html (2x2 grid of iframes with QC captions). Each viewer starts from the
camera saved with "Save camera angle" in the live viewer (RUN/viewer_camera.json), else the default view;
the live viewer re-exports its HTML when you save; re-run with --index-only to refresh the iframe links.
"""

import argparse
import html
import json
import os
import re
import urllib.parse
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
RECON = os.path.join(os.environ.get("PROJECT_STORAGE_ROOT", "/data/sky/projects/CounterfactualHOI"), "runs", "recon")

# name, run dir, object id, fps, title, live-viewer port
EXAMPLES = [
    ("real", "IMG_5710_real", "white_case", 30, "Real reference video: AirPods case", 8081),
    ("apple", "IMG_5710_apple", "apple", 24, "Generated (MiniMax H3): red apple", 8082),
    ("wood_block", "IMG_5710_wood_block", "wooden_block", 24, "Generated (MiniMax H3): wooden block", 8083),
    ("lemon", "IMG_5710_lemon", "lemon", 24, "Generated (MiniMax H3): lemon", 8084),
]


def export(run_dir, obj, fps, out_html):
    subprocess.run([sys.executable, os.path.join(HERE, "view_recon.py"), "--video-dir", run_dir, "--object-id", obj,
                    "--fps", str(fps), "--export-html", out_html], check=True)


def caption(run_dir):
    qc = json.load(open(os.path.join(run_dir, "qc", "qc.json")))["summary"]
    tr = json.load(open(os.path.join(run_dir, "qc", "tracking_eval.json")))["summary"]
    tilt = qc["table_normal_vs_gravity_deg"]
    return (f"table {qc['table_size_m'][0]:.2f}×{qc['table_size_m'][1]:.2f} m · "
            f"silhouette IoU {tr['visible_iou_mean']:.2f} · rotation flips (&gt;20°/frame) "
            f"{100 * tr['frac_steps_over_20deg']:.0f}% · up = GeoCalib gravity, "
            f"{tilt:.1f}° from the table normal{' ⚠' if tilt > 10 else ''}")


def camera_query(viewer_html):
    """initialCamera* params baked into an exported viewer, so index.html can pass them in the iframe URL too
    (the viewer otherwise sets them with history.replaceState, which some browsers block for file:// pages)."""
    m = re.search(r"var d=(\{[^}]*\});", open(viewer_html).read())
    return "?" + urllib.parse.urlencode(json.loads(m.group(1))) if m else ""


def write_index(out_dir):
    panels = []
    for name, run, _obj, _fps, title, _port in EXAMPLES:
        rd = os.path.join(RECON, run)
        src = f"{name}.html" + html.escape(camera_query(os.path.join(out_dir, f"{name}.html")))
        panels.append(f"""    <figure>
      <iframe src="{src}" loading="lazy" title="{html.escape(title)}"></iframe>
      <figcaption><b>{html.escape(title)}</b><br>{caption(rd)}</figcaption>
    </figure>""")
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CounterfactualHOI: Do-as-I-Do reconstructions (IMG_5710)</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; color: #222; background: #fafafa; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  p.sub {{ margin: 0 0 18px; color: #555; font-size: 14px; max-width: 1100px; }}
  .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
  figure {{ margin: 0; background: #fff; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }}
  iframe {{ width: 100%; height: 480px; border: 0; display: block; }}
  figcaption {{ padding: 8px 12px 10px; font-size: 13px; color: #444; border-top: 1px solid #eee; }}
  @media (max-width: 1000px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>Do-as-I-Do 4D hand–object–table reconstructions of IMG_5710</h1>
<p class="sub">Reference clip: the right hand picks up a gray AirPods case, spins it in the palm and sets it down.
The three generated clips replace the case with another object via MiniMax H3 (PRISM-style text-only V2V).
Each viewer shows the MANO hand (HaWoR, blue), the textured SAM3D object (Do-as-I-Do tracking), and the
reconstructed table (fitted plane + point cloud), with the real camera's frustum and video frame.
Drag to orbit, scroll to zoom; the playback bar is at the bottom of each viewer.</p>
<div class="grid">
{chr(10).join(panels)}
</div>
</body>
</html>
"""
    open(os.path.join(out_dir, "index.html"), "w").write(page)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(PROJECT_ROOT), "outputs", "recon_viser"))
    ap.add_argument("--index-only", action="store_true", help="only rewrite index.html")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if not args.index_only:
        for name, run, obj, fps, _title, _port in EXAMPLES:
            export(os.path.join(RECON, run), obj, fps, os.path.join(args.out, f"{name}.html"))
    write_index(args.out)
    print(f"wrote {os.path.join(args.out, 'index.html')}")


if __name__ == "__main__":
    main()
