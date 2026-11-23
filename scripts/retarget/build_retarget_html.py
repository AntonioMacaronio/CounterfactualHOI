"""Page with the retargeting results of the agent-refined reconstructions: refined (left) vs retargeted (right).

  python scripts/retarget/build_retarget_html.py [--out ../outputs/retarget_viser] [--refined-dir ../refine_viser_r5]
      (any python3)

OUT holds one exported viewer per clip (scripts/retarget/export_replay_html.py: the Sharpa hand in MuJoCo Warp,
with the human-hand reference it tracked in orange) and its .json (the physics check: how closely the simulated
object followed the reconstructed one). index.html gets two tabs, pp and prp; each shows one clip at a time, the
agent-refined reconstruction's viewer (--refined-dir, relative to OUT) on the left and the retargeted one on the
right (2 live viewers at a time), with the clip's numbers. Clips not retargeted yet show as pending.
"""

import argparse
import html
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WORKDIR = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OBJECTS = ["small metal case", "blue tape roll", "black spatula", "gray mug", "dish soap bottle"]
SETS = {"pp": "pick and place", "prp": "pick, rotate in the hand, place"}


def caption(out, refined, vid):
    js = os.path.join(out, f"{vid}.json")
    cap = []
    sm = os.path.join(out, refined, "summary", f"{vid}.json")
    if os.path.exists(sm):
        rv = json.load(open(sm)).get("reviewer")
        if rv:
            cap.append(f"reconstruction reviewer: {rv['overall']} ({', '.join(rv.get('votes', []))})")
    if os.path.exists(js):
        r = json.load(open(js))
        if "obj_pos_err_cm_median" in r:
            cap.append(f"<b>physics check</b>: the simulated object stays within 2 cm of the reconstructed one on "
                       f"<b>{r['frames_within_2cm']:.0%}</b> of frames (median error {r['obj_pos_err_cm_median']:.1f} cm, max "
                       f"{r['obj_pos_err_cm_max']:.0f} cm; rotation median {r['obj_rot_err_deg_median']:.0f}°)")
    return " · ".join(cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(WORKDIR, "outputs", "retarget_viser"))
    ap.add_argument("--refined-dir", default="../refine_viser_r5")
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)
    data, sections = {}, []
    for s, title in SETS.items():
        buttons = []
        for k, obj in enumerate(OBJECTS):
            vid = f"{s}_anthony_real_{k + 1}"
            done = os.path.exists(os.path.join(out, f"{vid}.html"))
            data[vid] = {"left": f"{args.refined_dir}/{vid}.html", "right": f"{vid}.html" if done else "",
                         "cap": caption(out, args.refined_dir, vid) if done else "retargeting still running"}
            buttons.append(f'<button class="clip{"" if done else " pending"}" data-vid="{vid}" data-set="{s}">'
                           f'{html.escape(vid.replace("_anthony_real", ""))} · {html.escape(obj)}{"" if done else " (pending)"}</button>')
        sections.append(f"""<section class="cmp" id="tab-{s}">
<div class="clips">{" ".join(buttons)}</div>
<div class="pair">
  <figure><iframe class="left" title="refined"></iframe><figcaption><b>Agent-refined reconstruction</b> (human hand + object)</figcaption></figure>
  <figure><iframe class="right" title="retargeted"></iframe><figcaption><b>Retargeted</b>: Sharpa hand on a UR3 arm in MuJoCo Warp, tracking the reconstruction (orange: the human hand it tracks)</figcaption></figure>
</div>
<p class="cap"></p>
</section>""")
    tabs = "\n".join(f'  <button class="tab" data-tab="{s}">{s} · {t}</button>' for s, t in SETS.items())
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CounterfactualHOI: retargeting the agent-refined reconstructions</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; color: #222; background: #fafafa; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  p.sub {{ margin: 0 0 14px; color: #555; font-size: 14px; max-width: 1150px; }}
  .tabs {{ display: flex; gap: 6px; margin-bottom: 14px; }}
  .tab {{ font: inherit; font-size: 15px; padding: 6px 22px; border: 1px solid #bbb; border-radius: 6px; background: #fff; cursor: pointer; }}
  .tab.active {{ background: #2d5be3; border-color: #2d5be3; color: #fff; }}
  .cmp {{ display: none; }} .cmp.active {{ display: block; }}
  .clips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }}
  .clip {{ font: inherit; font-size: 13px; padding: 4px 12px; border: 1px solid #bbb; border-radius: 14px; background: #fff; cursor: pointer; }}
  .clip.active {{ background: #1f2937; border-color: #1f2937; color: #fff; }}
  .clip.pending {{ color: #999; }}
  .pair {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
  figure {{ margin: 0; background: #fff; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }}
  iframe {{ width: 100%; height: 560px; border: 0; display: block; background: #f3f3f3; }}
  figcaption {{ padding: 8px 12px 10px; font-size: 13px; color: #444; border-top: 1px solid #eee; }}
  p.cap {{ font-size: 14px; color: #333; }}
  @media (max-width: 1000px) {{ .pair {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>Retargeting the agent-refined reconstructions to the Sharpa hand</h1>
<p class="sub">Each agent-refined reconstruction (left) is retargeted with Do-as-I-Do's pipeline: inverse kinematics of
the Sharpa hand (on a UR3 arm) to the wrist and fingertips, then physics optimization in MuJoCo Warp so the
simulated hand moves the simulated object along the reconstructed trajectory (right; the orange hand is the human
hand it tracks). The physics check under each pair says how closely the simulated object followed the reconstruction:
a retargeting that drops or pushes the object fails it. The two viewers play independently: drag each time slider.</p>
<div class="tabs">
{tabs}
</div>
{chr(10).join(sections)}
<script>
  const D = {json.dumps(data)};
  const cur = {{}};
  function load(set, on) {{
    const sec = document.getElementById('tab-' + set);
    const l = sec.querySelector('iframe.left'), r = sec.querySelector('iframe.right');
    if (!on) {{ l.removeAttribute('src'); r.removeAttribute('src'); return; }}
    const vid = cur[set], d = D[vid];
    l.src = d.left; if (d.right) r.src = d.right; else r.removeAttribute('src');
    sec.querySelector('p.cap').innerHTML = '<b>' + vid + '</b> · ' + d.cap;
    sec.querySelectorAll('.clip').forEach(c => c.classList.toggle('active', c.dataset.vid === vid));
  }}
  document.querySelectorAll('.clip').forEach(c => c.addEventListener('click', () => {{ cur[c.dataset.set] = c.dataset.vid; load(c.dataset.set, true); }}));
  function show(name) {{
    document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
    document.querySelectorAll('.cmp').forEach(s => {{
      const set = s.id.slice(4), on = set === name;
      s.classList.toggle('active', on);
      if (!cur[set]) cur[set] = Object.keys(D).find(v => v.startsWith(set + '_') && D[v].right) || Object.keys(D).find(v => v.startsWith(set + '_'));
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
    open(os.path.join(out, "index.html"), "w").write(page)
    print(f"wrote {os.path.join(out, 'index.html')} ({sum(bool(d['right']) for d in data.values())}/10 clips retargeted)")


if __name__ == "__main__":
    main()
