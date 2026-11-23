"""Page of the FlashDexRetarget results (website tab 5 · Retargeted motions): refined reconstruction (left) vs the
physics policy's Sharpa Wave rollout (right).

  python scripts/flashdex/build_flashdex_html.py [--out ../outputs/flashdex_viser]      (any python3)

OUT holds one exported viewer per clip (scripts/flashdex/export_viser.py) and its .json (tracking metrics).
index.html gets two tabs, pp and prp; each shows one clip at a time: the agent-refined reconstruction's viewer on
the left (--refined-dir, default ../refine_viser_r7/VID.html, relative to OUT) and the FlashDexRetarget rollout
on the right (2 live viewers at a time). Under them: the rollout's numbers, next to the Do-as-I-Do retargeting of the
same reconstruction (../retarget_viser/VID.json: tab 4's physics check), and a table of every clip.
"""

import argparse
import html
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WORKDIR = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OBJECTS = ["small metal case", "blue tape roll", "black spatula", "gray mug", "dish soap bottle"]
SETS = {"pp": "pick and place", "prp": "pick, rotate in the hand, place"}
POS_TH_CM = 5.0


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def daid_progress(d):
    """AdaDex progress of the Do-as-I-Do rollout from its per-frame object position error (position only: its
    json has no per-frame rotation)."""
    e = d.get("obj_pos_err_cm") if d else None
    if not e:
        return None
    bad = [i for i, x in enumerate(e) if x > POS_TH_CM]
    return 1.0 if not bad else bad[0] / len(e)


def caption(r, d):
    cap = []
    if r:
        cap.append(f"<b>FlashDexRetarget</b>: tracks <b>{r['progress']:.0%}</b> of the clip before the object first "
                   f"leaves 5 cm / 20° of the reconstruction ({'whole clip: success' if r['success'] else 'then fails'}; "
                   f"position only, 5 cm: {r.get('progress_pos', float('nan')):.0%}); "
                   f"object error mean {r['obj_pos_err_cm_mean']:.1f} cm / {r['obj_rot_err_deg_mean']:.0f}° "
                   f"(max {r['obj_pos_err_cm_max']:.0f} cm), within 2 cm on {r['frames_within_2cm']:.0%} of frames; "
                   f"fingertips {r.get('tip_err_cm_mean', float('nan')):.1f} cm from the human's"
                   + (f"; policy after {r['env_step'] / 1e6:.0f}M env steps" if r.get("env_step") else "")
                   + (f", trained on the round-{r['source'][-1]} reconstructions" if r.get("source", "")[-1:].isdigit() else ""))
    if d and "obj_pos_err_cm_median" in d:
        p = daid_progress(d)
        cap.append(f"<b>Do-as-I-Do</b> (round-5 reconstruction, tab 4): within 2 cm on {d['frames_within_2cm']:.0%} of frames, median "
                   f"{d['obj_pos_err_cm_median']:.1f} cm / {d['obj_rot_err_deg_median']:.0f}°"
                   + (f", progress {p:.0%} (position only)" if p is not None else ""))
    return "<br>".join(cap)


def status(rows, note):
    """One honest line: which run / round / policy step the rollouts come from, and per-clip progress."""
    done = [(v, r) for v, r, _ in rows if r]
    if not done:
        return "<p class=\"status\"><b>Status</b>: the policy is still training; no rollouts exported yet.</p>"
    r0 = done[0][1]
    run = os.path.basename(os.path.dirname(os.path.dirname(r0.get("archive", "")))) or "?"
    steps = {r.get("env_step") for _, r in done if r.get("env_step") is not None}
    step = f"{max(steps) / 1e6:.0f}M env steps" if steps else "? env steps"
    rnd = r0.get("source", "?").replace("refine_", "round ").replace("round r", "round ")
    prog = ", ".join(f"{v.replace('_anthony_real', '')} {r['progress']:.0%}" for v, r in done)
    mean = sum(r["progress"] for _, r in done) / len(done)
    mean_pos = sum(r.get("progress_pos", 0) for _, r in done) / len(done)
    fdr = sum(bool(r.get("fdr_success")) for _, r in done)
    ok = sum(bool(r["success"]) for _, r in done)
    line = (f"<b>Status</b>: FlashDexRetarget run <code>{html.escape(run)}</code>, trained on the {html.escape(rnd)} "
            f"reconstructions, policy after {step} (of a 300M budget). Progress per clip: {prog} (mean {mean:.0%}; "
            f"{ok}/{len(done)} clips tracked end to end; position-only progress mean {mean_pos:.0%}; FDR's own "
            f"success, time-mean error within 3 cm / 30°, on {fdr}/{len(done)} clips).")
    if note:
        line += " " + note
    return f'<p class="status">{line}</p>'


def table(rows):
    head = ("<tr><th>clip</th><th>FDR progress</th><th>FDR progress (pos.)</th><th>FDR success</th><th>FDR obj err (cm / °)</th>"
            "<th>FDR within 2 cm</th><th>DAID within 2 cm (r5)</th><th>DAID progress (pos., r5)</th></tr>")
    body = []
    for vid, r, d in rows:
        p = daid_progress(d)
        f = lambda x, s: s.format(x) if x is not None else "–"  # noqa: E731
        body.append("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            html.escape(vid.replace("_anthony_real", "")),
            f(r and r["progress"], "{:.0%}"), f(r and r.get("progress_pos"), "{:.0%}"), "yes" if r and r["success"] else ("no" if r else "–"),
            f"{r['obj_pos_err_cm_mean']:.1f} / {r['obj_rot_err_deg_mean']:.0f}" if r else "–",
            f(r and r["frames_within_2cm"], "{:.0%}"), f(d and d.get("frames_within_2cm"), "{:.0%}"), f(p, "{:.0%}")))
    return f'<table class="sum">{head}{"".join(body)}</table>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(WORKDIR, "outputs", "flashdex_viser"))
    ap.add_argument("--daid-dir", default="../retarget_viser")
    ap.add_argument("--note", default="", help="extra status text (HTML), e.g. how far training has got")
    ap.add_argument("--refined-dir", default="../refine_viser_r7", help="left viewers, relative to OUT (the site "
                    "rewrites this path)")
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)
    data, sections, rows = {}, [], []
    for s, title in SETS.items():
        buttons = []
        for k, obj in enumerate(OBJECTS):
            vid = f"{s}_anthony_real_{k + 1}"
            done = os.path.exists(os.path.join(out, f"{vid}.html"))
            left = f"{args.refined_dir}/{vid}.html"
            r = load(os.path.join(out, f"{vid}.json")) if done else None
            d = load(os.path.join(out, args.daid_dir, f"{vid}.json"))
            rows.append((vid, r, d))
            data[vid] = {"left": left, "right": f"{vid}.html" if done else "",
                         "cap": caption(r, d) if done else "policy still training"}
            buttons.append(f'<button class="clip{"" if done else " pending"}" data-vid="{vid}" data-set="{s}">'
                           f'{html.escape(vid.replace("_anthony_real", ""))} · {html.escape(obj)}{"" if done else " (pending)"}</button>')
        sections.append(f"""<section class="cmp" id="tab-{s}">
<div class="clips">{" ".join(buttons)}</div>
<div class="pair">
  <figure><iframe class="left" title="refined"></iframe><figcaption><b>Refined reconstruction</b> (human hand + object): the reference</figcaption></figure>
  <figure><iframe class="right" title="retargeted"></iframe><figcaption><b>FlashDexRetarget</b>: the Sharpa Wave hand in physics simulation, driven by one RL policy trained on all 10 clips (translucent orange: the reconstructed object it tracks; orange lines: the human hand)</figcaption></figure>
</div>
<p class="cap"></p>
</section>""")
    tabs = "\n".join(f'  <button class="tab" data-tab="{s}">{s} · {t}</button>' for s, t in SETS.items())
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CounterfactualHOI: physically plausible retargeted motions (FlashDexRetarget)</title>
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
  p.cap {{ font-size: 14px; color: #333; line-height: 1.5; }}
  p.status {{ font-size: 14px; color: #333; background: #fff7e6; border: 1px solid #f0d9a8; border-radius: 6px;
             padding: 8px 12px; max-width: 1150px; }}
  table.sum {{ border-collapse: collapse; font-size: 13px; margin-top: 18px; background: #fff; }}
  table.sum th, table.sum td {{ border: 1px solid #ddd; padding: 4px 10px; text-align: right; }}
  table.sum td:first-child, table.sum th:first-child {{ text-align: left; }}
  @media (max-width: 1000px) {{ .pair {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>Physically plausible retargeted motions: FlashDexRetarget on the Sharpa Wave hand</h1>
<p class="sub">Each refined reconstruction (left) is retargeted with
<a href="https://holiday-robot.github.io/FlashDexRetarget/">FlashDexRetarget</a>: a kinematic retarget of the human
hand to the Sharpa Wave hand (22 DoF on a floating wrist) gives the reference, and one reinforcement-learning policy,
trained on all 10 clips at once, drives the hand in physics simulation so the simulated object follows the
reconstructed one (right). <b>Progress</b> (from <a href="https://yuxuank.com/AdaDex/">AdaDex</a>) is the share of the
clip tracked before the object first deviates by more than 5 cm or 20° from the reconstruction; a clip succeeds when
the whole of it is tracked. Do-as-I-Do (tab 4) retargets the same reconstructions by sampling-based physics
optimization instead (its numbers are on the round-5 reconstructions). The two viewers play independently: drag each time slider.</p>
{status(rows, args.note)}
<div class="tabs">
{tabs}
</div>
{chr(10).join(sections)}
{table(rows)}
<script>
  const D = {json.dumps(data)};
  const cur = {{}};
  function load(set, on) {{
    const sec = document.getElementById('tab-' + set);
    const l = sec.querySelector('iframe.left'), r = sec.querySelector('iframe.right');
    if (!on) {{ l.removeAttribute('src'); r.removeAttribute('src'); return; }}
    const vid = cur[set], d = D[vid];
    l.src = d.left; if (d.right) r.src = d.right; else r.removeAttribute('src');
    sec.querySelector('p.cap').innerHTML = '<b>' + vid + '</b><br>' + d.cap;
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
    print(f"wrote {os.path.join(out, 'index.html')} ({sum(bool(d['right']) for d in data.values())}/10 clips exported)")


if __name__ == "__main__":
    main()
