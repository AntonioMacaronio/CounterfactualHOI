"""Build a browsable HTML gallery of one videogen run (source clip next to its generated edits).

  python -m videogen.build_html outputs/minimax_h3/real_20261119 [--config configs/videogen/real_20261119.json]

Reads RUN/<clip>/<mode>/<variant>/seed<k>.mp4 (+ .prompt.txt, _compare.mp4), RUN/sources/<clip>.mp4 and the
attempt logs in RUN/manifests/*.jsonl (or RUN/manifest.jsonl), and writes RUN/index.html. Paths are relative, so the
page works from file:// and travels with the folder (e.g. after `aws s3 sync` of the run). One tab per task (pp/prp),
one row per clip. A row plays all its videos from the start in sync; single videos play on hover. A ✓/✗/? rating
per video is kept in the browser's localStorage, and "Export ratings" downloads it as JSON. Stdlib only.
"""

import argparse
import html
import json
import re
from pathlib import Path

from .group_prompts import TYPE_A_EDITS, TYPE_B_OBJECTS
from .prompts import MODES

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_TABS = {"pick_place": "pp", "pick_rotate_place": "prp"}
SKIP_DIRS = {"sources", "logs", "manifests"}


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def load_manifest(run: Path) -> dict:
    recs = {}
    files = sorted((run / "manifests").glob("*.jsonl")) + [run / "manifest.jsonl"]
    for f in files:
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                recs[(r["clip"], r["mode"], r["variant"], r["seed"])] = r  # last attempt wins
    return recs


def edit_label(mode: str, variant: str) -> tuple[str, str]:
    table = {"type_a": TYPE_A_EDITS, "type_b": TYPE_B_OBJECTS}.get(mode)
    if table and variant in table:
        return table[variant]
    return variant, ""


def collect(run: Path, cfg: dict) -> list[dict]:
    manifest = load_manifest(run)
    cfg_variants = {(c, v["id"]): v for c, cl in cfg.get("clips", {}).items() for v in cl.get("variants", [])}
    clips = {}
    for mp4 in sorted(run.glob("*/*/*/seed*.mp4")):
        if mp4.name.endswith("_compare.mp4") or mp4.parts[-4] in SKIP_DIRS:
            continue
        clip, mode, variant = mp4.parts[-4], mp4.parts[-3], mp4.parts[-2]
        seed = int(re.fullmatch(r"seed(\d+)\.mp4", mp4.name).group(1))
        rec = manifest.get((clip, mode, variant, seed), {})
        v = cfg_variants.get((clip, variant), {})
        if v.get("target"):  # catalog swap (videogen/catalog.py): short name + full description
            category, text = f'{v.get("cls", variant)} ({v.get("category", "")})', v["target"]
        else:
            category, text = edit_label(mode, variant)
        prompt_file = mp4.with_suffix(".prompt.txt")
        compare = mp4.with_name(f"seed{seed}_compare.mp4")
        info = rec.get("output_info") or {}
        clips.setdefault(clip, []).append({
            "key": f"{clip}/{mode}/{variant}/seed{seed}",
            "mode": mode, "variant": variant, "seed": seed, "category": category, "text": text,
            "video": mp4.relative_to(run).as_posix(),
            "compare": compare.relative_to(run).as_posix() if compare.exists() else None,
            "prompt": prompt_file.read_text() if prompt_file.exists() else "",
            "infer_s": rec.get("inference_time_s"), "duration": info.get("duration"),
        })
    out = []
    mode_rank = {m: i for i, m in enumerate(MODES)}
    for clip in sorted(clips, key=natural_key):
        c = cfg.get("clips", {}).get(clip, {})
        task = c.get("task") or next((r.get("task") for k, r in manifest.items() if k[0] == clip), None)
        src = run / "sources" / f"{clip}.mp4"
        out.append({
            "clip": clip, "task": task, "tab": TASK_TABS.get(task, task or "other"), "object": c.get("object", ""),
            "source": src.relative_to(run).as_posix() if src.exists() else None,
            "edits": sorted(clips[clip], key=lambda e: (mode_rank.get(e["mode"], 99), natural_key(e["variant"]), e["seed"])),
        })
    return out


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; margin: 24px; color: #222; background: #fafafa; }
  h1 { font-size: 20px; margin: 0 0 4px; } .sub { color: #555; margin: 0 0 16px; max-width: 1100px; font-size: 14px; }
  .bar { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; margin-bottom: 16px; font-size: 14px; }
  .tab { padding: 6px 14px; border: 1px solid #bbb; border-radius: 6px; background: #fff; cursor: pointer; }
  .tab.active { background: #2d5be3; border-color: #2d5be3; color: #fff; }
  button.small { padding: 3px 9px; border: 1px solid #bbb; border-radius: 5px; background: #fff; cursor: pointer; font-size: 13px; }
  section { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin-bottom: 18px; }
  section h2 { font-size: 16px; margin: 0 8px 10px 0; display: inline-block; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(var(--w), 1fr)); gap: 10px; margin-top: 6px; }
  figure { margin: 0; border: 1px solid #e3e3e3; border-radius: 6px; overflow: hidden; background: #fcfcfc; }
  figure.source { border: 2px solid #2d5be3; }
  figure.good { border: 2px solid #1d9a4b; } figure.bad { border: 2px solid #d33; opacity: .75; } figure.unsure { border: 2px solid #e0a100; }
  video { width: 100%; display: block; background: #000; aspect-ratio: 16 / 9; }
  figure.wide { grid-column: span 2; } figure.wide video { aspect-ratio: 32 / 9; }
  figcaption { padding: 6px 8px; font-size: 12.5px; line-height: 1.35; }
  .id { font-weight: 600; } .muted { color: #777; } .text { color: #444; }
  .rate { display: flex; gap: 4px; margin-top: 5px; } .rate button { flex: 1; padding: 2px 0; border: 1px solid #ccc; border-radius: 4px; background: #fff; cursor: pointer; }
  .rate button.on[data-r=good] { background: #1d9a4b; color: #fff; } .rate button.on[data-r=bad] { background: #d33; color: #fff; }
  .rate button.on[data-r=unsure] { background: #e0a100; color: #fff; }
  details { margin-top: 4px; } details pre { white-space: pre-wrap; font-size: 11.5px; max-height: 260px; overflow: auto; background: #f4f4f4; padding: 6px; border-radius: 4px; }
</style></head><body>
<h1>__TITLE__</h1>
<p class="sub">Each row is one real clip (blue border) followed by its generated edits. <b>Play row</b> restarts every
video in the row together, so you can compare motion frame for frame; hovering plays a single video. <b>Compare</b>
swaps each card to the side-by-side source | generated clip. Ratings stay in this browser (localStorage) until you export them.</p>
<div class="bar">
  <span id="tabs"></span>
  <span id="modes"></span>
  <label><input type="checkbox" id="compare"> compare (source | generated)</label>
  <label>speed <select id="speed"><option>0.5</option><option selected>1</option><option>1.5</option><option>2</option></select></label>
  <label>size <input type="range" id="size" min="220" max="560" value="300"></label>
  <label>show <select id="filter"><option value="all">all</option><option value="unrated">unrated</option>
    <option value="good">good</option><option value="bad">bad</option><option value="unsure">unsure</option></select></label>
  <button class="small" id="export">Export ratings</button> <span id="counts" class="muted"></span>
</div>
<div id="rows"></div>
<script>
const DATA = __DATA__;
const RUN = __RUN__;
const STORE = "videogen-ratings:" + RUN;
const ratings = JSON.parse(localStorage.getItem(STORE) || "{}");
const tabs = [...new Set(DATA.map(c => c.tab))];
const modes = [...new Set(DATA.flatMap(c => c.edits.map(e => e.mode)))];
const MODE_NAMES = {type_a: "Type A (geometry)", type_b: "Type B (object swap)", prism: "PRISM", identity: "identity", object: "object", object_hand: "object+hand"};
let tab = tabs[0], shownModes = new Set(modes);
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));

function saveRatings() { localStorage.setItem(STORE, JSON.stringify(ratings)); counts(); }
function counts() {
  const all = DATA.flatMap(c => c.edits), n = r => all.filter(e => ratings[e.key] === r).length;
  $("#counts").textContent = `${all.length} videos: ${n("good")} good, ${n("bad")} bad, ${n("unsure")} unsure, ${all.length - n("good") - n("bad") - n("unsure")} unrated`;
}
function card(e) {
  const src = $("#compare").checked && e.compare ? e.compare : e.video, r = ratings[e.key] || "";
  const meta = [e.duration ? e.duration.toFixed(1) + " s" : "", e.infer_s ? Math.round(e.infer_s) + " s gen" : "", "seed " + e.seed].filter(Boolean).join(" · ");
  const wide = $("#compare").checked && e.compare ? " wide" : "";
  return `<figure class="${r}${wide}" data-key="${esc(e.key)}"><video muted loop playsinline preload="metadata" src="${esc(src)}#t=0.05"></video>
    <figcaption><span class="id">${esc(e.variant)}</span> · ${esc(e.category)} <span class="muted">(${esc(MODE_NAMES[e.mode] || e.mode)})</span>
    <div class="text">${esc(e.text)}</div><div class="muted">${esc(meta)}</div>
    <div class="rate">${["good", "bad", "unsure"].map(x => `<button data-r="${x}" class="${r === x ? "on" : ""}">${{good: "✓", bad: "✗", unsure: "?"}[x]}</button>`).join("")}</div>
    <details><summary class="muted">prompt</summary><pre>${esc(e.prompt)}</pre></details></figcaption></figure>`;
}
function render() {
  const f = $("#filter").value;
  $("#rows").innerHTML = DATA.filter(c => c.tab === tab).map(c => {
    const edits = c.edits.filter(e => shownModes.has(e.mode) && (f === "all" || (f === "unrated" ? !ratings[e.key] : ratings[e.key] === f)));
    const source = c.source && !$("#compare").checked ? `<figure class="source"><video muted loop playsinline preload="metadata" src="${esc(c.source)}#t=0.05"></video>
      <figcaption><span class="id">source</span> · ${esc(c.object)}<div class="muted">real clip as sent to H3</div></figcaption></figure>` : "";
    return `<section><h2>${esc(c.clip)}</h2><span class="muted">${esc(c.object)} · ${esc(c.task || "")} · ${c.edits.length} edits</span>
      <button class="small play">▶ play row</button> <button class="small pause">⏸ pause</button>
      <div class="grid">${source}${edits.map(card).join("")}</div></section>`;
  }).join("");
  document.querySelectorAll("video").forEach(v => {
    v.playbackRate = +$("#speed").value;
    v.addEventListener("mouseenter", () => v.play()); v.addEventListener("mouseleave", () => { if (!v.closest("section").dataset.playing) v.pause(); });
  });
}
document.addEventListener("click", ev => {
  const b = ev.target.closest("button"); if (!b) return;
  const sec = b.closest("section");
  if (b.classList.contains("play")) {
    const vids = [...sec.querySelectorAll("video")]; sec.dataset.playing = 1;
    vids.forEach(v => { v.pause(); v.currentTime = 0; }); Promise.all(vids.map(v => v.play().catch(() => {})));
  } else if (b.classList.contains("pause")) { delete sec.dataset.playing; sec.querySelectorAll("video").forEach(v => v.pause()); }
  else if (b.dataset.r) {
    const fig = b.closest("figure"), k = fig.dataset.key, r = ratings[k] === b.dataset.r ? undefined : b.dataset.r;
    if (r) ratings[k] = r; else delete ratings[k];
    fig.className = (r || "") + (fig.classList.contains("wide") ? " wide" : ""); fig.querySelectorAll(".rate button").forEach(x => x.classList.toggle("on", x.dataset.r === r)); saveRatings();
  }
});
$("#tabs").innerHTML = tabs.map(t => `<button class="tab${t === tab ? " active" : ""}" data-tab="${esc(t)}">${esc(t)}</button>`).join(" ");
$("#tabs").addEventListener("click", ev => { const t = ev.target.dataset.tab; if (!t) return; tab = t;
  document.querySelectorAll(".tab").forEach(x => x.classList.toggle("active", x.dataset.tab === t)); render(); });
$("#modes").innerHTML = modes.map(m => `<label><input type="checkbox" data-mode="${esc(m)}" checked> ${esc(MODE_NAMES[m] || m)}</label>`).join(" ");
$("#modes").addEventListener("change", ev => { const m = ev.target.dataset.mode; ev.target.checked ? shownModes.add(m) : shownModes.delete(m); render(); });
["#compare", "#filter"].forEach(s => $(s).addEventListener("change", render));
$("#speed").addEventListener("change", () => document.querySelectorAll("video").forEach(v => v.playbackRate = +$("#speed").value));
$("#size").addEventListener("input", () => document.documentElement.style.setProperty("--w", $("#size").value + "px"));
document.documentElement.style.setProperty("--w", "300px");
$("#export").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify({run: RUN, exported: new Date().toISOString(), ratings}, null, 1)], {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `ratings_${RUN}.json`; a.click();
});
render(); counts();
</script></body></html>
"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path, help="run directory, e.g. outputs/minimax_h3/real_20261119")
    p.add_argument("--config", type=Path, default=None,
                   help="clip config for object names/tasks (default: configs/videogen/<run name>.json if it exists)")
    p.add_argument("--title", default=None)
    args = p.parse_args(argv)
    cfg_path = args.config or REPO_ROOT / "configs/videogen" / f"{args.run.name}.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    data = collect(args.run, cfg)
    if not data:
        raise SystemExit(f"no generated videos under {args.run}")
    title = args.title or f"MiniMax H3 edits: {args.run.name}"
    page = (PAGE.replace("__TITLE__", html.escape(title))
                .replace("__RUN__", json.dumps(args.run.name))
                .replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    out = args.run / "index.html"
    out.write_text(page)
    n = sum(len(c["edits"]) for c in data)
    print(f"{n} videos from {len(data)} clips -> {out}")


if __name__ == "__main__":
    main()
